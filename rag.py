"""Hybrid RAG: embedding-поиск + BM25 keyword-поиск + Reciprocal Rank Fusion.

Архитектура:
    RAGIndex.build()       — нарезает базу на чанки, считает эмбеддинги
    BM25Index.build()      — токенизирует чанки и строит BM25 индекс
    HybridIndex.search()   — embedding + bm25, объединение через RRF
"""
from __future__ import annotations

import os
import re
import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from keywords import extract_keywords, tokenize_for_bm25

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_TARGET_SIZE = 900
CHUNK_MAX_SIZE = 1300
CHUNK_OVERLAP = 120
CACHE_FILE = Path("embeddings_cache.npz")

# Reciprocal Rank Fusion: классическая константа k=60
RRF_K = 60


# ---------------- Чтение и очистка markdown-исходника ----------------

def _read_kb(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Файл базы знаний не найден: {path}")
    try:
        text = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = p.read_text(encoding="cp1251")
    return _unescape_markdown(text)


def _unescape_markdown(text: str) -> str:
    """Снимает backslash-экранирование, которое часто остаётся при копипасте из Google Docs.

    Превращает `\\#`, `\\##`, `\\=`, `\\-`, `\\.` обратно в исходные символы.
    """
    # Снимаем экранирование перед типичными markdown-символами
    text = re.sub(r"\\([#=\-.])", r"\1", text)
    # Дополнительно убираем `====...====` визуальные разделители (заменяем на пустую строку)
    text = re.sub(r"^=+\s*$", "", text, flags=re.MULTILINE)
    # DOCUMENT N → ## DOCUMENT N (превратим в подзаголовок чтоб попасть в нарезку)
    text = re.sub(r"^DOCUMENT\s+(\d+)\s*$", r"## DOCUMENT \1", text, flags=re.MULTILINE)
    return text


# ---------------- Чанки ----------------

@dataclass
class Chunk:
    idx: int
    title: str
    text: str

    def preview(self, max_len: int = 220) -> str:
        body = self.text.split("\n\n", 1)[-1] if "\n\n" in self.text else self.text
        body = body.strip().replace("\n", " ")
        return body[:max_len] + ("…" if len(body) > max_len else "")


@dataclass
class SearchHit:
    """Результат поиска одного чанка с метаданными."""
    chunk_idx: int
    rank: int
    score: float
    source: str  # "embedding" / "bm25" / "hybrid"

    def to_dict(self, chunk: Chunk) -> dict:
        return {
            "chunk_idx": self.chunk_idx,
            "rank": self.rank,
            "score": round(float(self.score), 4),
            "source": self.source,
            "title": chunk.title,
            "preview": chunk.preview(),
            "text": chunk.text,
        }


def _split_into_sections(text: str) -> List[Tuple[str, str]]:
    """Делит markdown по заголовкам #, ##, ###. Возвращает [(title_path, body), ...]."""
    lines = text.splitlines()
    sections: List[Tuple[str, str]] = []
    current_h1, current_h2, current_h3 = "", "", ""
    buffer: List[str] = []

    def flush():
        body = "\n".join(buffer).strip()
        if not body:
            return
        parts = [p for p in (current_h1, current_h2, current_h3) if p]
        title = " / ".join(parts) if parts else "Без заголовка"
        sections.append((title, body))

    for raw in lines:
        line = raw.rstrip()
        m1 = re.match(r"^#\s+(.+)$", line)
        m2 = re.match(r"^##\s+(.+)$", line)
        m3 = re.match(r"^###\s+(.+)$", line)
        if m1:
            flush(); buffer = []
            current_h1 = m1.group(1).strip(); current_h2 = ""; current_h3 = ""
            continue
        if m2:
            flush(); buffer = []
            current_h2 = m2.group(1).strip(); current_h3 = ""
            continue
        if m3:
            flush(); buffer = []
            current_h3 = m3.group(1).strip()
            continue
        buffer.append(raw)
    flush()
    return sections


def _split_long_text(text: str, target: int, max_size: int, overlap: int) -> List[str]:
    if len(text) <= max_size:
        return [text]
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: List[str] = []
    current = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) + 2 <= target:
            current = f"{current}\n\n{para}" if current else para
        else:
            if current:
                chunks.append(current)
                tail = current[-overlap:] if overlap > 0 else ""
                current = f"{tail}\n\n{para}" if tail else para
            else:
                current = para
            while len(current) > max_size:
                chunks.append(current[:max_size])
                current = current[max_size - overlap:]
    if current:
        chunks.append(current)
    return chunks


def build_chunks(kb_text: str) -> List[Chunk]:
    sections = _split_into_sections(kb_text)
    chunks: List[Chunk] = []
    idx = 0
    for title, body in sections:
        parts = _split_long_text(body, CHUNK_TARGET_SIZE, CHUNK_MAX_SIZE, CHUNK_OVERLAP)
        for part in parts:
            text_with_title = f"[{title}]\n\n{part.strip()}"
            chunks.append(Chunk(idx=idx, title=title, text=text_with_title))
            idx += 1
    return chunks


def _kb_fingerprint(kb_text: str, n_chunks: int) -> str:
    h = hashlib.sha1()
    h.update(kb_text.encode("utf-8", errors="ignore"))
    h.update(str(n_chunks).encode())
    h.update(EMBEDDING_MODEL_NAME.encode())
    return h.hexdigest()


# ---------------- Embedding-индекс ----------------

class EmbeddingIndex:
    def __init__(self):
        self.model: SentenceTransformer | None = None
        self.embeddings: np.ndarray | None = None
        self.fingerprint: str = ""

    def build(self, chunks: List[Chunk], fingerprint: str) -> None:
        self.fingerprint = fingerprint
        logger.info("Загрузка модели эмбеддингов: %s", EMBEDDING_MODEL_NAME)
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")

        if self._load_cache(len(chunks)):
            logger.info("Эмбеддинги загружены из кэша.")
            return

        logger.info("Вычисление эмбеддингов для %d чанков...", len(chunks))
        texts = [c.text for c in chunks]
        embs = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        self.embeddings = embs
        self._save_cache()
        logger.info("Эмбеддинги готовы. Shape: %s", embs.shape)

    def _load_cache(self, expected_n: int) -> bool:
        if not CACHE_FILE.exists():
            return False
        try:
            data = np.load(CACHE_FILE, allow_pickle=False)
            if str(data.get("fingerprint", "")) != self.fingerprint:
                return False
            embs = data["embeddings"]
            if embs.shape[0] != expected_n:
                return False
            self.embeddings = embs.astype("float32")
            return True
        except Exception as e:
            logger.warning("Не удалось загрузить кэш эмбеддингов: %s", e)
            return False

    def _save_cache(self) -> None:
        try:
            np.savez(
                CACHE_FILE,
                embeddings=self.embeddings,
                fingerprint=np.array(self.fingerprint),
            )
        except Exception as e:
            logger.warning("Не удалось сохранить кэш эмбеддингов: %s", e)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Возвращает [(chunk_idx, similarity), ...] отсортированный по убыванию."""
        if self.model is None or self.embeddings is None:
            raise RuntimeError("Embedding-индекс не построен.")
        q_emb = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")[0]
        scores = self.embeddings @ q_emb
        top_idx = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i])) for i in top_idx]


# ---------------- BM25-индекс ----------------

class BM25Index:
    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self.tokenized_corpus: List[List[str]] = []

    def build(self, chunks: List[Chunk]) -> None:
        logger.info("Токенизация %d чанков для BM25...", len(chunks))
        self.tokenized_corpus = [tokenize_for_bm25(c.text) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        logger.info("BM25 индекс готов.")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        if self.bm25 is None:
            raise RuntimeError("BM25 индекс не построен.")
        query_tokens = tokenize_for_bm25(query)
        if not query_tokens:
            return []
        scores = self.bm25.get_scores(query_tokens)
        top_idx = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i])) for i in top_idx if scores[i] > 0]


# ---------------- Гибридный индекс ----------------

@dataclass
class HybridSearchResult:
    keywords: List[str]
    embedding_hits: List[dict]       # топ-5 от embedding
    bm25_hits: List[dict]            # топ-5 от BM25
    final_hits: List[dict]           # объединённый топ-5 после RRF
    max_similarity: float            # максимум по embedding
    below_threshold: bool            # сработал ли порог галлюцинаций


class HybridIndex:
    def __init__(self, kb_path: str, threshold: float = 0.35):
        self.kb_path = kb_path
        self.threshold = threshold
        self.chunks: List[Chunk] = []
        self.emb_idx = EmbeddingIndex()
        self.bm25_idx = BM25Index()

    def build(self) -> None:
        logger.info("Чтение базы знаний: %s", self.kb_path)
        kb_text = _read_kb(self.kb_path)
        self.chunks = build_chunks(kb_text)
        logger.info("Получено чанков: %d", len(self.chunks))
        fingerprint = _kb_fingerprint(kb_text, len(self.chunks))
        self.emb_idx.build(self.chunks, fingerprint)
        self.bm25_idx.build(self.chunks)

    def search(self, query: str, top_k: int = 5) -> HybridSearchResult:
        keywords = extract_keywords(query, max_keywords=10)

        emb_raw = self.emb_idx.search(query, top_k=top_k)
        bm25_raw = self.bm25_idx.search(query, top_k=top_k)

        # Reciprocal Rank Fusion
        rrf: dict[int, float] = {}
        for rank, (cidx, _) in enumerate(emb_raw, start=1):
            rrf[cidx] = rrf.get(cidx, 0.0) + 1.0 / (RRF_K + rank)
        for rank, (cidx, _) in enumerate(bm25_raw, start=1):
            rrf[cidx] = rrf.get(cidx, 0.0) + 1.0 / (RRF_K + rank)

        final_sorted = sorted(rrf.items(), key=lambda x: -x[1])[:top_k]

        emb_hits = [
            SearchHit(chunk_idx=ci, rank=r + 1, score=s, source="embedding").to_dict(self.chunks[ci])
            for r, (ci, s) in enumerate(emb_raw)
        ]
        bm25_hits = [
            SearchHit(chunk_idx=ci, rank=r + 1, score=s, source="bm25").to_dict(self.chunks[ci])
            for r, (ci, s) in enumerate(bm25_raw)
        ]
        final_hits = [
            SearchHit(chunk_idx=ci, rank=r + 1, score=s, source="hybrid").to_dict(self.chunks[ci])
            for r, (ci, s) in enumerate(final_sorted)
        ]

        max_sim = emb_raw[0][1] if emb_raw else 0.0
        below = max_sim < self.threshold

        return HybridSearchResult(
            keywords=keywords,
            embedding_hits=emb_hits,
            bm25_hits=bm25_hits,
            final_hits=final_hits,
            max_similarity=round(max_sim, 4),
            below_threshold=below,
        )
