"""Извлечение ключевых слов из текста БЕЗ использования LLM.

Алгоритм:
1. Токенизация (берём только слова длиной >= 3)
2. Лемматизация через pymorphy3 (заселения → заселение)
3. Отбрасываем стоп-слова и служебные части речи
4. Дедупликация с сохранением порядка
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import List

import pymorphy3

# Расширенный список русских стоп-слов
RUSSIAN_STOPWORDS = {
    # местоимения
    "я", "ты", "он", "она", "оно", "мы", "вы", "они",
    "мой", "твой", "его", "её", "наш", "ваш", "их", "свой",
    "себя", "это", "этот", "эта", "эти", "тот", "та", "те",
    "который", "которая", "которое", "которые",
    "кто", "что", "какой", "какая", "какое", "какие",
    # союзы / предлоги
    "и", "или", "но", "а", "да", "же", "ли", "бы", "не", "ни",
    "в", "во", "на", "за", "под", "над", "при", "по", "из", "до",
    "о", "об", "обо", "от", "у", "к", "ко", "с", "со", "без",
    "для", "про", "через", "между", "среди",
    "как", "так", "там", "тут", "где", "когда", "если", "чтобы",
    "потому", "поэтому", "однако", "только", "также", "тоже",
    # частицы / связки
    "быть", "есть", "был", "была", "было", "были",
    "уже", "ещё", "еще", "вот", "ну", "ли", "же",
    # числительные слова
    "один", "два", "три", "первый", "второй",
    # часто встречающиеся в вопросах
    "можно", "нужно", "надо", "должен", "должна", "должны",
    "какие", "какая", "какой", "какое",
    "пожалуйста", "спасибо",
    # часто встречающиеся обороты в базе знаний (мусорные)
    "регламент", "цель", "правила", "правило",
    "порядок", "действия", "действий",
    "запрещено", "рекомендации", "сотрудник", "сотрудники",
}

_MORPH = pymorphy3.MorphAnalyzer()
_WORD_RE = re.compile(r"[А-Яа-яЁёA-Za-z][А-Яа-яЁёA-Za-z\-]*")

# Части речи, которые отбрасываем (предлоги, союзы, частицы, междометия)
_DROP_POS = {"PREP", "CONJ", "PRCL", "INTJ", "NPRO"}


@lru_cache(maxsize=8192)
def _normalize_token(token: str) -> str | None:
    """Лемматизация одного токена. Возвращает лемму или None если стоп-слово."""
    low = token.lower()
    if len(low) < 3:
        return None
    if low in RUSSIAN_STOPWORDS:
        return None
    parses = _MORPH.parse(low)
    if not parses:
        return low
    best = parses[0]
    pos = str(best.tag.POS) if best.tag.POS else ""
    if pos in _DROP_POS:
        return None
    lemma = best.normal_form
    if lemma in RUSSIAN_STOPWORDS:
        return None
    return lemma


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Извлекает ключевые слова из текста.

    Возвращает список лемм в порядке появления, без дубликатов.
    """
    if not text or not text.strip():
        return []
    tokens = _WORD_RE.findall(text)
    seen: set = set()
    result: List[str] = []
    for tok in tokens:
        lemma = _normalize_token(tok)
        if lemma and lemma not in seen:
            seen.add(lemma)
            result.append(lemma)
            if len(result) >= max_keywords:
                break
    return result


def tokenize_for_bm25(text: str) -> List[str]:
    """Токенизация текста для индексации BM25 (без ограничения количества)."""
    if not text:
        return []
    tokens = _WORD_RE.findall(text)
    result: List[str] = []
    for tok in tokens:
        lemma = _normalize_token(tok)
        if lemma:
            result.append(lemma)
    return result
