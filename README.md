# 🏨 Hotel RAG — Интеллектуальная система поиска по корпоративным регламентам

> Система на базе **Hybrid RAG** для мгновенного поиска и консультации по внутренней нормативной документации отеля. Объединяет семантический поиск через нейронные эмбеддинги и классический алгоритм BM25 с прозрачной визуализацией логики принятия решений в реальном времени.

---

## 🎯 Ключевые возможности

- 🔍 **Гибридный поиск** — одновременно по смыслу (векторное сходство) и по ключевым словам (BM25); результаты объединяются через **Reciprocal Rank Fusion**
- 🏷️ **Извлечение ключевых слов без LLM** — лемматизация через pymorphy3 с фильтрацией стоп-слов
- 🛡️ **Двойная защита от галлюцинаций** — порог cosine similarity + строгий системный промпт
- 🔬 **Прозрачная логика** — боковая панель в реальном времени показывает все этапы RAG-конвейера
- 💾 **Кэш эмбеддингов** — SHA1-fingerprint по содержимому KB, пересчёт только при изменениях
- 🌗 **Адаптивный интерфейс** — тёмная и светлая темы, aurora-анимации, ken-burns на hero-баннере

---

## 🏗️ Архитектура

```
Запрос пользователя
         │
         ├──► pymorphy3 (лемматизация) ──► Фильтр стоп-слов ──► Ключевые слова
         │                                                              │
         │                                                              ▼
         │                                                  BM25Okapi.get_scores()
         │                                                              │
         └──► SentenceTransformer.encode() ──► cosine similarity ───────┤
                      (384-мерный вектор)                               │
                                                                        ▼
                                                     Reciprocal Rank Fusion (k=60)
                                                     score = Σ 1 / (60 + rank)
                                                                        │
                                                        max_similarity < 0.35?
                                                     ┌──────────────────┴─────────────────┐
                                                     ▼                                    ▼
                                          Стандартный отказ                    Qwen → OpenRouter
                                          (LLM не вызывается)                  (контекст + вопрос)
                                                                                          │
                                                                                          ▼
                                                                                   Ответ + панель
```

---

## ⚙️ Технологический стек

### 🖥️ Backend

| Компонент | Технология | Назначение |
|---|---|---|
| API-сервер | **FastAPI** + **uvicorn** | Асинхронные эндпоинты, lifespan-инициализация индекса |
| Шаблоны | **Jinja2** | Server-side рендеринг HTML |
| Валидация | **Pydantic v2** | Типизированные модели запросов и ответов |
| Конфигурация | **python-dotenv** | Управление секретами через переменные окружения |

### 🔍 RAG-ядро

| Компонент | Технология | Назначение |
|---|---|---|
| Семантические эмбеддинги | **sentence-transformers** `paraphrase-multilingual-MiniLM-L12-v2` | 384-мерные векторы, многоязычность, CPU |
| Вычисления | **PyTorch (CPU)** + **NumPy** | `embeddings @ query_vec` — косинусное сходство за O(n) |
| Нормализация | **scikit-learn** | L2-нормализация векторов перед поиском |
| Keyword-поиск | **rank-bm25** `BM25Okapi` | Вероятностная модель релевантности по терминам |
| Морфология | **pymorphy3** + **pymorphy3-dicts-ru** | Лемматизация русского языка без LLM |
| Слияние результатов | **RRF** | Объединение ranked-списков, k=60 |

### 🤖 LLM (каскадный вызов)

```
1. Qwen / DashScope  (qwen-plus)              ← основной провайдер
         ↓ ошибка 4xx / timeout / 5xx
2. OpenRouter — перебор 6 free-моделей:
   deepseek/deepseek-chat-v3-0324:free
   deepseek/deepseek-r1:free
   qwen/qwen3-235b-a22b:free
   qwen/qwen3-30b-a3b:free
   meta-llama/llama-3.3-70b-instruct:free
   google/gemini-2.0-flash-exp:free
```

- Таймаут на модель: **25 сек**
- Пауза между попытками: **0.8 сек**
- Классификация ошибок: `LLMConfigError` (4xx — не повторять) vs `LLMTransientError` (429/5xx — следующая модель)

### 🎨 Frontend

| Технология | Применение |
|---|---|
| **Vanilla JS** | Рендер панели RAG, markdown в ответах LLM, управление темой |
| **CSS Custom Properties** | Дуальная тема через `[data-theme="dark/light"]` |
| **CSS Animations** | Aurora-blobs (blur 80px), ken-burns на hero, particle-rise |

---

## 🔬 Алгоритм Hybrid Search — подробно

### Этап 1 — Лемматизация запроса (без LLM)

```python
tokens = re.findall(r"[А-Яа-яЁёA-Za-z][А-Яа-яЁёA-Za-z\-]*", text)
lemmas = [
    morph.parse(token)[0].normal_form
    for token in tokens
    if token.lower() not in RUSSIAN_STOPWORDS
    and morph.parse(token)[0].tag.POS not in _DROP_POS
]
```

Отфильтровываются служебные части речи (`PREP`, `CONJ`, `PRCL`, `INTJ`, `NPRO`) и ~100 русских стоп-слов.

### Этап 2 — Параллельный поиск

```python
embedding_hits = embedding_index.search(query, top_k=5)   # cosine similarity 0..1
bm25_hits      = bm25_index.search(lemmatized_tokens, top_k=5)  # BM25 score 0..∞
```

### Этап 3 — Reciprocal Rank Fusion

```python
RRF_K = 60
rrf_scores: dict[int, float] = {}
for rank, (chunk_idx, _) in enumerate(embedding_hits, start=1):
    rrf_scores[chunk_idx] = rrf_scores.get(chunk_idx, 0) + 1.0 / (RRF_K + rank)
for rank, (chunk_idx, _) in enumerate(bm25_hits, start=1):
    rrf_scores[chunk_idx] = rrf_scores.get(chunk_idx, 0) + 1.0 / (RRF_K + rank)
final = sorted(rrf_scores, key=lambda x: -rrf_scores[x])[:top_k]
```

Максимальный теоретический RRF-score при совпадении позиций: `2 × 1/(60+1) ≈ 0.0328`.

### Этап 4 — Порог галлюцинаций

```python
max_sim = max(score for _, score in embedding_hits)
if max_sim < HALLUCINATION_THRESHOLD:   # default 0.35
    return LLMResult(answer=NO_INFO_ANSWER, skipped_llm=True)
```

LLM не получает контекст и не вызывается вовсе — исключается возможность «творческого» ответа.

---

## 🗃️ Структура базы знаний и чанков

| Параметр | Значение |
|---|---|
| Формат | Markdown (`hotel_kb.md`) |
| Документов (регламентов) | 73 |
| Чанков после нарезки | 365 |
| Целевой размер чанка | 900 символов |
| Максимальный размер | 1 300 символов |
| Перекрытие (overlap) | 120 символов |

Кэш эмбеддингов (`embeddings_cache.npz`) хранит матрицу `(365, 384)` и инвалидируется при изменении контента KB.

---

## 📡 API-эндпоинты

| Метод | Путь | Описание |
|---|---|---|
| `GET` | `/` | Главная страница (Jinja2-шаблон) |
| `GET` | `/api/health` | Статус индекса, число чанков, порог |
| `POST` | `/chat` | Гибридный поиск + LLM-ответ |

### `POST /chat` — тело запроса

```json
{"message": "Во сколько check-out?", "history": []}
```

### `POST /chat` — расширенный ответ

```json
{
  "keywords": ["check-out", "отель"],
  "embedding_hits": [
    {"rank": 1, "score": 0.847, "title": "...", "preview": "...", "text": "...", "source": "embedding"}
  ],
  "bm25_hits": [{"rank": 1, "score": 7.39, "title": "...", "source": "bm25"}],
  "final_hits": [{"rank": 1, "score": 0.0325, "title": "...", "source": "hybrid"}],
  "max_similarity": 0.847,
  "below_threshold": false,
  "threshold": 0.35,
  "answer": "Check-out в отеле в 12:00...",
  "llm_model_used": "qwen/qwen-plus",
  "llm_error": null,
  "skipped_llm": false
}
```

Ответ содержит **все промежуточные данные** поиска — для отображения в панели «Логика работы RAG».

---

## 🎛️ Конфигурация

| Переменная окружения | По умолчанию | Описание |
|---|---|---|
| `QWEN_API_KEY` | — | Ключ DashScope (Alibaba Cloud), основной LLM |
| `QWEN_BASE_URL` | `https://dashscope-intl.aliyuncs.com/...` | URL API Qwen |
| `QWEN_MODEL` | `qwen-plus` | Модель Qwen |
| `OPENROUTER_API_KEY` | — | Ключ OpenRouter, резервные LLM |
| `KB_FILE` | `hotel_kb.md` | Путь к файлу базы знаний |
| `HALLUCINATION_THRESHOLD` | `0.35` | Минимальный cosine similarity для вызова LLM |
| `APP_HOST` | `0.0.0.0` | Хост uvicorn |
| `APP_PORT` | `8000` | Порт uvicorn |

---

## 🚀 Развёртывание

### Docker

```bash
cp .env.example .env
# Заполнить QWEN_API_KEY или OPENROUTER_API_KEY
docker-compose up --build
```

→ http://localhost:8801

### Без Docker

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
cp .env.example .env
python main.py
```

→ http://localhost:8000

### Replit

1. Импортировать репозиторий через GitHub
2. В разделе **Secrets** добавить `QWEN_API_KEY` и/или `OPENROUTER_API_KEY`
3. Нажать **Run** — публичная ссылка формируется автоматически

---

## 📊 Характеристики

| Метрика | Значение |
|---|---|
| Документов в базе знаний | 73 регламента |
| Чанков в индексе | 365 |
| Размерность эмбеддингов | 384 |
| Размер кэша (`embeddings_cache.npz`) | ~561 KB |
| Поиск top-5 по 365 чанкам | < 5 мс |
| Размер модели эмбеддингов | ~117 МБ |
| Время первого ответа (кэш загружен) | ~1–2 сек |

---

## 🗂️ Структура проекта

```
hotel_rag/
├── main.py                   # FastAPI: эндпоинты, lifespan
├── rag.py                    # HybridIndex (EmbeddingIndex + BM25Index + RRF)
├── llm.py                    # LLM-клиент: Qwen → OpenRouter, порог галлюцинаций
├── keywords.py               # Извлечение ключевых слов (pymorphy3, без LLM)
│
├── hotel_kb.md               # База знаний — 73 регламента
├── embeddings_cache.npz      # Кэш эмбеддингов (SHA1-fingerprint)
│
├── templates/
│   └── index.html            # Hero-баннер + чат + RAG-панель
├── static/
│   ├── style.css             # Фиолетово-малиновая тема, aurora, ken-burns
│   ├── app.js                # Рендер панели RAG, markdown, тема
│   └── hero-hotel.jpg        # Hero-изображение
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

---

## 🛠️ Зависимости

```
fastapi==0.115.6          uvicorn[standard]==0.32.1    jinja2==3.1.4
python-multipart==0.0.18  pydantic==2.10.3             python-dotenv==1.0.1
sentence-transformers==3.3.1
torch==2.5.1+cpu  (--extra-index-url https://download.pytorch.org/whl/cpu)
numpy==1.26.4     scikit-learn==1.5.2    httpx==0.27.2
rank-bm25==0.2.2
pymorphy3==2.0.2  pymorphy3-dicts-ru==2.4.417150.4580142
```

---

## 📝 Лицензия

MIT — свободное использование и модификация.

---

*Технологии: FastAPI · sentence-transformers · rank-bm25 · pymorphy3 · Qwen · OpenRouter · Docker*
