"""LLM-клиент: Qwen (DashScope) основной → OpenRouter free резерв.

Включает проверку порога галлюцинаций ДО вызова LLM:
если максимальный similarity ниже порога — возвращает стандартный отказ.
"""
from __future__ import annotations

import asyncio
import os
import logging
from dataclasses import dataclass
from typing import List

import httpx

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 25.0
PAUSE_BETWEEN_MODELS = 0.8

NO_INFO_ANSWER = (
    "К сожалению, я не нашёл информацию по этому вопросу в базе знаний. "
    "Попробуйте переформулировать или задайте вопрос по другой теме (регламенты "
    "check-in, check-out, работа с гостями, бронирование, дресс-код и т.д.)."
)

OPENROUTER_FREE_MODELS = [
    "deepseek/deepseek-chat-v3-0324:free",
    "deepseek/deepseek-r1:free",
    "qwen/qwen3-235b-a22b:free",
    "qwen/qwen3-30b-a3b:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemini-2.0-flash-exp:free",
]

SYSTEM_PROMPT = """Ты — AI-консультант премиального отеля, обученный на корпоративных регламентах для сотрудников.

ПРАВИЛА:
1. Отвечай ТОЛЬКО на основе предоставленного контекста из базы знаний.
2. Если в контексте нет точного ответа — обязательно скажи: "В базе знаний я не нашёл прямого ответа на этот вопрос".
3. Никогда не выдумывай факты, цифры, сроки, правила.
4. Отвечай простым деловым русским языком, кратко и по делу.
5. Структурируй сложные ответы маркированным списком.
6. В конце ответа можешь предложить уточняющий вопрос, если это поможет лучше помочь сотруднику."""


@dataclass
class LLMResult:
    answer: str
    model_used: str
    error: str | None = None
    skipped_llm: bool = False  # если ответили "не нашёл" без вызова LLM


class LLMConfigError(Exception):
    """4xx — конфиг (ключ/модель), повтор бесполезен."""


class LLMTransientError(Exception):
    """429/5xx/таймаут — перебрать следующую модель."""


def _build_messages(question: str, chunks: List[dict], history: List[dict]) -> List[dict]:
    context = "\n\n---\n\n".join(
        f"Фрагмент {c['rank']} (score {c['score']}):\n{c['text']}" for c in chunks
    )
    user_content = (
        f"Контекст из корпоративной базы знаний:\n{context}\n\n"
        f"Вопрос сотрудника: {question}\n\n"
        "Ответь по контексту, кратко и по делу. Если ответа в контексте нет — честно скажи об этом."
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in history[-8:]:
        role = m.get("role")
        content = m.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_content})
    return messages


async def _call_openai_compatible(
    base_url: str,
    api_key: str,
    model: str,
    messages: List[dict],
    extra_headers: dict | None = None,
    timeout: float = REQUEST_TIMEOUT,
) -> str:
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 900,
    }
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(url, json=payload, headers=headers)
    except httpx.TimeoutException as e:
        raise LLMTransientError(f"timeout {timeout}s: {e}") from e
    except httpx.HTTPError as e:
        raise LLMTransientError(f"network: {type(e).__name__}: {e}") from e

    if r.status_code == 429 or r.status_code >= 500:
        raise LLMTransientError(f"HTTP {r.status_code}: {r.text[:200]}")
    if 400 <= r.status_code < 500:
        raise LLMConfigError(f"HTTP {r.status_code}: {r.text[:200]}")

    try:
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, ValueError, TypeError) as e:
        raise LLMTransientError(f"bad response: {e}") from e


async def _try_model(
    label: str, base_url: str, api_key: str, model: str,
    messages: List[dict], extra_headers: dict | None = None, errors: List[str] | None = None,
) -> str | None:
    try:
        return await _call_openai_compatible(
            base_url=base_url, api_key=api_key, model=model,
            messages=messages, extra_headers=extra_headers,
        )
    except LLMConfigError as e:
        msg = f"{label} ({model}) config-error: {e}"
        logger.warning(msg)
        if errors is not None:
            errors.append(msg)
    except LLMTransientError as e:
        msg = f"{label} ({model}) transient: {e}"
        logger.warning(msg)
        if errors is not None:
            errors.append(msg)
    return None


async def generate_answer(
    question: str,
    chunks: List[dict],
    history: List[dict],
    below_threshold: bool = False,
) -> LLMResult:
    # Защита от галлюцинаций №1: если порог не пройден — не зовём LLM
    if below_threshold or not chunks:
        logger.info("Сработал порог галлюцинаций — отвечаем без LLM.")
        return LLMResult(
            answer=NO_INFO_ANSWER,
            model_used="threshold-guard",
            skipped_llm=True,
        )

    messages = _build_messages(question, chunks, history)
    errors: List[str] = []
    first_attempt = True

    async def pause_if_needed():
        nonlocal first_attempt
        if first_attempt:
            first_attempt = False
        else:
            await asyncio.sleep(PAUSE_BETWEEN_MODELS)

    # 1) Qwen (DashScope) — основной
    qwen_key = os.getenv("QWEN_API_KEY", "").strip()
    if qwen_key:
        await pause_if_needed()
        base = os.getenv(
            "QWEN_BASE_URL",
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        ).strip()
        model = os.getenv("QWEN_MODEL", "qwen-plus").strip()
        answer = await _try_model("Qwen", base, qwen_key, model, messages, errors=errors)
        if answer:
            return LLMResult(answer=answer, model_used=f"qwen/{model}")

    # 2) OpenRouter free — резерв
    or_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if or_key:
        for model in OPENROUTER_FREE_MODELS:
            await pause_if_needed()
            answer = await _try_model(
                "OpenRouter", "https://openrouter.ai/api/v1",
                or_key, model, messages,
                extra_headers={
                    "HTTP-Referer": "https://replit.com",
                    "X-Title": "Advanced-RAG-Hotel",
                },
                errors=errors,
            )
            if answer:
                return LLMResult(answer=answer, model_used=f"openrouter/{model}")

    if not errors:
        return LLMResult(
            answer="",
            model_used="",
            error="Не настроен ни один LLM-провайдер. Заполните QWEN_API_KEY или OPENROUTER_API_KEY в Secrets.",
        )
    return LLMResult(
        answer="",
        model_used="",
        error="Не удалось получить ответ LLM.\n\nДетали: " + " | ".join(errors[-3:]),
    )
