/* =========================================================
   Advanced-RAG-Hotel — frontend logic
   ========================================================= */

const $ = (id) => document.getElementById(id);

// ---------------- Тема ----------------

(function initTheme() {
  const saved = localStorage.getItem("hotel-rag-theme");
  const prefersLight =
    window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches;
  const theme = saved || (prefersLight ? "light" : "dark");
  document.documentElement.dataset.theme = theme;
})();

$("theme-toggle").addEventListener("click", () => {
  const cur = document.documentElement.dataset.theme || "dark";
  const next = cur === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  localStorage.setItem("hotel-rag-theme", next);
});

// ---------------- Частицы ----------------

(function spawnParticles() {
  const root = $("particles");
  if (!root) return;
  for (let i = 0; i < 22; i++) {
    const p = document.createElement("div");
    p.className = "particle";
    p.style.left = Math.random() * 100 + "%";
    const size = 2 + Math.random() * 4;
    p.style.width = size + "px";
    p.style.height = size + "px";
    p.style.animationDuration = 18 + Math.random() * 22 + "s";
    p.style.animationDelay = -Math.random() * 30 + "s";
    p.style.opacity = 0.3 + Math.random() * 0.5;
    root.appendChild(p);
  }
})();

// ---------------- Стартовые stat-цифры ----------------

(async function fetchHealth() {
  try {
    const r = await fetch("/api/health");
    const j = await r.json();
    if (j.chunks) $("stat-chunks").textContent = j.chunks;
  } catch (e) {
    /* пусто */
  }
})();

// ---------------- Markdown (минимальный) ----------------

function escapeHtml(s) {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function renderMarkdown(src) {
  if (!src) return "";
  let s = escapeHtml(src);

  // защита inline-кода
  const codes = [];
  s = s.replace(/`([^`]+)`/g, (m, c) => {
    codes.push(c);
    return `\x00CODE${codes.length - 1}\x00`;
  });

  // bold / italic
  s = s.replace(/\*\*([^\n*]+)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/(^|[^*])\*([^\n*]+)\*([^*]|$)/g, "$1<em>$2</em>$3");

  // списки
  const lines = s.split("\n");
  const out = [];
  let inUL = false,
    inOL = false;
  for (const ln of lines) {
    const ul = /^\s*[-•]\s+(.+)$/.exec(ln);
    const ol = /^\s*\d+[\.\)]\s+(.+)$/.exec(ln);
    if (ul) {
      if (inOL) { out.push("</ol>"); inOL = false; }
      if (!inUL) { out.push("<ul>"); inUL = true; }
      out.push(`<li>${ul[1]}</li>`);
    } else if (ol) {
      if (inUL) { out.push("</ul>"); inUL = false; }
      if (!inOL) { out.push("<ol>"); inOL = true; }
      out.push(`<li>${ol[1]}</li>`);
    } else {
      if (inUL) { out.push("</ul>"); inUL = false; }
      if (inOL) { out.push("</ol>"); inOL = false; }
      out.push(ln);
    }
  }
  if (inUL) out.push("</ul>");
  if (inOL) out.push("</ol>");
  s = out.join("\n");

  // параграфы из двойных переносов
  s = s
    .split(/\n{2,}/)
    .map((p) => (/<(ul|ol|h\d|pre)/.test(p) ? p : `<p>${p.replace(/\n/g, "<br>")}</p>`))
    .join("\n");

  s = s.replace(/\x00CODE(\d+)\x00/g, (m, i) => `<code>${escapeHtml(codes[+i])}</code>`);
  return s;
}

// ---------------- Чат ----------------

const chatWindow = $("chat-window");
const chatForm = $("chat-form");
const chatInput = $("chat-message");
const chatSend = $("chat-send");
const charCounter = $("char-counter");
const statusText = $("status-text");

const MAX_LEN = 1000;
const history = [];

function setStatus(text, busy = false) {
  statusText.textContent = text;
  statusText.parentElement.classList.toggle("busy", busy);
}

chatInput.addEventListener("input", () => {
  const n = chatInput.value.length;
  charCounter.textContent = `${n} / ${MAX_LEN}`;
  charCounter.classList.toggle("warn", n >= MAX_LEN * 0.8 && n < MAX_LEN);
  charCounter.classList.toggle("over", n >= MAX_LEN);
});

function addMessage({ role, text, klass = "", markdown = false }) {
  const wrap = document.createElement("div");
  wrap.className = `msg msg-${role} ${klass}`;
  const av = document.createElement("div");
  av.className = "msg-avatar";
  av.textContent = role === "assistant" ? "AI" : "Вы";
  const bubble = document.createElement("div");
  bubble.className = "msg-bubble";
  const t = document.createElement("div");
  t.className = "msg-text" + (markdown ? " md" : "");
  if (markdown) t.innerHTML = renderMarkdown(text);
  else t.textContent = text;
  bubble.appendChild(t);
  wrap.appendChild(av);
  wrap.appendChild(bubble);
  chatWindow.appendChild(wrap);
  chatWindow.scrollTop = chatWindow.scrollHeight;
  return wrap;
}

function addTyping() {
  const wrap = document.createElement("div");
  wrap.className = "msg msg-assistant msg-typing";
  wrap.innerHTML = `
    <div class="msg-avatar">AI</div>
    <div class="msg-bubble">
      <div class="typing-dots"><span></span><span></span><span></span></div>
    </div>`;
  chatWindow.appendChild(wrap);
  chatWindow.scrollTop = chatWindow.scrollHeight;
  return wrap;
}

function hideQuickQuestions() {
  const qq = $("quick-questions");
  if (!qq) return;
  qq.classList.add("fading-out");
  setTimeout(() => qq.remove(), 400);
}

// ---------------- Панель RAG ----------------

const ragEmpty = $("rag-empty");
const ragContent = $("rag-content");

function scoreClass(value, source) {
  // эмпирически: для embedding 0.35 — граница; для BM25 — больше 1 уже хорошо
  if (source === "embedding" || source === "hybrid") {
    if (value < 0.35) return "s-low";
    if (value < 0.6) return "s-mid";
    return "s-high";
  }
  if (source === "bm25") {
    if (value < 1) return "s-low";
    if (value < 3) return "s-mid";
    return "s-high";
  }
  return "s-mid";
}

function renderHits(containerId, hits) {
  const c = $(containerId);
  c.innerHTML = "";
  if (!hits || !hits.length) {
    c.innerHTML = `<div class="rag-hint" style="font-style:italic">— ничего не найдено</div>`;
    return;
  }
  hits.forEach((h) => {
    const div = document.createElement("div");
    div.className = "hit";
    const sc = scoreClass(h.score, h.source);
    div.innerHTML = `
      <div class="hit-row">
        <span class="hit-rank">${h.rank}</span>
        <span class="hit-title" title="${escapeHtml(h.title)}">${escapeHtml(h.title)}</span>
        <span class="hit-score ${sc}">${h.score}</span>
      </div>
      <div class="hit-preview">${escapeHtml(h.preview)}</div>
      <div class="hit-full">${escapeHtml(h.text)}</div>`;
    div.addEventListener("click", () => div.classList.toggle("expanded"));
    c.appendChild(div);
  });
}

function renderKeywords(keywords) {
  const c = $("keyword-chips");
  c.innerHTML = "";
  if (!keywords || !keywords.length) {
    c.innerHTML = `<div class="rag-hint" style="font-style:italic">— нет ключевых слов</div>`;
    $("cnt-keywords").textContent = "0";
    return;
  }
  keywords.forEach((k) => {
    const chip = document.createElement("span");
    chip.className = "kw-chip";
    chip.textContent = k;
    c.appendChild(chip);
  });
  $("cnt-keywords").textContent = keywords.length;
}

function renderLLMInfo(model, error, skipped) {
  const c = $("llm-info");
  const cnt = $("cnt-llm");
  if (error) {
    c.innerHTML = `<span style="color:var(--accent-red-bright)">${escapeHtml(error)}</span>`;
    cnt.textContent = "ошибка";
    cnt.className = "rag-step-count alert";
    return;
  }
  if (skipped) {
    c.innerHTML = `<span class="lbl">Статус:</span><span class="val">LLM не вызывался</span><br>
      <span class="lbl">Причина:</span><span class="val">Сработал порог галлюцинаций</span>`;
    cnt.textContent = "skip";
    cnt.className = "rag-step-count alert";
    return;
  }
  c.innerHTML = `<span class="lbl">Модель:</span> <span class="val model">${escapeHtml(model)}</span>`;
  cnt.textContent = "ok";
  cnt.className = "rag-step-count ok";
}

function renderThreshold(maxSim, threshold, below) {
  const c = $("threshold-info");
  const cnt = $("cnt-guard");
  const pct = Math.min(100, Math.max(0, maxSim * 100));
  const thrPct = Math.min(100, Math.max(0, threshold * 100));
  const statusHtml = below
    ? `<span style="color:var(--accent-red-bright);font-weight:700">⚠ Ниже порога — ответ заменён</span>`
    : `<span style="color:#10B981;font-weight:700">✓ Выше порога — LLM может отвечать</span>`;
  c.innerHTML = `
    <div>${statusHtml}</div>
    <div class="threshold-bar">
      <div class="threshold-bar-fill" style="width:${pct}%"></div>
      <div class="threshold-marker" style="left:${thrPct}%" data-label="порог ${threshold}"></div>
    </div>
    <div class="threshold-row"><span class="lbl">Макс. similarity:</span><span class="val">${maxSim}</span></div>
    <div class="threshold-row"><span class="lbl">Порог:</span><span class="val">${threshold}</span></div>`;
  cnt.textContent = below ? "блок" : "ok";
  cnt.className = "rag-step-count " + (below ? "alert" : "ok");
}

function renderRAGPanel(data) {
  ragEmpty.hidden = true;
  ragContent.hidden = false;

  renderKeywords(data.keywords);

  renderHits("hits-emb", data.embedding_hits);
  $("cnt-emb").textContent = (data.embedding_hits || []).length;

  renderHits("hits-bm25", data.bm25_hits);
  $("cnt-bm25").textContent = (data.bm25_hits || []).length;

  renderHits("hits-final", data.final_hits);
  $("cnt-final").textContent = (data.final_hits || []).length;

  renderLLMInfo(data.llm_model_used, data.llm_error, data.skipped_llm);
  renderThreshold(data.max_similarity, data.threshold, data.below_threshold);
}

// ---------------- Отправка ----------------

async function sendMessage(text) {
  text = (text || "").trim();
  if (!text) return;

  hideQuickQuestions();
  addMessage({ role: "user", text });
  history.push({ role: "user", content: text });

  chatInput.value = "";
  charCounter.textContent = `0 / ${MAX_LEN}`;
  charCounter.classList.remove("warn", "over");
  chatSend.disabled = true;
  setStatus("Поиск + LLM...", true);

  const typing = addTyping();

  try {
    const r = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, history }),
    });
    const data = await r.json();
    typing.remove();

    if (data.error) {
      addMessage({ role: "assistant", text: data.error, klass: "msg-error" });
      setStatus("Ошибка");
      return;
    }

    renderRAGPanel(data);

    if (data.skipped_llm) {
      addMessage({
        role: "assistant",
        text: data.answer,
        klass: "msg-warning",
      });
    } else if (data.llm_error) {
      addMessage({
        role: "assistant",
        text: data.llm_error,
        klass: "msg-error",
      });
    } else {
      addMessage({
        role: "assistant",
        text: data.answer,
        markdown: true,
      });
      history.push({ role: "assistant", content: data.answer });
    }
    setStatus("Готов к диалогу");
  } catch (e) {
    typing.remove();
    addMessage({
      role: "assistant",
      text: `Ошибка сети: ${e.message}`,
      klass: "msg-error",
    });
    setStatus("Ошибка");
  } finally {
    chatSend.disabled = false;
    chatInput.focus();
  }
}

chatForm.addEventListener("submit", (e) => {
  e.preventDefault();
  sendMessage(chatInput.value);
});

// Быстрые вопросы
document.querySelectorAll(".qq-chip").forEach((btn) => {
  btn.addEventListener("click", () => sendMessage(btn.dataset.q));
});

// Восстановление блока «Популярные вопросы»
function restoreQuickQuestions() {
  const qq = document.createElement("div");
  qq.className = "quick-questions";
  qq.id = "quick-questions";
  qq.innerHTML = `
    <div class="qq-label">Популярные вопросы:</div>
    <div class="qq-chips">
      <button type="button" class="qq-chip" data-q="Во сколько check-out в отеле?">Во сколько check-out?</button>
      <button type="button" class="qq-chip" data-q="Как оформить раннее заселение?">Раннее заселение</button>
      <button type="button" class="qq-chip" data-q="Что делать при overbooking?">Что делать при overbooking?</button>
      <button type="button" class="qq-chip" data-q="Какие правила работы с VIP-гостями?">VIP-гости</button>
      <button type="button" class="qq-chip" data-q="Как реагировать на агрессивного гостя?">Агрессивный гость</button>
      <button type="button" class="qq-chip" data-q="Какой дресс-код у сотрудников ресепшн?">Дресс-код</button>
    </div>`;
  qq.querySelectorAll(".qq-chip").forEach((btn) => {
    btn.addEventListener("click", () => sendMessage(btn.dataset.q));
  });
  chatWindow.appendChild(qq);
}

// Очистка чата
$("chat-clear").addEventListener("click", () => {
  if (!confirm("Очистить весь чат?")) return;
  chatWindow.innerHTML = "";
  history.length = 0;
  // Восстановим стартовое приветствие
  addMessage({
    role: "assistant",
    text:
      "Добрый день! Я AI-консультант премиального отеля для сотрудников. " +
      "Подскажу регламенты по любым ситуациям: заселение, выезд, работа с гостями, " +
      "бронирование, дресс-код, ЧП и т.д. Что вас интересует?",
  });
  // Восстановим популярные вопросы
  restoreQuickQuestions();
  // Сбросим панель
  ragContent.hidden = true;
  ragEmpty.hidden = false;
});
