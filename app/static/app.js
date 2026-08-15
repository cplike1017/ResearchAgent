/* Agent Runtime Web UI 前端逻辑 */
"use strict";

const state = {
  sessionId: null,
  agentMode: "react",
  streaming: false,
};

const $ = (sel) => document.querySelector(sel);

/* ---------------- 初始化 ---------------- */
async function init() {
  await Promise.all([loadCapabilities(), loadSessions()]);
  setConnStatus(true);
  bindEvents();
}

function setConnStatus(online) {
  $("#conn-status").className = "status-dot" + (online ? " online" : "");
  $("#conn-text").textContent = online ? "已连接" : "连接失败";
}

/* ---------------- 能力列表 ---------------- */
async function loadCapabilities() {
  try {
    const [tools, skills, mcp] = await Promise.all([
      fetch("/api/web/tools").then((r) => r.json()),
      fetch("/api/web/skills").then((r) => r.json()),
      fetch("/api/web/mcp").then((r) => r.json()),
    ]);

    renderList("#tool-list", tools.tools.map((t) => ({
      text: t.name + (t.risk_level !== "low" ? ` [${t.risk_level}]` : ""),
      title: t.description,
    })));
    $("#tool-count").textContent = tools.count;

    renderList("#skill-list", skills.skills.map((s) => ({
      text: s.name, title: s.description,
    })));
    $("#skill-count").textContent = skills.count;

    renderList("#mcp-list", mcp.servers.map((s) => ({
      text: `${s.name} (${s.tool_count} tools)`, title: `transport: ${s.transport}`,
    })));
    $("#mcp-count").textContent = mcp.count;
  } catch (e) {
    setConnStatus(false);
  }
}

function renderList(sel, items) {
  const ul = $(sel);
  ul.innerHTML = "";
  items.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item.text;
    if (item.title) li.title = item.title;
    ul.appendChild(li);
  });
}

/* ---------------- 会话 ---------------- */
async function loadSessions() {
  try {
    const data = await fetch("/api/web/sessions").then((r) => r.json());
    renderList("#session-list", (data.sessions || []).map((s) => ({
      text: s.session_id.slice(-16), title: `created: ${s.created_at}`,
    })));
  } catch (e) { /* 忽略 */ }
}

/* ---------------- 消息渲染 ---------------- */
function addMessage(role, content, meta) {
  const div = document.createElement("div");
  div.className = "msg " + role;
  if (meta) {
    const metaEl = document.createElement("div");
    metaEl.className = "msg-meta";
    metaEl.textContent = meta;
    div.appendChild(metaEl);
  }
  const contentEl = document.createElement("div");
  contentEl.textContent = content;
  div.appendChild(contentEl);
  $("#messages").appendChild(div);
  scrollToBottom();
  return div;
}

function addToolMsg(data) {
  const div = document.createElement("div");
  div.className = "msg tool";
  const status = data.success ? '<span class="tool-ok">✓</span>' : '<span class="tool-err">✗</span>';
  div.innerHTML = `
    <div class="tool-name">🛠 ${data.tool}</div>
    <div>${status} 参数: ${JSON.stringify(data.arguments)}</div>
    ${data.data ? `<div class="tool-ok">→ ${data.data}</div>` : ""}
    ${data.error ? `<div class="tool-err">→ ${data.error.message || JSON.stringify(data.error)}</div>` : ""}
  `;
  $("#messages").appendChild(div);
  scrollToBottom();
}

function addErrorMsg(message) {
  addMessage("error", `⚠️ ${message}`);
}

function scrollToBottom() {
  const msgs = $("#messages");
  msgs.scrollTop = msgs.scrollHeight;
}

/* ---------------- 发送 ---------------- */
async function send() {
  const input = $("#input");
  const message = input.value.trim();
  if (!message || state.streaming) return;

  addMessage("user", message);
  input.value = "";
  setStreaming(true);

  const assistantEl = addMessage("assistant", "");
  assistantEl.classList.add("streaming");
  const contentEl = assistantEl.querySelector("div:last-child");
  contentEl.textContent = "";

  try {
    // SSE 流式：实时展示每步决策与工具调用
    const resp = await fetch("/api/web/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        session_id: state.sessionId,
        agent_mode: state.agentMode,
      }),
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let finalAnswer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // 解析 SSE 帧
      let idx;
      while ((idx = buffer.indexOf("\n\n")) !== -1) {
        const frame = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        handleFrame(frame, contentEl, (ans) => { finalAnswer = ans; });
      }
    }
    if (finalAnswer) contentEl.textContent = finalAnswer;
    assistantEl.classList.remove("streaming");
  } catch (e) {
    assistantEl.classList.remove("streaming");
    contentEl.textContent = "⚠️ 请求失败: " + e.message;
  }
  setStreaming(false);
  loadSessions();
}

function handleFrame(frame, contentEl, onAnswer) {
  const lines = frame.split("\n");
  const eventLine = lines.find((l) => l.startsWith("event:"));
  const dataLine = lines.find((l) => l.startsWith("data:"));
  if (!eventLine || !dataLine) return;
  const event = eventLine.slice(6).trim();
  let data;
  try { data = JSON.parse(dataLine.slice(5).trim()); } catch (e) { return; }

  switch (event) {
    case "step":
      if (data.tool_calls && data.tool_calls.length) {
        data.tool_calls.forEach((tc) => {
          addToolMsg({ tool: tc.name, arguments: tc.arguments, success: true, data: "等待执行..." });
        });
      }
      break;
    case "tool_result":
      addToolMsg(data);
      break;
    case "final":
      contentEl.textContent = data.content || "";
      break;
    case "done":
      state.sessionId = data.session_id;
      onAnswer(data.answer || "");
      break;
    case "error":
      addErrorMsg(data.message);
      break;
  }
}

function setStreaming(v) {
  state.streaming = v;
  $("#send").disabled = v;
}

/* ---------------- 事件绑定 ---------------- */
function bindEvents() {
  const input = $("#input");
  const sendBtn = $("#send");

  sendBtn.addEventListener("click", send);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  });
  input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 120) + "px";
  });

  document.querySelectorAll(".mode-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".mode-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.agentMode = btn.dataset.mode;
      $("#mode-badge").textContent = btn.dataset.mode;
    });
  });
}

init();
