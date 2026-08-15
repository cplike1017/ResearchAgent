/* Agent Runtime Web UI 前端逻辑 v2 */
"use strict";

const state = {
  sessionId: null,
  agentMode: "react",
  streaming: false,
  traceCache: {}, // session_id -> trace tree（会话历史里的 trace 树）
};

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

/* ================= 初始化 ================= */
async function init() {
  await Promise.all([loadCapabilities(), loadSessions()]);
  setConnStatus(true);
  bindEvents();
}

function setConnStatus(online) {
  $("#conn-status").className = "status-dot" + (online ? " online" : "");
  $("#conn-text").textContent = online ? "已连接" : "连接失败";
}

/* ================= 能力列表 ================= */
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
      text: `${s.name} (${s.tool_count})`, title: `transport: ${s.transport}`,
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
    li.dataset.id = item.id || "";
    ul.appendChild(li);
  });
}

/* ================= 会话 ================= */
async function loadSessions() {
  try {
    const data = await fetch("/api/web/sessions").then((r) => r.json());
    const ul = $("#session-list");
    ul.innerHTML = "";
    (data.sessions || []).forEach((s) => {
      const li = document.createElement("li");
      li.className = "session-item";
      li.dataset.sessionId = s.session_id;
      li.innerHTML = `
        <span class="si-icon">💬</span>
        <span class="si-name">${esc(s.session_id.slice(-16))}</span>
        <span class="si-time">${esc((s.updated_at || "").slice(11, 19))}</span>
      `;
      if (state.sessionId === s.session_id) li.classList.add("active");
      li.addEventListener("click", () => openSession(s.session_id));
      ul.appendChild(li);
    });
  } catch (e) { /* 忽略 */ }
}

async function openSession(sessionId) {
  state.sessionId = sessionId;
  // 高亮
  $$("#session-list .session-item").forEach((el) => el.classList.toggle("active", el.dataset.sessionId === sessionId));
  // 加载消息
  try {
    const data = await fetch(`/api/web/sessions/${sessionId}/messages`).then((r) => r.json());
    renderHistory(data.messages || []);
    setStatus(`会话 ${sessionId.slice(-12)}`);
  } catch (e) {
    addErrorMsg("加载会话失败: " + e.message);
  }
}

function renderHistory(messages) {
  $("#messages").innerHTML = "";
  messages.forEach((m) => {
    if (m.role === "user") {
      addMessage("user", typeof m.content === "string" ? m.content : JSON.stringify(m.content));
    } else if (m.role === "assistant") {
      // 带 tool_calls 的 assistant 消息 content 为 null，跳过（由后续 tool 消息体现）
      if (m.content === null || m.content === undefined || m.content === "") {
        if (m.tool_calls && m.tool_calls.length) return; // 跳过决策消息
        return;
      }
      const content = typeof m.content === "string" ? m.content : JSON.stringify(m.content);
      addMessage("assistant", content);
    } else if (m.role === "tool") {
      addToolMsg({ tool: m.name || "tool", arguments: {}, success: true, data: contentPreview(m.content) });
    }
  });
  scrollToBottom();
}

function contentPreview(s) {
  try {
    const obj = JSON.parse(s);
    return (obj.data || obj.content || s).toString().slice(0, 200);
  } catch (e) {
    return String(s).slice(0, 200);
  }
}

function newSession() {
  state.sessionId = null;
  $("#messages").innerHTML = `
    <div class="welcome">
      <h2>🤖 Agent Runtime</h2>
      <p>ReAct / Plan 双模式 · 记忆 · MCP · 技能</p>
      <p class="sub">工作流完全透明：每步决策、工具调用、耗时、Trace 树实时可见</p>
    </div>`;
  $$("#session-list .session-item").forEach((el) => el.classList.remove("active"));
  setStatus("新会话");
}

/* ================= 消息渲染 ================= */
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
    <div class="tool-name">🛠 ${esc(data.tool)}</div>
    <div>${status} 参数: ${esc(JSON.stringify(data.arguments))}</div>
    ${data.data && data.data !== "等待执行..." ? `<div class="tool-ok">→ ${esc(String(data.data).slice(0, 200))}</div>` : ""}
    ${data.error ? `<div class="tool-err">→ ${esc(data.error.message || JSON.stringify(data.error))}</div>` : ""}
  `;
  $("#messages").appendChild(div);
  scrollToBottom();
}

function addErrorMsg(message) {
  addMessage("error", `⚠️ ${message}`);
}

function setStatus(text) {
  const el = $("#chat-status");
  if (el) el.textContent = text;
}

function scrollToBottom() {
  const msgs = $("#messages");
  msgs.scrollTop = msgs.scrollHeight;
}

function esc(s) {
  // 纯字符串 HTML 转义（不依赖 DOM，更健壮）
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

/* ================= 工作流面板（Trace 树 v2） ================= */
const SPAN_ICONS = {
  gateway: "🚪", worker: "⚙️", agent: "🧠", llm: "💬", tool: "🛠",
  tool_gateway: "🛡️", context_builder: "📦", checkpoint: "💾", queue: "📨",
  memory: "🧠", eval: "📊", generic: "·",
};

function addWorkflowPanel(data, opts) {
  const panel = document.createElement("div");
  panel.className = "workflow";
  panel.id = "wf-" + (data.trace_id || Date.now());

  // Header
  const header = document.createElement("div");
  header.className = "workflow-header";
  const planInfo = data.plan && data.plan.length
    ? ` · 计划 ${data.plan.length} 步${data.plan_revisions ? ` · 重规划 ${data.plan_revisions}` : ""}`
    : "";
  header.innerHTML = `
    <span class="wf-title">🔍 Agent 工作流</span>
    <span class="wf-meta">${data.trace_id ? data.trace_id.slice(-12) : ""}${planInfo}</span>
    <span class="wf-toggle">收起 ▴</span>
  `;

  // Body
  const body = document.createElement("div");
  body.className = "workflow-body";
  body.innerHTML = renderWorkflowBody(data);
  header.addEventListener("click", () => {
    panel.classList.toggle("open");
    header.querySelector(".wf-toggle").textContent = panel.classList.contains("open") ? "收起 ▴" : "展开 ▾";
  });
  panel.appendChild(header);
  panel.appendChild(body);
  $("#messages").appendChild(panel);
  panel.classList.add("open");
  scrollToBottom();
  // 绑定树节点折叠
  bindTreeToggles(panel);
  return panel;
}

function renderWorkflowBody(data) {
  let html = "";

  // 1) Plan 步骤（横向流程）
  if (data.plan && data.plan.length) {
    html += `<div class="plan-flow">`;
    data.plan.forEach((s, i) => {
      const icon = s.status === "SUCCEEDED" ? "✅" : s.status === "FAILED" ? "❌" : s.status === "SKIPPED" ? "⏭️" : "⏳";
      const cls = s.status === "SUCCEEDED" ? "ok" : s.status === "FAILED" ? "fail" : "skip";
      html += `<div class="pf-step ${cls}">
        <span class="pf-icon">${icon}</span>
        <span class="pf-desc">${esc(s.description)}</span>
        ${s.result ? `<span class="pf-result">${esc(s.result.slice(0, 40))}</span>` : ""}
      </div>`;
      if (i < data.plan.length - 1) html += `<span class="pf-arrow">→</span>`;
    });
    html += `</div>`;
  }

  // 2) Trace 树（含耗时条）
  if (data.trace && data.trace.spans && data.trace.spans.length) {
    const root = data.trace.spans;
    const totalDur = Math.max(root.reduce((a, n) => a + (n.duration_ms || 0), 0), 1);
    html += `<div class="trace-tree">`;
    root.forEach((span) => { html += renderTraceNodeV2(span, 0, totalDur); });
    html += `</div>`;
  } else {
    html += `<div class="trace-tree"><div class="tn-row"><span class="tn-name">(Tracing 未启用)</span></div></div>`;
  }

  // 3) 工具时间线
  if (data.tool_calls && data.tool_calls.length) {
    html += `<div class="tool-timeline"><div class="tl-title">工具调用（${data.tool_calls.length}）</div>`;
    data.tool_calls.forEach((tc) => {
      html += `<div class="tl-item">
        <span class="tl-icon">🛠</span>
        <span class="tl-name">${esc(tc.name)}</span>
        <span class="tl-args">${esc(JSON.stringify(tc.arguments || {}))}</span>
      </div>`;
    });
    html += `</div>`;
  }

  return html;
}

function renderTraceNodeV2(span, depth, totalDur) {
  const isErr = span.status === "ERROR";
  const cls = isErr ? "err" : "ok";
  const icon = SPAN_ICONS[span.span_type] || SPAN_ICONS.generic;
  const dur = span.duration_ms || 0;
  const pct = Math.max((dur / totalDur) * 100, 0.5);
  const hasChildren = span.children && span.children.length;

  let html = `<div class="trace-node">`;
  html += `<div class="tn-row ${isErr ? "is-err" : ""}" style="padding-left:${depth * 14}px">`;
  // 折叠箭头
  html += hasChildren
    ? `<span class="tn-caret open" data-caret>▾</span>`
    : `<span class="tn-caret-placeholder"></span>`;
  html += `<span class="tn-icon">${icon}</span>`;
  html += `<span class="tn-name">${esc(span.name)}</span>`;
  if (span.attributes && span.attributes.tool_name) {
    html += `<span class="tn-tool">${esc(span.attributes.tool_name)}</span>`;
  }
  // 耗时条 + 数值
  html += `<span class="tn-bar-wrap"><span class="tn-bar" style="width:${Math.min(pct * 3, 60)}px"></span></span>`;
  html += `<span class="tn-dur">${dur.toFixed(dur >= 100 ? 0 : 1)}ms</span>`;
  html += `<span class="tn-status ${cls}">${isErr ? "❌" : "✓"}</span>`;
  html += `</div>`;
  // 错误详情
  if (span.error) {
    html += `<div class="tn-err" style="margin-left:${depth * 14 + 30}px">${esc(span.error.type || "Error")}: ${esc((span.error.message || "").slice(0, 150))}</div>`;
  }
  // 子节点
  if (hasChildren) {
    html += `<div class="tn-children" data-children>`;
    span.children.forEach((c) => { html += renderTraceNodeV2(c, depth + 1, totalDur); });
    html += `</div>`;
  }
  html += `</div>`;
  return html;
}

function bindTreeToggles(panel) {
  panel.querySelectorAll("[data-caret]").forEach((caret) => {
    caret.addEventListener("click", (e) => {
      e.stopPropagation();
      const row = caret.closest(".tn-row");
      const children = row.nextElementSibling && row.nextElementSibling.classList.contains("tn-children")
        ? row.nextElementSibling : null;
      if (!children) return;
      const open = caret.classList.contains("open");
      caret.classList.toggle("open", !open);
      children.style.display = open ? "none" : "";
    });
  });
}

/* ================= 发送 ================= */
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
    const resp = await fetch("/api/web/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, session_id: state.sessionId, agent_mode: state.agentMode }),
    });
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let finalAnswer = "";
    let finalData = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buffer.indexOf("\n\n")) !== -1) {
        const frame = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        handleFrame(frame, contentEl, (ans, data) => { finalAnswer = ans; finalData = data; });
      }
    }
    if (finalAnswer) contentEl.textContent = finalAnswer;
    assistantEl.classList.remove("streaming");
    if (finalData) {
      state.sessionId = finalData.session_id;
      // 工作流面板（自动展开）
      addWorkflowPanel({
        trace: finalData.trace, trace_id: finalData.trace_id,
        plan: finalData.plan, plan_revisions: finalData.plan_revisions,
        tool_calls: finalData.tool_calls,
      });
    }
  } catch (e) {
    assistantEl.classList.remove("streaming");
    contentEl.textContent = "⚠️ 请求失败: " + e.message;
  }
  setStreaming(false);
  loadSessions();
}

function handleFrame(frame, contentEl, onComplete) {
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
      onComplete(data.answer || "", data);
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

/* ================= 事件绑定 ================= */
function bindEvents() {
  const input = $("#input");
  const sendBtn = $("#send");
  const newBtn = $("#new-session");

  sendBtn.addEventListener("click", send);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
  });
  input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 120) + "px";
  });
  if (newBtn) newBtn.addEventListener("click", newSession);

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
