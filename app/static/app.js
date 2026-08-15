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

/* ---------------- 工作流面板（Trace 树可视化）---------------- */
function addWorkflowPanel(data) {
  const panel = document.createElement("div");
  panel.className = "workflow";
  panel.id = "wf-" + (data.trace_id || Date.now());

  // Header（点击折叠）
  const header = document.createElement("div");
  header.className = "workflow-header";
  const planInfo = data.plan && data.plan.length
    ? ` · 计划 ${data.plan.length} 步${data.plan_revisions ? ` · 重规划 ${data.plan_revisions}` : ""}`
    : "";
  header.innerHTML = `
    <span class="wf-title">🔍 Agent 工作流</span>
    <span class="wf-meta">${data.trace_id ? data.trace_id.slice(-12) : ""}${planInfo}</span>
    <span class="wf-toggle">展开 ▾</span>
  `;

  // Body
  const body = document.createElement("div");
  body.className = "workflow-body";

  let html = "";

  // 1) Plan 步骤
  if (data.plan && data.plan.length) {
    html += `<div class="plan-steps">`;
    data.plan.forEach((s) => {
      const icon = s.status === "SUCCEEDED" ? "✅" : s.status === "FAILED" ? "❌" : s.status === "SKIPPED" ? "⏭️" : "⏳";
      const cls = s.status === "SUCCEEDED" ? "ok" : s.status === "FAILED" ? "fail" : "skip";
      html += `<div class="plan-step">
        <span class="ps-status ${cls}">${icon}</span>
        <span class="ps-desc">${esc(s.description)}</span>
        <span class="ps-result">${esc((s.result || "").slice(0, 50))}</span>
      </div>`;
    });
    html += `</div>`;
  }

  // 2) Trace 树
  if (data.trace && data.trace.spans && data.trace.spans.length) {
    html += `<div class="trace-tree">`;
    data.trace.spans.forEach((span) => {
      html += renderTraceNode(span, 0);
    });
    html += `</div>`;
  } else {
    html += `<div class="trace-tree"><div class="tn-row"><span class="tn-name">(Tracing 未启用)</span></div></div>`;
  }

  // 3) 工具调用时间线
  if (data.tool_calls && data.tool_calls.length) {
    html += `<div class="tool-timeline"><div class="tl-title">工具调用（${data.tool_calls.length}）</div>`;
    data.tool_calls.forEach((tc, i) => {
      html += `<div class="tl-item">
        <span class="tl-icon">🛠</span>
        <span class="tl-name">${esc(tc.name)}</span>
        <span class="tl-args">${esc(JSON.stringify(tc.arguments || {}))}</span>
      </div>`;
    });
    html += `</div>`;
  }

  body.innerHTML = html;
  header.addEventListener("click", () => {
    panel.classList.toggle("open");
    header.querySelector(".wf-toggle").textContent = panel.classList.contains("open") ? "收起 ▴" : "展开 ▾";
  });
  panel.appendChild(header);
  panel.appendChild(body);
  $("#messages").appendChild(panel);
  scrollToBottom();
  // 自动展开
  panel.classList.add("open");
  header.querySelector(".wf-toggle").textContent = "收起 ▴";
}

function renderTraceNode(span, depth) {
  const statusCls = span.status === "ERROR" ? "err" : "ok";
  const statusIcon = span.status === "ERROR" ? "❌" : "✓";
  const icons = {
    "gateway": "🚪", "worker": "⚙️", "agent": "🧠", "llm": "💬", "tool": "🛠",
    "tool_gateway": "🛡", "context_builder": "📦", "checkpoint": "💾", "queue": "📨",
    "memory": "🧠", "eval": "📊",
  };
  const icon = icons[span.span_type] || "·";
  let html = `<div class="trace-node">`;
  html += `<div class="tn-row" style="padding-left:${depth * 12}px">
    <span class="tn-icon">${icon}</span>
    <span class="tn-name">${esc(span.name)}</span>
    <span class="tn-type">${esc(span.span_type)}</span>
    ${span.attributes && span.attributes.tool_name ? `<span class="tn-tool">${esc(span.attributes.tool_name)}</span>` : ""}
    <span class="tn-status ${statusCls}">${statusIcon}</span>
    <span class="tn-dur">${span.duration_ms != null ? span.duration_ms.toFixed(1) + "ms" : ""}</span>
  </div>`;
  if (span.error) {
    html += `<div class="tn-err">${esc(span.error.type || "Error")}: ${esc((span.error.message || "").slice(0, 120))}</div>`;
  }
  if (span.children && span.children.length) {
    html += `<div class="tn-children">`;
    span.children.forEach((c) => { html += renderTraceNode(c, depth + 1); });
    html += `</div>`;
  }
  html += `</div>`;
  return html;
}

function esc(s) {
  const div = document.createElement("div");
  div.textContent = s == null ? "" : String(s);
  return div.innerHTML;
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
    // 工作流面板（含 Trace 树）
    if (finalData) {
      addWorkflowPanel({
        trace: finalData.trace,
        trace_id: finalData.trace_id,
        plan: finalData.plan,
        plan_revisions: finalData.plan_revisions,
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
      state.sessionId = data.session_id;
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
