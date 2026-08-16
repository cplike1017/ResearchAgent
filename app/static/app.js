/* ReAgent Web UI 前端逻辑 v4 */
"use strict";

const state = {
  sessionId: null,
  agentMode: "react",
  streaming: false,
  abortCtrl: null, // 当前 SSE 的 AbortController（用于停止）
};

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

/* ================= 初始化 ================= */
async function init() {
  await Promise.all([loadCapabilities(), loadSessions(), loadAgents()]);
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

/* ================= 子 Agent 档案（动态注册） ================= */
async function loadAgents() {
  try {
    const data = await fetch("/api/web/agents").then((r) => r.json());
    const ul = $("#agent-list");
    ul.innerHTML = "";
    (data.agents || []).forEach((a) => {
      const li = document.createElement("li");
      li.className = "agent-item" + (a.builtin ? " builtin" : "");
      li.title = (a.description || "") + (a.allowed_tools ? `\n工具: ${a.allowed_tools.join(", ")}` : "\n工具: 全部");
      li.innerHTML = `
        <span class="si-icon">${a.builtin ? "📦" : "🧩"}</span>
        <span class="si-name">${esc(a.name)}</span>
        <span class="si-badge">${a.builtin ? "内置" : "自定义"}</span>
        ${a.builtin ? "" : `<span class="si-del" data-unregister="${esc(a.name)}" title="注销档案">✕</span>`}
      `;
      li.querySelector("[data-unregister]")?.addEventListener("click", (e) => {
        e.stopPropagation();
        unregisterAgent(a.name);
      });
      ul.appendChild(li);
    });
  } catch (e) { /* 忽略 */ }
}

async function unregisterAgent(name) {
  if (!confirm(`注销档案「${name}」？`)) return;
  try {
    const r = await fetch(`/api/web/agents/${encodeURIComponent(name)}`, { method: "DELETE" });
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      throw new Error(err.detail || r.statusText);
    }
    loadAgents();
    addToolMsg({ tool: "unregister_agent", arguments: { name }, success: true, data: "已注销" });
  } catch (e) {
    addErrorMsg("注销失败: " + e.message);
  }
}

async function registerAgent() {
  const name = $("#agent-name").value.trim();
  const description = $("#agent-desc").value.trim();
  const system_prompt = $("#agent-prompt").value.trim();
  if (!name || !system_prompt) {
    showAgentError("档案名与系统提示必填");
    return;
  }
  const toolsRaw = $("#agent-tools").value.trim();
  const allowed_tools = toolsRaw
    ? toolsRaw.split(/[,，]/).map((s) => s.trim()).filter(Boolean)
    : null;
  try {
    const r = await fetch("/api/web/agents", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, description, system_prompt, allowed_tools }),
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data.detail || r.statusText);
    $("#agent-form").style.display = "none";
    $("#agent-name").value = $("#agent-desc").value = $("#agent-prompt").value = $("#agent-tools").value = "";
    hideAgentError();
    loadAgents();
    addToolMsg({ tool: "register_agent", arguments: { name }, success: true, data: "注册成功，可立即用于编排" });
  } catch (e) {
    showAgentError("注册失败: " + e.message);
  }
}

function showAgentError(msg) { const el = $("#agent-error"); el.textContent = msg; el.style.display = "block"; }
function hideAgentError() { const el = $("#agent-error"); el.style.display = "none"; }

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
        <span class="si-del" title="删除会话">✕</span>
      `;
      if (state.sessionId === s.session_id) li.classList.add("active");
      li.addEventListener("click", (e) => {
        if (e.target.classList.contains("si-del")) {
          e.stopPropagation();
          deleteSession(s.session_id);
          return;
        }
        openSession(s.session_id);
      });
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
  loadOrchestrations(sessionId);
}

/* ================= 编排记录（委派结果持久化） ================= */
async function loadOrchestrations(sessionId) {
  try {
    const q = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : "";
    const data = await fetch(`/api/web/orchestrations${q}`).then((r) => r.json());
    const ul = $("#orch-list");
    ul.innerHTML = "";
    $("#orch-count").textContent = data.count || 0;
    (data.runs || []).forEach((run) => {
      const li = document.createElement("li");
      li.className = "orch-item";
      const icon = run.status === "SUCCEEDED" ? "✅" : run.status === "PARTIAL" ? "⚠️" : "❌";
      li.innerHTML = `
        <span class="si-icon">${icon}</span>
        <span class="si-name">${esc(run.task.slice(0, 18)) || "(无任务)"}</span>
        <span class="si-time">${run.depth > 1 ? `L${run.depth} · ` : ""}${esc(run.created_at.slice(11, 19))}</span>
      `;
      li.title = `${run.task}\n状态: ${run.status} · 子 Agent: ${run.agent_count} · ${(run.duration_ms / 1000).toFixed(1)}s`;
      li.dataset.runId = run.run_id;
      li.addEventListener("click", () => openOrchestrationDetail(run.run_id));
      ul.appendChild(li);
    });
  } catch (e) { /* 忽略 */ }
}

async function openOrchestrationDetail(runId) {
  try {
    const r = await fetch(`/api/web/orchestrations/${encodeURIComponent(runId)}`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const data = await r.json();
    renderOrchestrationPanel(data);
  } catch (e) {
    addErrorMsg("加载编排详情失败: " + e.message);
  }
}

function renderOrchestrationPanel(data) {
  const panel = document.createElement("div");
  panel.className = "workflow";
  panel.id = "orch-detail-" + data.run_id;

  const header = document.createElement("div");
  header.className = "workflow-header";
  const statusIcon = data.status === "SUCCEEDED" ? "✅" : data.status === "PARTIAL" ? "⚠️" : "❌";
  header.innerHTML = `
    <span class="wf-title">🌐 编排详情</span>
    <span class="wf-meta">${esc(data.run_id.slice(-12))} · ${statusIcon} ${esc(data.status)} · ${formatMs(data.duration_ms)}${data.depth > 1 ? ` · L${data.depth}` : ""}</span>
    <span class="wf-toggle">收起 ▴</span>
  `;
  const body = document.createElement("div");
  body.className = "workflow-body";
  body.innerHTML = renderOrchestrationBody(data);
  header.addEventListener("click", () => {
    panel.classList.toggle("open");
    header.querySelector(".wf-toggle").textContent = panel.classList.contains("open") ? "收起 ▴" : "展开 ▾";
  });
  panel.appendChild(header);
  panel.appendChild(body);
  // 插入到消息流顶部（最新编排）
  const messages = $("#messages");
  messages.insertBefore(panel, messages.firstChild);
  panel.classList.add("open");
  bindOrchestrationToggles(panel);
}

function renderOrchestrationBody(data) {
  let html = "";
  // 任务
  html += `<div class="orch-task">📋 ${esc(data.task)}</div>`;

  // 计划（分工）
  if (data.plan && data.plan.steps && data.plan.steps.length) {
    html += `<div class="orch-section"><div class="tl-title">分工计划 ${data.plan.rationale ? `（${esc(data.plan.rationale)}）` : ""}</div>`;
    html += `<div class="plan-flow">`;
    data.plan.steps.forEach((s, i) => {
      const deps = s.depends_on && s.depends_on.length ? ` ⬅${s.depends_on.join(",")}` : "";
      html += `<div class="pf-step ok">
        <span class="pf-icon">👤</span>
        <span class="pf-desc"><b>${esc(s.agent)}</b>${esc(deps)}</span>
      </div>`;
      if (i < data.plan.steps.length - 1) html += `<span class="pf-arrow">→</span>`;
    });
    html += `</div></div>`;
  }

  // 子 Agent 结果卡片
  if (data.agent_results && data.agent_results.length) {
    html += `<div class="orch-section"><div class="tl-title">子 Agent 结果（${data.agent_results.length}）</div>`;
    data.agent_results.forEach((ar, i) => {
      const icon = ar.status === "SUCCEEDED" ? "✅" : ar.status === "FAILED" ? "❌" : "⏭️";
      const tools = (ar.tool_calls || []).map((t) => t.name).join(", ") || "无工具调用";
      html += `
        <div class="subagent-card" data-idx="${i}">
          <div class="sa-header">
            <span class="sa-icon">${icon}</span>
            <span class="sa-name">${esc(ar.agent)}</span>
            <span class="sa-status ${ar.status === "SUCCEEDED" ? "ok" : "err"}">${esc(ar.status)}</span>
            <span class="sa-meta">${ar.steps} 步 · ${formatMs(ar.duration_ms)} · 🛠 ${esc(tools)}</span>
            <span class="ts-caret">▾</span>
          </div>
          <div class="sa-body">
            ${ar.error ? `<div class="ts-section"><div class="ts-label err-label">⚠️ 错误</div><pre class="ts-code err-code">${esc(ar.error)}</pre></div>` : ""}
            <div class="ts-section"><div class="ts-label">📤 回答</div>
              <div class="md-body">${window.marked && typeof window.marked.parse === "function" ? window.marked.parse(ar.answer || "(无输出)") : esc(ar.answer || "(无输出)")}</div>
            </div>
          </div>
        </div>`;
    });
    html += `</div>`;
  }

  // 嵌套子编排（多级）
  if (data.children && data.children.length) {
    html += `<div class="orch-section"><div class="tl-title">嵌套子编排（${data.children.length}）</div>`;
    data.children.forEach((c) => {
      const icon = c.status === "SUCCEEDED" ? "✅" : "⚠️";
      html += `<div class="orch-child" data-child="${esc(c.run_id)}">${icon} L${c.depth} · ${esc(c.task.slice(0, 40))}</div>`;
    });
    html += `</div>`;
  }

  // 最终答案
  if (data.final_answer) {
    html += `<div class="orch-section"><div class="tl-title">最终合成答案</div>`;
    html += `<div class="md-body">${window.marked && typeof window.marked.parse === "function" ? window.marked.parse(data.final_answer) : esc(data.final_answer)}</div></div>`;
  }

  // Trace 树
  if (data.trace && data.trace.spans && data.trace.spans.length) {
    const root = data.trace.spans;
    const totalDur = Math.max(root.reduce((a, n) => a + (n.duration_ms || 0), 0), 1);
    html += `<div class="orch-section"><div class="tl-title">Trace 树</div><div class="trace-tree">`;
    root.forEach((span) => { html += renderTraceNodeV2(span, 0, totalDur); });
    html += `</div></div>`;
  }
  return html;
}

function bindOrchestrationToggles(panel) {
  // 子 agent 卡片展开
  panel.querySelectorAll(".subagent-card").forEach((card) => {
    card.querySelector(".sa-header").addEventListener("click", () => {
      card.classList.toggle("open");
      card.querySelector(".ts-caret").textContent = card.classList.contains("open") ? "▴" : "▾";
    });
  });
  // 嵌套子编排：点击加载详情
  panel.querySelectorAll(".orch-child").forEach((el) => {
    el.addEventListener("click", () => openOrchestrationDetail(el.dataset.child));
  });
  // Trace 树折叠/详情
  bindTreeToggles(panel);
  bindToolStepToggles(panel);
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
      <h2>🤖 ReAgent</h2>
      <p>ReAct / Plan 双模式 · 记忆 · MCP · 技能 · 多 Agent 编排</p>
      <p class="sub">工作流完全透明：每步决策、工具调用、耗时、Trace 树实时可见</p>
    </div>`;
  $$("#session-list .session-item").forEach((el) => el.classList.remove("active"));
  // 清空编排记录列表
  $("#orch-list").innerHTML = "";
  $("#orch-count").textContent = "0";
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
  if (role === "assistant" && window.marked && typeof window.marked.parse === "function") {
    // Markdown 渲染
    contentEl.className = "md-body";
    contentEl.innerHTML = window.marked.parse(String(content || ""));
    // 复制按钮（assistant 消息）
    const actions = document.createElement("div");
    actions.className = "msg-actions";
    const copyBtn = document.createElement("button");
    copyBtn.className = "copy-btn";
    copyBtn.textContent = "复制";
    copyBtn.addEventListener("click", () => {
      navigator.clipboard.writeText(String(content || "")).then(() => {
        copyBtn.textContent = "已复制 ✓";
        setTimeout(() => (copyBtn.textContent = "复制"), 1500);
      });
    });
    actions.appendChild(copyBtn);
    div.appendChild(actions);
  } else {
    contentEl.textContent = content;
  }
  div.appendChild(contentEl);
  $("#messages").appendChild(div);
  scrollToBottom();
  return div;
}

function addToolMsg(data) {
  // 流式中的工具调用卡片：点击可展开查看参数/输出/耗时
  const div = document.createElement("div");
  div.className = "msg tool ts-card";
  div.dataset.toolCallId = data.tool_call_id || data.tool || "";
  const dur = data.duration_ms != null ? ` · ⏱ ${formatMs(data.duration_ms)}` : "";
  const statusIcon = data.success === false ? "❌" : "✅";
  div.innerHTML = `
    <div class="ts-header">
      <span class="ts-icon">${data.tool === "delegate" ? "🌐" : "🛠"}</span>
      <span class="ts-name">${esc(data.tool)}</span>
      <span class="ts-status ${data.success === false ? "err" : "ok"}">${statusIcon}</span>
      <span class="ts-dur">${dur}</span>
      <span class="ts-args-preview">${esc(JSON.stringify(data.arguments || {}).slice(0, 50))}</span>
      <span class="ts-caret">▾</span>
    </div>
    <div class="ts-body">
      <div class="ts-section">
        <div class="ts-label">📋 参数</div>
        <pre class="ts-code">${esc(JSON.stringify(data.arguments || {}, null, 2))}</pre>
      </div>
      ${data.data && data.data !== "等待执行..." ? `
      <div class="ts-section">
        <div class="ts-label">📤 输出</div>
        ${toolOutputHtml(data.tool, data.data)}
      </div>` : `<div class="ts-section"><div class="ts-label">⏳ 等待执行...</div></div>`}
      ${data.error ? `
      <div class="ts-section">
        <div class="ts-label err-label">⚠️ 错误</div>
        <pre class="ts-code err-code">${esc(data.error.message || JSON.stringify(data.error))}</pre>
      </div>` : ""}
    </div>
  `;
  // 点击展开
  div.querySelector(".ts-header").addEventListener("click", () => {
    div.classList.toggle("open");
    const caret = div.querySelector(".ts-caret");
    caret.textContent = div.classList.contains("open") ? "▴" : "▾";
  });
  $("#messages").appendChild(div);
  scrollToBottom();
  return div;
}

// delegate 等编排工具的输出：结构化渲染而非原始 JSON
function toolOutputHtml(tool, raw) {
  if (tool === "delegate") {
    const summary = parseDelegateOutput(raw);
    if (summary) {
      const icon = summary.status === "SUCCEEDED" ? "✅" : summary.status === "PARTIAL" ? "⚠️" : "❌";
      let html = `<div class="delegate-summary">
        <div class="ds-row"><span>${icon} 编排状态: <b>${esc(summary.status)}</b></span>
        <span>${summary.duration_ms != null ? `⏱ ${formatMs(summary.duration_ms)}` : ""}</span></div>`;
      if (summary.agents && summary.agents.length) {
        html += `<div class="ds-agents">${summary.agents.map((a) => `
          <span class="ds-agent ${a.status === "SUCCEEDED" ? "ok" : "err"}">${a.status === "SUCCEEDED" ? "✅" : "❌"} ${esc(a.agent)}</span>`).join("")}</div>`;
      }
      if (summary.final_answer) {
        html += `<div class="ds-answer">${esc(String(summary.final_answer).slice(0, 300))}</div>`;
      }
      html += `</div>`;
      return html;
    }
  }
  return `<pre class="ts-code">${esc(String(raw).slice(0, 300))}</pre>`;
}

// 解析 delegate 工具输出：可能是对象或 JSON 字符串
function parseDelegateOutput(raw) {
  try {
    let obj = raw;
    if (typeof raw === "string") {
      try { obj = JSON.parse(raw); } catch (e) { return null; }
    }
    // 工具信封（data 字段包装）
    if (obj && typeof obj === "object" && "data" in obj && !("status" in obj)) {
      const inner = obj.data;
      if (typeof inner === "string") { try { return JSON.parse(inner); } catch (e) { return null; } }
      return inner;
    }
    return obj && typeof obj === "object" && "agent_results" in obj ? obj : null;
  } catch (e) {
    return null;
  }
}

// 按工具名找到最近一张卡片（流式结果回填）
function updateToolCard(name, data) {
  const cards = $$("#messages .ts-card[data-tool-call-id]");
  for (let i = cards.length - 1; i >= 0; i--) {
    const card = cards[i];
    if (card.dataset.toolCallId === name || (card.dataset.toolCallId === "" && card.querySelector(".ts-name").textContent === name)) {
      // 更新状态和输出
      const statusEl = card.querySelector(".ts-status");
      statusEl.textContent = data.success === false ? "❌" : "✅";
      statusEl.className = "ts-status " + (data.success === false ? "err" : "ok");
      if (data.duration_ms != null) {
        const durEl = card.querySelector(".ts-dur");
        durEl.textContent = " · ⏱ " + formatMs(data.duration_ms);
      }
      const body = card.querySelector(".ts-body");
      if (data.data && data.data !== "等待执行...") {
        body.innerHTML = `
          <div class="ts-section">
            <div class="ts-label">📋 参数</div>
            <pre class="ts-code">${esc(JSON.stringify(data.arguments || {}, null, 2))}</pre>
          </div>
          <div class="ts-section">
            <div class="ts-label">📤 输出</div>
            ${toolOutputHtml(data.tool, data.data)}
          </div>
          ${data.error ? `
          <div class="ts-section">
            <div class="ts-label err-label">⚠️ 错误</div>
            <pre class="ts-code err-code">${esc(data.error.message || JSON.stringify(data.error))}</pre>
          </div>` : ""}
        `;
      }
      return true;
    }
  }
  return false;
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
  // 绑定树节点折叠 + 工具卡片展开
  bindTreeToggles(panel);
  bindToolStepToggles(panel);
  return panel;
}

function bindToolStepToggles(panel) {
  panel.querySelectorAll(".ts-card").forEach((card) => {
    const header = card.querySelector(".ts-header");
    header.addEventListener("click", () => {
      card.classList.toggle("open");
      const caret = card.querySelector(".ts-caret");
      caret.textContent = card.classList.contains("open") ? "▴" : "▾";
    });
  });
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

  // 3) 工具调用步骤卡片（可点击展开详情）
  if (data.tool_calls && data.tool_calls.length) {
    html += `<div class="tool-steps"><div class="tl-title">工具调用流程（${data.tool_calls.length} 步）</div>`;
    data.tool_calls.forEach((tc, i) => {
      html += renderToolStepCard(tc, i);
    });
    html += `</div>`;
  }

  return html;
}

function renderToolStepCard(tc, index) {
  const dur = tc.duration_ms != null ? formatMs(tc.duration_ms) : "";
  const statusIcon = tc.status === "ERROR" ? "❌" : tc.error ? "❌" : "✅";
  const statusCls = tc.status === "ERROR" || tc.error ? "err" : "ok";
  const argsPreview = JSON.stringify(tc.arguments || {}).slice(0, 60);
  const dataStr = tc.data ? String(tc.data).slice(0, 120) : "";
  const errStr = tc.error ? (tc.error.message || JSON.stringify(tc.error)).slice(0, 120) : "";
  return `
    <div class="ts-card" data-index="${index}">
      <div class="ts-header">
        <span class="ts-num">${index + 1}</span>
        <span class="ts-icon">🛠</span>
        <span class="ts-name">${esc(tc.name)}</span>
        <span class="ts-status ${statusCls}">${statusIcon}</span>
        ${dur ? `<span class="ts-dur">⏱ ${dur}</span>` : ""}
        <span class="ts-args-preview">${esc(argsPreview)}</span>
        <span class="ts-caret">▾</span>
      </div>
      <div class="ts-body">
        <div class="ts-section">
          <div class="ts-label">📋 参数</div>
          <pre class="ts-code">${esc(JSON.stringify(tc.arguments || {}, null, 2))}</pre>
        </div>
        ${dataStr ? `
        <div class="ts-section">
          <div class="ts-label">📤 输出</div>
          <pre class="ts-code">${esc(dataStr)}${String(tc.data).length > 120 ? "\n..." : ""}</pre>
        </div>` : ""}
        ${errStr ? `
        <div class="ts-section">
          <div class="ts-label err-label">⚠️ 错误</div>
          <pre class="ts-code err-code">${esc(errStr)}</pre>
        </div>` : ""}
        ${dur ? `
        <div class="ts-meta">⏱ 耗时: ${dur}</div>` : ""}
      </div>
    </div>`;
}

function formatMs(ms) {
  if (ms == null) return "";
  if (ms >= 1000) return (ms / 1000).toFixed(1) + "s";
  return Math.round(ms) + "ms";
}

function renderTraceNodeV2(span, depth, totalDur) {
  const isErr = span.status === "ERROR";
  const cls = isErr ? "err" : "ok";
  const icon = SPAN_ICONS[span.span_type] || SPAN_ICONS.generic;
  const dur = span.duration_ms || 0;
  const pct = Math.max((dur / totalDur) * 100, 0.5);
  const hasChildren = span.children && span.children.length;
  // 详情是否有内容（决定是否可展开）
  const hasDetails = span.error || span.input !== undefined || span.output !== undefined
    || (span.attributes && Object.keys(span.attributes).length);

  let html = `<div class="trace-node">`;
  // 行（点击展开详情；caret 折叠子节点）
  html += `<div class="tn-row ${isErr ? "is-err" : ""} ${hasDetails ? "clickable" : ""}" style="padding-left:${depth * 14}px">`;
  html += hasChildren
    ? `<span class="tn-caret open" data-caret>▾</span>`
    : `<span class="tn-caret-placeholder"></span>`;
  html += `<span class="tn-icon">${icon}</span>`;
  html += `<span class="tn-name">${esc(span.name)}</span>`;
  if (span.attributes && span.attributes.tool_name) {
    html += `<span class="tn-tool">${esc(span.attributes.tool_name)}</span>`;
  }
  html += `<span class="tn-bar-wrap"><span class="tn-bar" style="width:${Math.min(pct * 3, 60)}px"></span></span>`;
  html += `<span class="tn-dur">${dur.toFixed(dur >= 100 ? 0 : 1)}ms</span>`;
  html += `<span class="tn-status ${cls}">${isErr ? "❌" : "✓"}</span>`;
  // 详情展开指示（有详情才显示）
  html += hasDetails ? `<span class="tn-expand" data-expand>详情 ▾</span>` : "";
  html += `</div>`;

  // 详情面板（点击行展开）
  if (hasDetails) {
    html += `<div class="tn-detail" data-detail style="margin-left:${depth * 14 + 30}px">`;
    html += renderSpanDetail(span);
    html += `</div>`;
  }

  // 错误详情（保留在行内，展开面板里也有）
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

function renderSpanDetail(span) {
  let html = "";
  // 元信息
  html += `<div class="tn-detail-meta">`;
  if (span.span_id) html += `<span class="ts-meta">id: ${esc(span.span_id.slice(-12))}</span>`;
  if (span.start_time) html += `<span class="ts-meta">start: ${esc(span.start_time.slice(11, 19))}</span>`;
  if (span.end_time) html += `<span class="ts-meta">end: ${esc(span.end_time.slice(11, 19))}</span>`;
  if (span.duration_ms != null) html += `<span class="ts-meta">耗时: ${formatMs(span.duration_ms)}</span>`;
  html += `</div>`;

  // attributes（如 model / tokens / tool_name）
  if (span.attributes && Object.keys(span.attributes).length) {
    html += `<div class="ts-section"><div class="ts-label">🏷 属性</div>`;
    html += `<pre class="ts-code">${esc(JSON.stringify(span.attributes, null, 2))}</pre></div>`;
  }
  // input
  if (span.input !== undefined && span.input !== null) {
    html += `<div class="ts-section"><div class="ts-label">📥 输入</div>`;
    html += `<pre class="ts-code">${esc(JSON.stringify(span.input, null, 2).slice(0, 600))}</pre></div>`;
  }
  // output
  if (span.output !== undefined && span.output !== null) {
    html += `<div class="ts-section"><div class="ts-label">📤 输出</div>`;
    html += `<pre class="ts-code">${esc(JSON.stringify(span.output, null, 2).slice(0, 600))}</pre></div>`;
  }
  // error
  if (span.error) {
    html += `<div class="ts-section"><div class="ts-label err-label">⚠️ 错误</div>`;
    html += `<pre class="ts-code err-code">${esc(JSON.stringify(span.error, null, 2))}</pre></div>`;
  }
  return html;
}

function bindTreeToggles(panel) {
  // caret：折叠子节点
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
  // 行点击：展开/收起详情面板
  panel.querySelectorAll(".tn-row.clickable").forEach((row) => {
    row.addEventListener("click", (e) => {
      if (e.target.closest("[data-caret]")) return; // caret 已处理
      const detail = row.nextElementSibling && row.nextElementSibling.classList.contains("tn-detail")
        ? row.nextElementSibling : null;
      if (!detail) return;
      const expand = row.querySelector("[data-expand]");
      const isOpen = detail.classList.contains("open");
      detail.classList.toggle("open", !isOpen);
      if (expand) expand.textContent = isOpen ? "详情 ▾" : "详情 ▴";
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
  contentEl.className = "md-body";
  contentEl.textContent = "";

  // 停止按钮
  state.abortCtrl = new AbortController();
  $("#stop").style.display = "block";

  try {
    const resp = await fetch("/api/web/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, session_id: state.sessionId, agent_mode: state.agentMode }),
      signal: state.abortCtrl.signal,
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
    if (finalAnswer) {
      contentEl.innerHTML = (window.marked && typeof window.marked.parse === "function")
        ? window.marked.parse(finalAnswer) : esc(finalAnswer);
      // 补复制按钮
      ensureCopyButton(assistantEl, finalAnswer);
    }
    assistantEl.classList.remove("streaming");
    if (finalData) {
      state.sessionId = finalData.session_id;
      addWorkflowPanel({
        trace: finalData.trace, trace_id: finalData.trace_id,
        plan: finalData.plan, plan_revisions: finalData.plan_revisions,
        tool_calls: finalData.tool_calls,
      });
    }
  } catch (e) {
    if (e.name === "AbortError") {
      contentEl.textContent = "⏹ 已停止生成。";
    } else {
      assistantEl.classList.remove("streaming");
      contentEl.textContent = "⚠️ 请求失败: " + e.message;
    }
  }
  $("#stop").style.display = "none";
  setStreaming(false);
  loadSessions();
  // 编排记录可能新增（delegate 工具）
  if (state.sessionId) loadOrchestrations(state.sessionId);
}

function ensureCopyButton(assistantEl, content) {
  if (assistantEl.querySelector(".copy-btn")) return;
  const actions = document.createElement("div");
  actions.className = "msg-actions";
  const copyBtn = document.createElement("button");
  copyBtn.className = "copy-btn";
  copyBtn.textContent = "复制";
  copyBtn.addEventListener("click", () => {
    navigator.clipboard.writeText(content).then(() => {
      copyBtn.textContent = "已复制 ✓";
      setTimeout(() => (copyBtn.textContent = "复制"), 1500);
    });
  });
  actions.appendChild(copyBtn);
  assistantEl.appendChild(actions);
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
      // 回填对应的卡片（按工具名匹配最近的未完成卡片）
      if (!updateToolCard(data.tool, data)) {
        addToolMsg(data); // 找不到则新增
      }
      break;
    case "final":
      contentEl.innerHTML = (window.marked && typeof window.marked.parse === "function")
        ? window.marked.parse(data.content || "") : esc(data.content || "");
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
  if (!v) state.abortCtrl = null;
}

function stopStreaming() {
  if (state.abortCtrl) {
    state.abortCtrl.abort();
    $("#stop").style.display = "none";
  }
}

/* ================= 事件绑定 ================= */
function bindEvents() {
  const input = $("#input");
  const sendBtn = $("#send");
  const newBtn = $("#new-session");
  const stopBtn = $("#stop");
  const uploadBtn = $("#upload-btn");
  const fileInput = $("#file-input");

  sendBtn.addEventListener("click", send);
  if (stopBtn) stopBtn.addEventListener("click", stopStreaming);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
  });
  input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 120) + "px";
  });
  if (newBtn) newBtn.addEventListener("click", newSession);

  // 上传文件
  if (uploadBtn && fileInput) {
    uploadBtn.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", async () => {
      const file = fileInput.files[0];
      if (!file) return;
      const fd = new FormData();
      fd.append("file", file);
      try {
        const r = await fetch("/api/web/upload", { method: "POST", body: fd });
        const data = await r.json();
        addToolMsg({ tool: "upload", arguments: { file: file.name }, success: true, data: data.hint });
        loadFiles();
      } catch (e) {
        addErrorMsg("上传失败: " + e.message);
      }
      fileInput.value = "";
    });
  }

  document.querySelectorAll(".mode-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".mode-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.agentMode = btn.dataset.mode;
      $("#mode-badge").textContent = btn.dataset.mode;
    });
  });

  // 档案注册表单
  const addAgentBtn = $("#add-agent-btn");
  if (addAgentBtn) {
    addAgentBtn.addEventListener("click", () => {
      const form = $("#agent-form");
      form.style.display = form.style.display === "none" ? "block" : "none";
      if (form.style.display === "block") $("#agent-name").focus();
    });
  }
  $("#agent-save")?.addEventListener("click", registerAgent);
  $("#agent-cancel")?.addEventListener("click", () => {
    $("#agent-form").style.display = "none";
    hideAgentError();
  });
}

/* ================= 文件列表 ================= */
async function loadFiles() {
  try {
    const data = await fetch("/api/web/files").then((r) => r.json());
    renderList("#file-list", (data.files || []).map((f) => ({
      text: `${f.name} (${(f.size / 1024).toFixed(1)}KB)`,
      title: f.name,
    })));
  } catch (e) { /* 忽略 */ }
}

/* ================= 会话删除 ================= */
async function deleteSession(sessionId) {
  if (!confirm(`删除会话 ${sessionId.slice(-12)}？`)) return;
  try {
    await fetch(`/api/web/sessions/${encodeURIComponent(sessionId)}`, { method: "DELETE" });
    if (state.sessionId === sessionId) newSession();
    loadSessions();
    if (state.sessionId) loadOrchestrations(state.sessionId);
  } catch (e) {
    addErrorMsg("删除失败: " + e.message);
  }
}

init();
loadFiles();
