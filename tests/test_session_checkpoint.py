"""
Stage 3 测试：Session 持久化 + Checkpoint 版本 / 断点恢复。

验收点：
    正常执行 + 中途 Crash + Checkpoint 恢复；
    恢复必须来自 Checkpoint（不重放用户输入）；
    version 递增且旧 Checkpoint 不覆盖新状态。
"""
import pytest

from app.agent.react_loop import LoopHooks
from app.errors import CheckpointError


class SimulatedCrash(RuntimeError):
    """模拟进程崩溃：在指定钩子处抛出。"""


def _crash(*args, **kwargs):
    raise SimulatedCrash("模拟进程崩溃")


# ---------------------------------------------------------------------------
# Session 仓库
# ---------------------------------------------------------------------------
def test_session_create_and_get(session_repo):
    rec = session_repo.create_session("session_abc")
    assert rec.session_id == "session_abc"
    assert rec.status.value == "ACTIVE"
    got = session_repo.get_session("session_abc")
    assert got is not None
    assert got.session_id == "session_abc"
    assert session_repo.get_session("nope") is None


def test_message_persistence_roundtrip(session_repo):
    session_repo.create_session("session_roundtrip")
    session_repo.add_message("session_roundtrip", {"role": "user", "content": "你好"})
    session_repo.add_message(
        "session_roundtrip",
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function"}]},
    )
    msgs = session_repo.list_messages("session_roundtrip")
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[1]["tool_calls"][0]["id"] == "c1"  # 复杂结构完整往返


def test_session_status_update(session_repo):
    session_repo.create_session("session_status")
    session_repo.update_status("session_status", "CLOSED")
    assert session_repo.get_session("session_status").status.value == "CLOSED"


# ---------------------------------------------------------------------------
# Checkpoint 仓库：版本
# ---------------------------------------------------------------------------
def test_checkpoint_version_increments(checkpoint_repo):
    for step in (1, 2, 3):
        checkpoint_repo.save(
            session_id="session_ver", turn_id="turn_1", step=step, state={"step": step, "status": "RUNNING"}
        )
    assert checkpoint_repo.versions("session_ver") == [1, 2, 3]
    latest = checkpoint_repo.load_latest("session_ver")
    assert latest.version == 3
    assert latest.state["step"] == 3


def test_checkpoint_old_does_not_overwrite_new(checkpoint_repo):
    """追加写：旧 Checkpoint 不可能覆盖新状态。"""
    checkpoint_repo.save(session_id="s", turn_id="t", step=1, state={"status": "OLD"})
    checkpoint_repo.save(session_id="s", turn_id="t", step=2, state={"status": "NEW"})
    assert checkpoint_repo.load_latest("s").state["status"] == "NEW"
    assert checkpoint_repo.load_at_version("s", 1).state["status"] == "OLD"
    assert checkpoint_repo.load_at_version("s", 2).state["status"] == "NEW"
    assert len(checkpoint_repo.versions("s")) == 2  # 两个版本都保留


# ---------------------------------------------------------------------------
# 运行时：Session 持久化
# ---------------------------------------------------------------------------
async def test_runtime_persists_messages_to_session(full_runtime):
    await full_runtime.run("查询北京天气", session_id="session_persist")
    msgs = full_runtime.session_repo.list_messages("session_persist")
    roles = [m["role"] for m in msgs]
    assert roles == ["user", "assistant", "tool", "assistant"]


async def test_multi_turn_history_accumulates(full_runtime):
    await full_runtime.run("查询北京天气", session_id="session_multi")
    await full_runtime.run("你好", session_id="session_multi")
    msgs = full_runtime.session_repo.list_messages("session_multi")
    # 第一回合 4 条 + 第二回合 2 条
    assert len(msgs) == 6


async def test_runtime_creates_checkpoints(full_runtime):
    await full_runtime.run("查询北京天气", session_id="session_ckpt")
    versions = full_runtime.checkpoint_repo.versions("session_ckpt")
    # 第1轮: before_llm / after_decision / after_tool
    # 第2轮: before_llm / after_decision(无工具) / before_final
    assert versions == [1, 2, 3, 4, 5, 6]


# ---------------------------------------------------------------------------
# 断点恢复：LLM 决策后崩溃（PENDING_TOOL）
# ---------------------------------------------------------------------------
async def test_checkpoint_recovery_after_crash(full_runtime):
    """用户请求 -> LLM 决定调用 Tool -> 保存 Checkpoint -> 崩溃 -> 恢复继续。"""
    session_id = "session_crash_pending"

    # 第一次执行：after_decision 钩子触发崩溃（此时"LLM 决策后"检查点已保存）
    with pytest.raises(SimulatedCrash):
        await full_runtime.run(
            "查询北京天气", session_id=session_id, extra_hooks=LoopHooks(after_decision=_crash)
        )

    # 检查点已保存：状态为 PENDING_TOOL，待执行 get_weather
    cp = full_runtime.checkpoint_repo.load_latest(session_id)
    assert cp is not None
    assert cp.state["status"] == "PENDING_TOOL"
    assert cp.state["pending_tool_calls"][0]["name"] == "get_weather"
    assert cp.state["pending_tool_calls"][0]["arguments"] == {"city": "北京"}
    version_before = cp.version

    # 恢复：从 Checkpoint 继续，不重放用户输入
    result = await full_runtime.resume(session_id)
    assert "北京" in result.answer

    # 恢复后产生新版本（避免旧 Checkpoint 覆盖新状态）
    cp2 = full_runtime.checkpoint_repo.load_latest(session_id)
    assert cp2.version > version_before

    # 证明：用户消息只出现一次（没有从头完整重跑）
    user_msgs = [m for m in result.messages if m["role"] == "user"]
    assert len(user_msgs) == 1
    assert user_msgs[0]["content"] == "查询北京天气"


async def test_recovery_from_running_state(full_runtime):
    """工具执行完成后崩溃（RUNNING）：恢复时直接继续循环。"""
    session_id = "session_crash_running"
    with pytest.raises(SimulatedCrash):
        await full_runtime.run(
            "计算 123 * 456", session_id=session_id, extra_hooks=LoopHooks(after_tool=_crash)
        )
    cp = full_runtime.checkpoint_repo.load_latest(session_id)
    assert cp.state["status"] == "RUNNING"
    roles = [m["role"] for m in cp.state["messages"]]
    assert roles == ["user", "assistant", "tool"]  # 工具结果已在消息中

    result = await full_runtime.resume(session_id)
    assert "56088" in result.answer


async def test_resume_without_checkpoint_raises(full_runtime):
    with pytest.raises(CheckpointError):
        await full_runtime.resume("session_no_ckpt")


# ---------------------------------------------------------------------------
# Session 与 Checkpoint 的区别（数据视角）
# ---------------------------------------------------------------------------
def test_session_vs_checkpoint_difference(session_repo, checkpoint_repo):
    """Session 保存业务数据（消息历史）；Checkpoint 保存执行快照（状态机）。"""
    session_repo.create_session("s1")
    session_repo.add_message("s1", {"role": "user", "content": "你好"})
    checkpoint_repo.save(session_id="s1", turn_id="t1", step=3, state={"status": "PENDING_TOOL", "step": 3})

    session_messages = session_repo.list_messages("s1")
    assert len(session_messages) == 1

    cp = checkpoint_repo.load_latest("s1")
    assert cp.state["status"] == "PENDING_TOOL"  # 执行状态，不是消息
    assert cp.step == 3
