"""
Stage 3 Demo：Session + Checkpoint 断点恢复。

运行：python -m demos.stage3_demo

流程：
    用户请求
      ↓
    LLM 决定调用 Tool
      ↓
    保存 Checkpoint（LLM 决策后）
      ↓
    模拟进程 Crash
      ↓
    （重新启动：新建运行时实例，指向同一 SQLite）
      ↓
    加载最新 Checkpoint
      ↓
    继续执行（不重放用户输入）
      ↓
    Final Answer

关键证明：恢复后的 State 来自 Checkpoint，而非用户输入。
"""
import asyncio
import json

from app.agent.react_loop import LoopHooks
from app.agent.runtime import AgentRuntime
from app.checkpoint.repository import SQLiteCheckpointRepository
from app.config import get_settings
from app.session.repository import SQLiteSessionRepository


class SimulatedCrash(RuntimeError):
    """模拟进程崩溃。"""


def _crash_after_decision(*args, **kwargs):
    raise SimulatedCrash("模拟进程崩溃（LLM 决策后、工具执行前）")


async def main() -> None:
    settings = get_settings()
    session_id = "stage3_demo_session"

    # ------------------------------------------------------------------
    print("=" * 64)
    print("第一步：正常执行，LLM 决策后模拟崩溃")
    print("=" * 64)
    session_repo = SQLiteSessionRepository(settings.database_url)
    checkpoint_repo = SQLiteCheckpointRepository(settings.database_url)
    runtime = AgentRuntime(settings=settings, session_repo=session_repo, checkpoint_repo=checkpoint_repo)

    try:
        await runtime.run(
            "查询北京天气",
            session_id=session_id,
            extra_hooks=LoopHooks(after_decision=_crash_after_decision),
        )
        print(">>> 未崩溃（不应该走到这里）")
    except SimulatedCrash:
        print(">>> [崩溃] 进程在此处中断（after_decision 检查点已保存）")

    # 展示崩溃前保存的检查点
    cp = checkpoint_repo.load_latest(session_id)
    print(f"\n--- 崩溃点检查点 ---")
    print(f"checkpoint_id : {cp.checkpoint_id}")
    print(f"version       : {cp.version}")
    print(f"turn_id       : {cp.turn_id}")
    print(f"step          : {cp.step}")
    print(f"恢复前 state  :")
    print(json.dumps(cp.state, ensure_ascii=False, indent=2))

    # 会话中已持久化的消息（此时只有用户消息）
    persisted = session_repo.list_messages(session_id)
    print(f"\nSession 中已持久化消息 {len(persisted)} 条:")
    for m in persisted:
        print(f"  [{m['role']}] {str(m.get('content'))[:40]}")

    # ------------------------------------------------------------------
    print("\n" + "=" * 64)
    print("第二步：模拟进程重启（新建运行时，加载 Checkpoint 继续）")
    print("=" * 64)
    # 全新的运行时实例 —— 模拟重启后的进程，唯一共享的是 SQLite 数据库
    new_session_repo = SQLiteSessionRepository(settings.database_url)
    new_checkpoint_repo = SQLiteCheckpointRepository(settings.database_url)
    restarted = AgentRuntime(
        settings=settings, session_repo=new_session_repo, checkpoint_repo=new_checkpoint_repo
    )

    result = await restarted.resume(session_id)

    latest = new_checkpoint_repo.load_latest(session_id)
    print(f"\n--- 恢复后 ---")
    print(f"最新 checkpoint_id : {latest.checkpoint_id}")
    print(f"最新 version       : {latest.version}")
    print(f"恢复后 state.status: {latest.state['status']}")
    print(f"Final Answer       : {result.answer}")
    print(f"最终消息序列（用户输入只出现 1 次，证明未从头重跑）:")
    for m in result.messages:
        role = m["role"]
        content = m.get("content")
        preview = content[:50] + "..." if isinstance(content, str) and len(content) > 50 else content
        if role == "assistant" and content is None:
            preview = "(tool_calls 消息)"
        print(f"  [{role}] {preview}")

    # 展示完整版本演进
    versions = new_checkpoint_repo.versions(session_id)
    print(f"\nCheckpoint 版本演进（追加写，旧版本保留）: {versions}")

    session_repo.close()
    checkpoint_repo.close()
    new_session_repo.close()
    new_checkpoint_repo.close()


if __name__ == "__main__":
    asyncio.run(main())
