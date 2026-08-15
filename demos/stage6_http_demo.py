"""
最终验收脚本：通过 Docker 全链路跑一次真实 Agent 请求并展示 Trace 树。

依赖：docker compose up --build -d 已启动（api:8000 / redis / worker）
运行：python -m demos.stage6_http_demo
"""
import json
import time

import httpx

BASE = "http://localhost:8000"

# trust_env=False：绕过环境代理，直连本地 API（部分沙箱环境会拦截 localhost HTTP）
_client = httpx.Client(trust_env=False)


def walk(nodes, prefix=""):
    """递归打印 Trace 树。"""
    for i, n in enumerate(nodes):
        last = i == len(nodes) - 1
        conn = "└── " if last else "├── "
        print(f"{prefix}{conn}{n['name']}  [{n['span_type']}] {n['duration_ms']}ms {n['status']}")
        if n.get("error"):
            print(f"{prefix}{'    ' if last else '│   '}    └── error: {n['error']['type']}: {n['error']['message'][:60]}")
        walk(n["children"], prefix + ("    " if last else "│   "))


def run_one(message: str, idempotency_key: str) -> dict:
    """提交一个请求，轮询完成，返回 job。"""
    r = _client.post(f"{BASE}/api/chat", json={"message": message, "idempotency_key": idempotency_key})
    print(f"POST /api/chat {message!r} -> {r.status_code} {r.json()}")
    body = r.json()
    job_id = body["job_id"]
    job = None
    for _ in range(100):
        job = _client.get(f"{BASE}/api/jobs/{job_id}").json()
        if job["status"] in ("SUCCEEDED", "FAILED"):
            break
        time.sleep(0.3)
    print(f"最终状态: {job['status']}")
    print(f"result  : {json.dumps(job['result'], ensure_ascii=False)}")
    return job


def main() -> None:
    import sys

    message = sys.argv[1] if len(sys.argv) > 1 else "查询北京天气"
    key = sys.argv[2] if len(sys.argv) > 2 else "final-demo-1"

    print("=" * 64)
    print("Docker 全链路演示：HTTP -> Redis -> Worker -> Agent -> Tool")
    print("=" * 64)

    # 1) 提交一次真实请求
    job = run_one(message, key)

    # 2) 展示同一 trace_id 下的完整调用树（跨进程链路）
    trace_id = job["result"]["trace_id"]
    print(f"\ntrace_id: {trace_id}")
    tree = _client.get(f"{BASE}/api/traces/{trace_id}").json()
    print("\nTrace 调用树（gateway -> redis -> worker -> agent -> tool 同一 trace_id）：")
    walk(tree["spans"])

    # 3) 幂等演示：相同 idempotency_key 重复提交 -> 返回同一个 job
    r2 = _client.post(f"{BASE}/api/chat", json={"message": message, "idempotency_key": key})
    print(f"\n幂等验证：重复提交相同 idempotency_key -> {r2.json()['job_id']}（与第一次一致 = {r2.json()['job_id'] == job['job_id']}）")


if __name__ == "__main__":
    main()
