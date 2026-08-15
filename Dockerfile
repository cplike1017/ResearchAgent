# agent-runtime 镜像
# 同一镜像既跑 API（uvicorn），也跑 Worker（python -m app.worker.worker）
FROM python:3.12-slim

WORKDIR /app

# 先装依赖（利用 Docker 层缓存）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY app ./app
COPY evals ./evals
COPY demos ./demos

ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8

# 默认命令：API；Worker 通过 docker-compose 的 command 覆盖
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
