# ReAgent 演示视频录制指南

> 目标：一条 2 分钟内的屏幕录制，让面试官 30 秒建立"这个项目真实可用"
> 的印象，剩余时间展示 Trace 树的可视化冲击力。

## 工具

| 工具 | 用途 | 说明 |
|---|---|---|
| OBS Studio（免费） | 录屏 | 录 1080p 浏览器窗口即可，无需摄像头 |
| 浏览器开发者工具 | 放大 | Ctrl+Shift+P → 输入 zoom，放大到 120-150% 让文字清晰 |
| ffmpeg（可选） | 压缩 | `ffmpeg -i demo_raw.mp4 -c:v libx264 -crf 23 -preset slow demo.mp4` |
| 剪映 / CapCut（可选） | 剪辑 | 加字幕、剪掉等待时间 |

## 演示脚本（约 100 秒，三段）

### 第 1 段：单 Agent 能力（0:00 - 0:30）

```text
动作：打开 http://localhost:8000，新建会话，选择 Plan 模式
输入：帮我查北京和上海的天气并对比
旁白：ReAct 循环 + Plan-and-Execute，左侧能看到工具列表、会话历史
要点：点击工作流面板里的 llm_call 和 tool.execute 节点，展示展开详情
      （参数、输出、耗时）——证明"全透明"不是口号
```

### 第 2 段：多 Agent 编排（0:30 - 1:30）⭐ 核心段落

```text
动作：新建会话，输入：
  用 delegate 工具，把「调研 2024 大模型进展并给出 RAG 建议，
  先检索后分析最后成稿」交给子 agent 执行
旁白：主 agent 决定调用 delegate 工具 → 编排器自动分工
      （researcher 检索 → analyst 分析 → writer 成稿，带依赖链）
要点：
  1. 等 delegate 工具卡片出现，展开它——展示"✅ 编排状态 + 子 agent chips"
  2. 打开下方 Trace 树：orchestrator.run → agent.run ×3 →
     各自的 llm_call / tool.execute，逐个展开，展示层级
  3. 如果用了多级编排，指出嵌套的 orchestrator.run (depth=2)
  4. 等待最终回答，展示 writer 子 agent 落盘的文件（左侧文件列表出现）
```

### 第 3 段：编排记录回放（1:30 - 1:50）

```text
动作：点击左侧会话列表里的当前会话 → 下方"编排记录"面板出现本次记录
      点击该记录 → 打开编排详情面板（分工计划 + 子 agent 结果卡片 +
      嵌套子编排 + 最终答案 + Trace 树）
旁白：每次 delegate 编排的结果都结构化持久化，可随时回放——
      这不只是聊天记录，而是完整的执行档案
```

## 录制注意事项

1. **提前演练两次**：真实 LLM 单次编排可能 2-4 分钟，旁白语速要配合；
   建议提前跑一遍确认任务能稳定成功（避免现场超时/网络抖动）；
2. **网络不稳时**：可改用 stub 模式演示 UI（`LLM_PROVIDER=stub`），
   但多 Agent 编排需要真实 LLM 才有分工效果；折中方案：演示前先手动
   跑一次编排把记录存好，录制时打开历史记录回放，不依赖现场网络；
3. **字幕**：每段开头一句旁白即可，不逐字配字幕；
4. **发布**：视频放 GitHub README（`![demo](docs/demo.mp4)` 或转 GIF
   用 `docs/demo.gif`），或放 B 站/YouTube 链接；
5. **时长红线**：超过 2 分钟就剪，宁可快剪掉 LLM 等待时间。

## GIF 制作（README 首页用）

```bash
# 从录屏截取 6-8 秒"编排详情面板展开"片段转 GIF（适合首页）
ffmpeg -ss 00:00:40 -t 8 -i demo_raw.mp4 -vf "fps=12,scale=960:-1" docs/demo.gif
```

README 已预留 GIF 占位，生成后替换 `docs/demo-guide.md` 同目录的引用即可。
