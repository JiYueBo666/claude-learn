# cc-mini 项目架构与设计思想总结

## 项目概览

`cc-mini` 是一个**类 Claude Code 的 AI 编程助手 CLI 工具**。它通过大模型 API（Anthropic/OpenAI）驱动，让 AI 代理能在终端里读取文件、执行命令、编辑代码、与用户交互。

---

## 一、整体架构：四层分层设计

```
┌─────────────────────────────────────────────┐
│  Layer 4: TUI / CLI 交互层                   │  ← 用户看到的界面
│  (tui/app.py, tui/prompt.py, tui/shell.py)   │
├─────────────────────────────────────────────┤
│  Layer 3: 功能特性层 (features/)              │  ← 扩展能力
│  ├── coordinator/  多 Agent 协调             │
│  ├── memory/       长期记忆 (Dream)          │
│  ├── plan/         计划模式                  │
│  ├── compact/      上下文压缩                │
│  ├── sandbox/      命令沙箱                  │
│  ├── skills/       可插拔技能                │
│  └── todo/         任务管理                  │
├─────────────────────────────────────────────┤
│  Layer 2: 工具层 (tools/)                     │  ← AI 能调用的能力
│  (Read, Edit, Write, Bash, Glob, Grep, ...)  │
├─────────────────────────────────────────────┤
│  Layer 1: 核心引擎层 (core/)                  │  ← 心脏
│  (engine.py, llm.py, permissions.py, ...)    │
└─────────────────────────────────────────────┘
```

### 为什么这么分层？

1. **核心引擎层（Layer 1）**只关心"如何与 LLM 对话"，完全不关心文件系统、UI、记忆等。
2. **工具层（Layer 2）**被注册到引擎中，引擎通过 `Tool` 抽象基类调用它们，实现解耦。
3. **特性层（Layer 3）**在引擎外层包裹功能——记忆、协调、计划等，**不侵入引擎内部**。
4. **TUI 层（Layer 4）**负责渲染和交互，消费引擎产生的事件流。

---

## 二、核心引擎（core/engine.py）—— 心脏

这是整个项目最关键的文件，它的核心职责是：

> **维护对话状态，处理 LLM 流式响应，调度工具调用，管理重试和错误。**

### 2.1 事件流驱动模型

`Engine.submit()` 不直接返回最终结果，而是返回一个 **事件生成器（Iterator）**：

```python
yield ("text", "Hello")              # 流式文本片段
yield ("tool_call", name, input, activity)   # AI 想调用工具
yield ("tool_executing", ...)        # 获得权限，开始执行
yield ("tool_result", name, input, result)   # 工具执行结果
yield ("waiting",)                   # 文本结束，等待工具调用
yield ("error", "API error...")      # 非致命错误
```

**设计思想**：TUI 层可以**实时消费**这些事件，做到：
- 文本边生成边显示
- 工具调用前弹出权限确认
- 工具执行时显示 spinner
- 出错时即时提示

这比"阻塞等待全部完成再返回"要优雅得多。

### 2.2 工具并行执行

在 `engine.py:394-404`，有一个非常精巧的批处理逻辑：

```python
# 把工具调用分成批次：连续只读工具并行执行，非只读工具串行执行
batches: list[list] = []
for tu in tool_uses:
    is_concurrent = t is not None and t.is_read_only()
    if batches and batches[-1][0] == is_concurrent and is_concurrent:
        batches[-1][1].append(tu)
    else:
        batches.append((is_concurrent, [tu]))
```

**为什么这样设计？**
- `Read`、`Glob`、`Grep` 等只读工具**互不干扰**，可以并行加速
- `Edit`、`Write`、`Bash` 等写操作**有副作用**，必须串行保证顺序和可预测性

### 2.3 重试与容错

引擎内建了完整的指数退避重试机制（`engine.py:22-28`）：
- 自动处理 `RateLimitError`、`APIConnectionError`
- 识别 `Retry-After` 头部
- 上下文溢出时自动折半 `max_tokens` 重试
- 认证失败直接终止（不需要重试）

---

## 三、LLM 抽象层（core/llm.py）—— 屏蔽提供商差异

支持 **Anthropic** 和 **OpenAI** 两家 API，核心设计是**统一数据模型**：

```python
@dataclass
class LLMMessage:
    content: list[dict]        # 标准化内容块
    usage: LLMUsage | None     # 标准化用量
    stop_reason: str | None    # 标准化停止原因
```

内部有两个流式实现：
- `_AnthropicStream` —— 包装 `anthropic.messages.stream()`
- `_OpenAIStream` —— 手动组装 OpenAI 的 SSE chunk 为 `tool_use` 块

**关键设计决策**：所有内部逻辑都用 Anthropic 风格的数据结构（`tool_use`、`tool_result` 等），OpenAI 的响应在进入引擎前就被**规范化**了。这样上层代码不需要关心用的是哪家 API。

---

## 四、权限系统（core/permissions.py）—— 安全闸门

这是 AI 工具最容易被忽视但至关重要的部分。

### 4.1 三级权限模式

| 模式 | 说明 |
|---|---|
| **Default** | 只读工具自动通过，写操作/Bash 需要用户确认 (y/n/always) |
| **Plan Mode** | 只允许 Read/Glob/Grep 和计划文件编辑，防止 AI 在规划时乱改代码 |
| **Dream Mode** | 只允许在 memory 目录内写文件，隔离自动记忆整理过程 |

### 4.2 交互式确认

权限确认实现了**原始终端输入**（`os.read(fd, 1)`），不需要按回车：
- `y` —— 允许一次
- `n` —— 拒绝
- `a` —— 总是允许该工具
- `ESC` —— 取消（与全局 ESC 监听器协作）

**为什么不用 `input()`？** 因为 `input()` 会缓冲整行输入，体验差；而且需要处理 ESC 键的优先级。

---

## 五、Coordinator 模式（features/coordinator/）—— 多 Agent 协作

启用 `--coordinator` 后，主引擎变成**协调器**，可以委派任务给后台 **Worker 引擎**。

- Worker 引擎拥有独立的 LLM 上下文和工具集
- 主引擎通过 `AgentTool`、`SendMessageTool`、`TaskStopTool` 管理 Worker
- Worker 的进度通过**通知队列**异步回报给主引擎

**为什么需要这个？**
- 长时间任务（如"分析整个代码库"）可以放在后台执行，不阻塞主对话
- Worker 有独立的上下文，不会被主对话的历史消息污染

---

## 六、Memory / Dream 系统（features/memory/）—— 持久化记忆

### 6.1 自动记忆（Auto Memory）

每次对话后，如果 AI 的回复中包含 `<memory>...</memory>` 标签，内容会被**自动提取**并追加到每日日志。

### 6.2 Dream 整理（/dream）

当积累足够多的新会话后，系统会自动（或手动 `/dream`）触发：
1. 快照当前对话状态
2. 把记忆日志喂给 LLM，让它总结成主题文件
3. 更新 `MEMORY.md` 索引
4. 恢复原来的对话

**设计意图**：让 AI 在跨会话时记得"你是谁"、"项目背景"、"你的偏好"。

---

## 七、TUI 层（tui/）—— 终端交互体验

主入口 `app.py` 实现了**交互式 REPL**：

### 7.1 双模式输入
- 默认模式：输入发送给 AI
- 终端模式（按 `!` 切换）：输入直接作为 shell 命令执行

### 7.2 Companion 伴侣系统
一个独立的"虚拟宠物"系统，有：
- 随机孵化、稀有度、属性
- 基于对话内容的情绪变化
- 在底部工具栏实时动画
- 可以直接跟它对话

**设计意图**：增加工具的"人格化"和趣味性，让它不只是冰冷的命令行。

---

## 八、关键设计思想总结

| 设计 | 原因 |
|---|---|
| **事件流生成器** | 实时响应，支持流式输出和交互式权限确认 |
| **Tool 抽象基类** | 新工具只需要实现 4 个方法即可接入系统 |
| **LLM 数据规范化** | 上层逻辑不依赖特定提供商，切换模型无感 |
| **权限分层** | Default / Plan / Dream 三层，防止 AI 越权 |
| **只读工具并行** | 加速无风险操作，写操作串行保证安全 |
| **Coordinator 多 Agent** | 长任务后台化，主对话不被阻塞 |
| **Memory 系统** | 跨会话持久化，让 AI 越用越懂你 |
