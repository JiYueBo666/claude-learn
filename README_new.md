# Claude Learn

Claude Learn是一个基于OpenAI API的AI智能代理系统，具有工具调用、技能加载和子代理协调等高级功能。该系统设计用于自动化代码开发、文件操作和任务管理。

## 项目概述

Claude Learn实现了一个完整的AI代理循环，使大语言模型能够通过工具与文件系统交互，执行复杂任务。系统核心特性包括：

- **工具调用系统**：提供文件读写、编辑、bash命令执行等基础能力
- **技能加载机制**：按需加载领域专业知识，避免上下文膨胀
- **子代理协调**：将复杂任务分解为独立的子任务，隔离上下文
- **对话压缩**：自动压缩历史对话，保持上下文在合理范围内
- **任务管理**：内置待办事项系统，跟踪多步骤任务进度

## 项目结构

```
.
├── __init__.py          # 包初始化文件
├── agent_loop.py        # 核心代理循环实现
├── utils.py             # 工具函数模块
├── hello.py             # 示例问候模块
├── pyproject.toml       # 项目配置文件
├── requirements.txt     # 依赖列表
└── skills/              # 技能目录
    ├── agent-builder/   # AI代理构建技能
    ├── code-review/     # 代码审查技能
    ├── mcp-builder/     # MCP构建技能
    └── pdf/             # PDF处理技能
```

## 核心功能

### 1. 代理循环 (Agent Loop)

`agent_loop.py`实现了核心的代理执行循环：

- 接收用户输入
- 调用OpenAI API获取响应
- 执行工具调用（文件操作、bash命令等）
- 管理对话历史和上下文
- 自动压缩长对话

### 2. 工具系统

系统提供以下基础工具：

- `bash`: 执行shell命令
- `read_file`: 读取文件内容
- `write_file`: 写入文件
- `edit_file`: 编辑文件内容
- `todo`: 管理待办事项
- `load_skill`: 加载技能知识
- `compact`: 手动触发对话压缩
- `task`: 启动子代理

### 3. 技能系统

技能系统通过`SkillLoader`类实现，支持：

- 从`skills/`目录自动加载技能
- YAML格式的元数据解析
- 按需加载技能内容
- 技能描述注入系统提示

每个技能包含：
- 名称和描述
- 使用场景
- 关键词标签
- 详细知识内容

### 4. 对话管理

系统提供两种对话压缩机制：

1. **微压缩 (micro_compact)**: 精简旧的工具调用结果，保留最近3条完整内容
2. **自动压缩 (auto_compact)**: 当对话超过阈值(50000 tokens)时，使用AI生成摘要并保存完整历史到文件

## 安装

1. 克隆仓库：
```bash
git clone <repository-url>
cd claude-learn
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 配置环境变量：
创建`.env`文件并添加：
```
API_KEY=your_openai_api_key
BASE_URL=https://api.openai.com/v1
MODEL=gpt-4
```

## 使用方法

### 基本使用

直接运行agent_loop：
```bash
python agent_loop.py
```

在交互式提示符中输入任务，例如：
```
s02 >> 创建一个Python脚本，实现斐波那契数列
```

### 作为模块使用

```python
from agent_loop import agent_loop

# 初始化消息历史
messages = [{"role": "user", "content": "你的任务"}]

# 运行代理循环
agent_loop(messages)
```

## 技能使用

系统内置以下技能：

1. **agent-builder**: 设计和构建AI代理
2. **code-review**: 代码审查
3. **mcp-builder**: MCP构建
4. **pdf**: PDF处理

代理会根据任务自动加载相关技能，也可以通过`load_skill`工具手动加载。

## 开发

### 添加新技能

在`skills/`目录下创建新目录，并添加`SKILL.md`文件：

```markdown
---
name: your-skill
description: 技能描述
tags: tag1, tag2
---

# 技能标题

技能详细内容...
```

### 扩展工具

在`agent_loop.py`中添加新的工具处理函数，并更新`TOOL_HANDERS`和工具定义。

## 注意事项

- 系统会自动阻止危险命令（如`rm -rf /`、`sudo`等）
- 所有文件操作限制在工作目录内
- 对话历史会自动保存到`.transcripts/`目录
- 子代理共享文件系统但不共享对话历史

## 许可证

[添加许可证信息]
