# Claude Multi-Agent System

基于OpenAI API构建的智能多代理系统，支持中文交流，提供任务管理、团队协作和自动化工作流功能。

## 🚀 概述

本项目提供一个强大的多代理AI框架，支持多个AI代理协同工作于复杂任务，相互通信并自主管理工作流程。系统包含任务管理、后台处理、技能加载和团队协调等核心功能。

## 🏗️ 架构

### 核心组件

- **代理循环系统** (`agent_loop.py`, `agent_loop2.py`): AI代理的主要协调系统
- **实用工具** (`utils.py`): 常用工具和辅助函数，包含shell命令执行、文件操作等
- **数据模型** (`models.py`): 核心数据结构和管理器（TodoManager、TaskManager、TeammateManager等）
- **配置管理** (`config.py`): 系统配置和常量定义

### 核心管理器

- **TodoManager**: 跟踪和管理待办事项列表（支持pending、in_progress、completed状态）
- **TaskManager**: 处理持久化任务，支持任务依赖关系
- **TeammateManager**: 管理多个AI代理/团队成员，支持自动领取任务
- **BackgroundManager**: 在后台线程中运行命令，支持超时控制
- **SkillLoader**: 加载和管理专业技能知识库
- **MessageBus**: 处理代理间通信，支持信箱机制

## 📦 安装指南

1. 克隆仓库:
```bash
git clone <repository-url>
cd claude-multi-agent
```

2. 安装依赖:
```bash
pip install -r requirements.txt
# 或使用pipx安装：
pipx install .
```

3. 配置环境变量 `.env`:
```bash
API_KEY=your_openai_api_key
BASE_URL=https://api.openai.com/v1
MODEL=gpt-4-turbo-preview
```

## 🛠️ 使用示例

### 基础设置

```python
from models import TODO, TASK_MGR, TEAM, BUS, BG, SKILLS

# 系统已自动初始化，可直接使用
print(TODO.render())  # 查看待办事项
print(TASK_MGR.list_all())  # 列出所有任务
print(TEAM.list_all())  # 列出团队成员
```

### 任务管理

```python
# 创建新任务
task = TASK_MGR.create("构建API端点", "创建用户管理的REST API")

# 列出所有任务
print(TASK_MGR.list_all())

# 领取任务并开始工作
TASK_MGR.claim(task_id=1, owner="开发代理")

# 更新任务状态
TASK_MGR.update(task_id=1, status="completed")
```

### 多代理团队

```python
# 创建新团队成员
TEAM.spawn(
    name="developer",
    role="后端开发工程师", 
    prompt="你是一名后端开发工程师。负责构建稳健的API和处理数据库操作。"
)

# 发送消息给其他代理
BUS.send("lead", "developer", "请构建用户认证系统")

# 广播消息给所有团队成员
BUS.broadcast("lead", "重要通知：系统维护计划", TEAM.member_names())

# 读取收件箱
messages = BUS.read_inbox("developer")
```

### 后台处理

```python
# 运行后台命令
bg_task = BG.run("python main.py", timeout=300)

# 检查后台任务状态
status = BG.check(bg_task)

# 查看所有后台任务
print(BG.check())
```

### 技能加载

```python
# 加载技能
skill_content = SKILLS.load("coding")

# 查看可用技能列表
print(SKILLS.descriptions())
```

### 待办事项管理

```python
# 更新待办事项
TODO.update([
    {"content": "完成API开发", "status": "in_progress", "activeForm": "developer"},
    {"content": "测试系统部署", "status": "pending", "activeForm": "tester"},
    {"content": "编写文档", "status": "completed", "activeForm": "writer"}
])

# 渲染待办列表
print(TODO.render())
```

## 📁 项目结构

```
.
├── agent_loop.py           # 主要的代理协调系统
├── agent_loop2.py          # 备用的代理循环实现
├── models.py              # 核心数据模型和管理器  
├── utils.py               # 实用工具函数和辅助函数
├── config.py              # 配置管理和常量定义
├── __init__.py            # 包初始化文件
├── hello.py               # 示例python文件
├── background_task_file.txt # 后台任务测试文件
├── .env                   # 环境变量配置文件
├── pyproject.toml         # Python包配置
├── requirements.txt       # 依赖包列表
├── .tasks/                # 任务存储目录
├── .team/                 # 团队配置存储
│   └── inbox/            # 代理收件箱
├── .transcripts/          # 对话转录文件
└── skills/                # 技能库目录
```

## 🔧 配置说明

### 主要配置参数 (`config.py`)

```python
# 工作目录和存储路径
WORKDIR = Path.cwd()  # 系统工作目录
TEAM_DIR = WORKDIR / ".team"  # 团队配置目录
INBOX_DIR = TEAM_DIR / "inbox"  # 代理收件箱目录
TASKS_DIR = WORKDIR / ".tasks"  # 任务存储目录
SKILLS_DIR = WORKDIR / "skills"  # 技能库目录
TRANSCRIPT_DIR = WORKDIR / ".transcripts"  # 对话转录目录

# 系统参数
TOKEN_THRESHOLD = 100000  # Token阈值
POLL_INTERVAL = 5  # 空闲代理轮询间隔(秒)
IDLE_TIMEOUT = 60  # 空闲代理超时时间(秒)
MAX_TODO = 20  # 最大待办事项数量
```

### 环境变量 (`.env`)

```bash
# OpenAI API配置
API_KEY=your_openai_api_key
BASE_URL=https://api.openai.com/v1
MODEL=gpt-4-turbo-preview
```

### 消息类型支持

系统支持的消息类型：
- `message`: 普通消息
- `broadcast`: 广播消息
- `shutdown_request`: 关机请求
- `shutdown_response`: 关机响应
- `plan_approval_response`: 计划审批响应

## 💡 核心功能

### 动态任务分配
- 自动任务领取：空闲代理自动检测并领取待处理任务
- 任务依赖管理：支持任务间的阻塞关系
- 进度跟踪：实时状态更新（pending、in_progress、completed）

### 代理间通信
- 直接消息传递：支持代理间一对一的直接通信
- 广播消息：向所有团队成员发送公告消息
- 收件箱管理：每个代理都有独立的收件箱

### 技能管理
- 模块化技能加载：支持按名称动态加载技能
- 技能描述和元数据：通过SKILL.md文件管理技能信息
- 技能目录结构：支持层次化技能组织

### 后台处理
- 异步命令执行：在后台线程中运行耗时操作
- 状态监控：实时监控后台任务执行状态
- 超时控制：防止后台任务无限期运行

### 对话管理
- 自动上下文压缩：智能压缩长对话历史
- 转录存储：保存完整的对话历史记录
- Token优化：自动处理token限制和优化

### 多代理协调
- 团队生命周期管理：自动管理代理的启动、空闲和关闭
- 自动身份维护：在上下文压缩后重新注入代理身份
- 循环工作流：支持代理在工作状态和空闲状态间自动切换

## 🔄 工作流程

1. **系统初始化**: 加载配置和管理器，建立工作环境
2. **任务创建**: 领导者创建带描述的任务
3. **任务分配**: 空闲代理自动检测并领取任务
4. **任务执行**: 代理使用可用工具处理分配的任务
5. **通信协调**: 代理间通过消息系统沟通进展和协调
6. **任务完成**: 任务完成后标记并处理依赖关系
7. **空闲管理**: 无任务时代理进入空闲轮询状态

## 🧪 开发指南

### 运行测试
```bash
python -m unittest discover tests/
```

### 添加新技能
1. 在 `skills/` 目录下创建技能子目录
2. 添加 `SKILL.md` 文件，包含技能描述和元数据
```markdown
---
name: skill-name
description: 技能描述
version: 1.0.0
---
技能内容描述...
```
3. 使用 `load_skill("skill-name")` 加载技能

### 功能扩展

新工具函数可以通过扩展 `agent_loop.py` 中的工具处理程序和 `utils.py` 中的相应函数来添加。

### 调试工具

系统使用 `loguru` 库进行日志记录，所有主要操作都有详细的日志输出便于调试。

### 性能优化

- 使用 `lru_cache` 实现管理器单例模式
- 智能的上下文压缩减少token消耗
- 异步后台处理避免阻塞主线程

## 📊 API参考

### 核心管理器方法

#### TodoManager (TODO)
- `TODO.update(items: list)`: 更新待办事项列表
- `TODO.render() -> str`: 渲染待办事项状态显示
- `TODO.has_open_items() -> bool`: 检查是否有未完成项

#### TaskManager (TASK_MGR)
- `TASK_MGR.create(subject: str, description: str = "") -> str`: 创建新任务
- `TASK_MGR.claim(task_id: int, owner: str) -> str`: 领取任务
- `TASK_MGR.update(task_id: int, status: str = None, add_blocked_by: list = None, remove_blocked_by: list = None) -> str`: 更新任务状态
- `TASK_MGR.get(task_id: int) -> str`: 获取任务详情
- `TASK_MGR.list_all() -> str`: 列出所有任务

#### TeammateManager (TEAM)
- `TEAM.spawn(name: str, role: str, prompt: str) -> str`: 创建团队成员
- `TEAM.list_all() -> str`: 列出所有团队成员
- `TEAM.member_names() -> list`: 获取团队成员名称列表

#### MessageBus (BUS)
- `BUS.send(sender: str, to: str, content: str, msg_type: str = "message", extra: dict = None) -> str`: 发送消息
- `BUS.read_inbox(name: str) -> list`: 读取收件箱
- `BUS.broadcast(sender: str, content: str, names: list) -> str`: 广播消息

#### BackgroundManager (BG)
- `BG.run(command: str, timeout: int = 120) -> str`: 运行后台命令
- `BG.check(task_id: str = None) -> str`: 检查后台任务状态
- `BG.drain() -> list`: 获取后台任务通知

#### SkillLoader (SKILLS)
- `SKILLS.load(name: str) -> str`: 加载技能内容
- `SKILLS.descriptions() -> str`: 获取可用技能描述

### 工具函数 (Tool Functions)

```python
# 文件操作
read_file(path: str, limit: int = None) -> str  # 读取文件内容
write_file(path: str, content: str) -> str      # 写入文件
edit_file(path: str, old_text: str, new_text: str) -> str  # 替换文件内容

# 系统操作
bash(command: str) -> str                        # 执行shell命令

# 任务和技能
task(prompt: str, agent_type: str = "Explore")   # 创建子任务代理
load_skill(name: str) -> str                     # 加载技能
```

### 安全特性 🔒

- **受限的shell命令执行**: 过滤危险命令（如rm -rf /, sudo等）
- **路径安全验证**: 防止路径逃逸
- **环境变量保护**: 安全的API密钥管理
- **命令超时控制**: 防止无限期执行
- **输入验证**: 严格的参数检查和类型验证

## 🧪 Development

### Running Tests
```bash
python -m unittest discover tests/
```

### Adding New Skills
1. Create a skill directory under `skills/`
2. Add a `SKILL.md` file with skill description and metadata
3. Use `load_skill("skill-name")` to load the skill

### Extending Functionality
New tool functions can be added by extending the tool handlers in `agent_loop.py` and the corresponding functions in `utils.py`.

## 📄 许可证

本项目采用 MIT 许可证 - 详见 LICENSE 文件。

## 🤝 贡献指南

1. Fork 本仓库
2. 创建功能分支
3. 提交您的更改
4. 推送分支
5. 创建 Pull Request

## 📞 支持

如需支持或有问题，请在仓库中提交 issue 或联系开发团队。

---

**基于 OpenAI 强大AI技术构建，用心打造**