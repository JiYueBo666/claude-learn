# Claude Multi-Agent System

A sophisticated multi-agent AI system built on OpenAI's API that enables collaborative task management, team coordination, and autonomous workflow automation.

## 🚀 Overview

This project provides a powerful framework for managing multiple AI agents that can work together on complex tasks, communicate with each other, and autonomously manage their workflow. The system includes task management, background processing, skill loading, and team coordination capabilities.

## 🏗️ Architecture

### Core Components

- **Agent Loop** (`agent_loop.py`): Main coordination system for AI agents
- **Utility Functions** (`utils.py`): Common utilities and helper functions
- **Data Models** (`models.py`): Core data structures and managers
- **Configuration** (`config.py`): System configuration and constants

### Key Managers

- **TodoManager**: Track and manage task lists
- **TaskManager**: Handle persistent tasks with dependencies
- **TeammateManager**: Manage multiple AI agents/team members
- **BackgroundManager**: Run commands in background threads
- **SkillLoader**: Load and manage specialized skills/knowledge
- **MessageBus**: Handle inter-agent communication

## 📦 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd claude-learn
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables in `.env`:
```bash
API_KEY=your_openai_api_key
BASE_URL=https://api.openai.com/v1
MODEL=gpt-4-turbo-preview
```

## 🛠️ Usage

### Basic Setup

```python
from models import TODO, TASK_MGR, TEAM, BUS

# Initialize the system
todo = TODO
task_manager = TASK_MGR
team_manager = TEAM
message_bus = BUS
```

### Task Management

```python
# Create a new task
task = task_manager.create("Build API endpoints", "Create REST API for user management")

# List all tasks
print(task_manager.list_all())

# Claim and work on a task
task_manager.claim(task_id=1, owner="developer-agent")
```

### Multi-Agent Team

```python
# Spawn a new teammate
team_manager.spawn(
    name="developer",
    role="Backend Developer", 
    prompt="You are a backend developer. Build robust APIs and handle database operations."
)

# Send messages between agents
BUS.send("lead", "developer", "Please build the user authentication system")

# Broadcast messages to all teammates
BUS.broadcast("lead", "Important: System maintenance scheduled", team_manager.member_names())
```

### Background Processing

```python
from models import BG

# Run commands in background
bg_task = BG.run("python main.py", timeout=300)

# Check background task status
status = BG.check(bg_task)
```

## 📁 Project Structure

```
.
├── agent_loop.py           # Main agent coordination system
├── models.py              # Core data models and managers  
├── utils.py               # Utility functions and helpers
├── config.py              # Configuration and constants
├── __init__.py            # Package initialization
├── .env                   # Environment variables
├── pyproject.toml         # Python package configuration
├── .tasks/                # Task storage directory
├── .team/                 # Team configuration
├── .transcripts/          # Conversation transcripts
└── skills/                # Skill library directory
```

## 🔧 Configuration

Key configuration parameters in `config.py`:

- `WORKDIR`: Working directory for the system
- `TASKS_DIR`: Directory for storing task files
- `TEAM_DIR`: Directory for team configuration
- `TRANSCRIPT_DIR`: Directory for conversation transcripts
- `MAX_CICLE`: Maximum cycles for agent interactions
- `IDLE_TIMEOUT`: Timeout for idle agents
- `POLL_INTERVAL`: Polling interval for idle agents

## 💡 Features

### Dynamic Task Assignment
- Automatic task claiming by available agents
- Task dependency management
- Progress tracking and status updates

### Inter-Agent Communication
- Direct messaging between agents
- Broadcast messaging to all teammates
- Inbox management for message handling

### Skill Management
- Loadable skill modules
- Skill description and metadata support
- Dynamic skill activation

### Background Processing
- Asynchronous command execution
- Status monitoring and result retrieval
- Timeout management

### Conversation Management
- Automatic conversation compression
- Transcript storage and retrieval
- Token limit optimization

## 🔄 Workflow

1. **Initialization**: System loads configuration and managers
2. **Task Creation**: Leader creates tasks with descriptions
3. **Agent Assignment**: Available agents automatically claim tasks
4. **Task Execution**: Agents work on assigned tasks using available tools
5. **Communication**: Agents communicate progress and coordinate
6. **Completion**: Tasks are marked as completed or reassigned
7. **Idle Management**: Agents enter idle state when no tasks available

## 📊 API Reference

### Core Functions

- `TODO.update(items)`: Update todo list
- `TASK_MGR.create(subject, description)`: Create new task
- `TASK_MGR.claim(task_id, owner)`: Claim a task
- `TEAM.spawn(name, role, prompt)`: Create new agent
- `BUS.send(sender, to, content)`: Send message
- `BG.run(command, timeout)`: Run background command

### Tool Functions

```python
# File operations
read_file(path)
write_file(path, content)
edit_file(path, old_text, new_text)

# System operations
bash(command)
task(prompt, agent_type)
load_skill(name)
```

## 🔒 Security Features

- Restricted shell command execution
- Path safety validation
- Environment variable protection
- Command timeout enforcement

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

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a pull request

## 📞 Support

For support and questions, please open an issue in the repository or contact the development team.

---

**Built with ❤️ using OpenAI's powerful AI technology**