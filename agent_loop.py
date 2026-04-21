import json
import os
import subprocess
import threading
import uuid
import time
from dataclasses import dataclass
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# ===================== OpenAI 客户端初始化（标准格式） =====================
api_key = os.environ["API_KEY"]
base_url = os.environ["BASE_URL"]
model_id = os.environ["MODEL"]

# 标准 OpenAI 客户端初始化
client = OpenAI(api_key=api_key, base_url=base_url)

# ===================== 全局常量配置 =====================
WORKDIR = Path.cwd()
SKILLS_DIR = WORKDIR / "skills"
THRESHOLD = 50000
KEEP_RECENT = 3
TRANSCRIPT_DIR = WORKDIR / ".transcripts"
PRESERVE_RESULT_TOOLS = {"read_file"}
TASKS_DIR = WORKDIR / ".tasks"
TEAM_DIR = WORKDIR / ".teams"
INBOX_DIR = TEAM_DIR / "inbox"

# 系统提示词（标准格式）
SYSTEM = f"You are a team lead at {WORKDIR}. Manage teammates with shutdown and plan approval protocols."
VALID_MSG_TYPES = {
    "message",
    "broadcast",
    "shutdown_request",
    "shutdown_response",
    "plan_approval_response",
}

# 全局状态追踪
shutdown_requests = {}
plan_requests = {}
_tracker_lock = threading.Lock()


# ===================== 日志工具 =====================
def Logger(msg: str):
    print(msg)


# ===================== 消息总线（OpenAI 格式兼容） =====================
class MessageBus:
    def __init__(self, inbox_dir: Path):
        self.dir = inbox_dir
        self.dir.mkdir(parents=True, exist_ok=True)

    def send(
        self,
        sender: str,
        to: str,
        content: str,
        msg_type: str = "message",
        extra: dict = None,
    ):
        if msg_type not in VALID_MSG_TYPES:
            return f"Error: Invalid type '{msg_type}'. Valid: {VALID_MSG_TYPES}"
        msg = {
            "type": msg_type,
            "from": sender,
            "content": content,
            "timestamp": time.time(),
        }
        if extra:
            msg.update(extra)
        inbox_path = self.dir / f"{to}.jsonl"
        with open(inbox_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")

        print(f"{sender} -> {to}: {content} ({msg_type})")
        return f"Sent {msg_type} to {to}"

    def read_inbox(self, name: str) -> list:
        inbox_path = self.dir / f"{name}.jsonl"
        if not inbox_path.exists():
            return []
        messages = []
        for line in inbox_path.read_text(encoding="utf-8").strip().splitlines():
            if line:
                messages.append(json.loads(line))
        inbox_path.write_text("", encoding="utf-8")
        print(f"{name} read inbox: {len(messages)} messages")
        return messages

    def broadcast(self, sender: str, content: str, teammates: list) -> str:
        count = 0
        for name in teammates:
            if name != sender:
                self.send(sender, name, content, "broadcast")
                count += 1
        print(f"{sender} broadcasted to {count} teammates")
        return f"Broadcast to {count} teammates"


# 全局消息总线实例
BUS = MessageBus(INBOX_DIR)


# ===================== 队友管理器（OpenAI 标准调用） =====================
class TeammateManager:
    def __init__(self, team_dir: Path):
        self.dir = team_dir
        self.dir.mkdir(exist_ok=True)
        self.config_path = self.dir / "config.json"
        self.config = self._load_config()
        self.threads = {}

    def _load_config(self) -> dict:
        if self.config_path.exists():
            return json.loads(self.config_path.read_text(encoding="utf-8"))
        return {"team_name": "default", "members": []}

    def _save_config(self):
        self.config_path.write_text(
            json.dumps(self.config, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _find_member(self, name: str) -> dict:
        for m in self.config["members"]:
            if m["name"] == name:
                return m
        return None

    def spawn(self, name: str, role: str, prompt: str) -> str:
        member = self._find_member(name)
        if member:
            if member["status"] not in ("idle", "shutdown"):
                return f"Error: '{name}' is currently {member['status']}"
            member["status"] = "working"
            member["role"] = role
        else:
            member = {"name": name, "role": role, "status": "working"}
            self.config["members"].append(member)
        self._save_config()

        # 启动线程（OpenAI 格式队友循环）
        thread = threading.Thread(
            target=self._teammate_loop,
            args=(name, role, prompt),
            daemon=True,
        )
        self.threads[name] = thread
        thread.start()
        return f"Spawned '{name}' (role: {role})"

    def _teammate_loop(self, name: str, role: str, prompt: str):
        """队友核心循环：严格遵循 OpenAI 标准聊天格式"""
        sys_prompt = (
            f"You are '{name}', role: {role}, at {WORKDIR}. "
            f"Submit plans via plan_approval before major work."
        )
        # OpenAI 标准消息列表
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ]
        tools = self._get_teammate_tools()
        shutdown_exit = False

        # 最大循环次数
        for _ in range(50):
            # 读取收件箱消息
            inbox = BUS.read_inbox(name)
            for msg in inbox:
                messages.append(
                    {
                        "role": "user",
                        "content": json.dumps(msg, ensure_ascii=False, indent=2),
                    }
                )

            try:
                # ✅ OpenAI 官方标准 API 调用
                response = client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    tools=tools,
                    max_tokens=8000,
                    tool_choice="auto",
                )
            except Exception as e:
                Logger(f"Teammate {name} error: {str(e)}")
                break

            # 获取 OpenAI 标准响应消息
            assistant_msg = response.choices[0].message
            messages.append(assistant_msg.model_dump())

            # 无工具调用则跳过
            if not assistant_msg.tool_calls:
                continue

            # 执行工具调用（OpenAI 标准格式）
            tool_results = []
            for tool_call in assistant_msg.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                # 执行工具
                output = self._exec_tool(name, tool_name, tool_args)
                Logger(f"[{name}] {tool_name}: {str(output)[:120]}")

                # ✅ OpenAI 标准工具结果格式
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(output),
                    }
                )

                # 关机指令处理
                if tool_name == "shutdown_response" and tool_args.get("approve"):
                    shutdown_exit = True

            # 将工具结果加入对话上下文
            messages.extend(tool_results)

        # 更新队友状态
        member = self._find_member(name)
        if member:
            member["status"] = "shutdown" if shutdown_exit else "idle"
            self._save_config()

    def _exec_tool(self, sender: str, tool_name: str, args: dict) -> str:
        """工具执行逻辑（标准封装）"""
        if tool_name == "bash":
            return self._run_bash(args["command"])
        if tool_name == "read_file":
            return self._run_read(args["path"], args.get("limit"))
        if tool_name == "write_file":
            return self._run_write(args["path"], args["content"])
        if tool_name == "edit_file":
            return self._run_edit(args["path"], args["old_text"], args["new_text"])
        if tool_name == "send_message":
            return BUS.send(
                sender, args["to"], args["content"], args.get("msg_type", "message")
            )
        if tool_name == "read_inbox":
            return json.dumps(BUS.read_inbox(sender), indent=2, ensure_ascii=False)
        return f"Unknown tool: {tool_name}"

    def _get_teammate_tools(self) -> list:
        """✅ OpenAI 标准函数调用工具定义"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "description": "Run a shell command in the workspace",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "Shell command to execute",
                            }
                        },
                        "required": ["command"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read file contents from workspace",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "File path to read",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Max lines to read (optional)",
                            },
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["path", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "edit_file",
                    "description": "Replace exact text in a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "old_text": {"type": "string"},
                            "new_text": {"type": "string"},
                        },
                        "required": ["path", "old_text", "new_text"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "send_message",
                    "description": "Send message to a teammate",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "to": {"type": "string"},
                            "content": {"type": "string"},
                            "msg_type": {
                                "type": "string",
                                "enum": list(VALID_MSG_TYPES),
                            },
                        },
                        "required": ["to", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_inbox",
                    "description": "Read and clear your inbox",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]

    # 安全文件操作工具
    def _safe_path(self, p: str) -> Path:
        path = (WORKDIR / p).resolve()
        if not path.is_relative_to(WORKDIR):
            raise ValueError(f"Path escapes workspace: {p}")
        return path

    def _run_read(self, path: str, limit: int = None) -> str:
        text = self._safe_path(path).read_text(encoding="utf-8")
        lines = text.splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit]
        return "\n".join(lines)[:THRESHOLD]

    def _run_write(self, path: str, content: str) -> str:
        try:
            fp = self._safe_path(path)
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(content, encoding="utf-8")
            return f"Wrote {len(content)} bytes to {path}"
        except Exception as e:
            return f"Error: {str(e)}"

    def _run_edit(self, path: str, old_text: str, new_text: str) -> str:
        try:
            fp = self._safe_path(path)
            content = fp.read_text(encoding="utf-8")
            if old_text not in content:
                return f"Error: Text not found in {path}"
            fp.write_text(content.replace(old_text, new_text, 1), encoding="utf-8")
            return f"Edited {path}"
        except Exception as e:
            return f"Error: {str(e)}"

    def _run_bash(self, command: str) -> str:
        dangerous = ["rm -rf /", "sudo", "shutdown", "reboot"]
        if any(cmd in command for cmd in dangerous):
            return "Error: Dangerous command blocked"

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=WORKDIR,
                capture_output=True,
                text=True,
                timeout=120,
                encoding="utf-8",
            )
        except subprocess.TimeoutExpired:
            return "Error: Command timed out (120s)"
        except Exception as e:
            return f"Error: {str(e)}"

        output = (result.stdout + result.stderr).strip()
        return output[:THRESHOLD] if output else "(no output)"

    def list_all(self) -> str:
        if not self.config["members"]:
            return "No teammates."
        lines = [f"Team: {self.config['team_name']}"]
        for m in self.config["members"]:
            lines.append(f"  {m['name']} ({m['role']}): {m['status']}")
        return "\n".join(lines)

    def member_names(self) -> list:
        return [m["name"] for m in self.config["members"]]


# 全局团队管理器实例
TEAM = TeammateManager(TEAM_DIR)

# ===================== 核心工具处理器（OpenAI 标准） =====================
TOOL_HANDLERS = {
    "bash": lambda **kw: TEAM._run_bash(kw["command"]),
    "read_file": lambda **kw: TEAM._run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: TEAM._run_write(kw["path"], kw["content"]),
    "edit_file": lambda **kw: TEAM._run_edit(
        kw["path"], kw["old_text"], kw["new_text"]
    ),
    "spawn_teammate": lambda **kw: TEAM.spawn(kw["name"], kw["role"], kw["prompt"]),
    "list_teammates": lambda **kw: TEAM.list_all(),
    "send_message": lambda **kw: BUS.send(
        "lead", kw["to"], kw["content"], kw.get("msg_type", "message")
    ),
    "read_inbox": lambda **kw: json.dumps(
        BUS.read_inbox("lead"), indent=2, ensure_ascii=False
    ),
    "broadcast": lambda **kw: BUS.broadcast("lead", kw["content"], TEAM.member_names()),
}

# ===================== ✅ OpenAI 官方标准工具定义（完整） =====================
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "shutdown_request",
            "description": "Request a teammate to shut down gracefully",
            "parameters": {
                "type": "object",
                "properties": {"teammate": {"type": "string"}},
                "required": ["teammate"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "shutdown_response",
            "description": "Check shutdown request status",
            "parameters": {
                "type": "object",
                "properties": {"request_id": {"type": "string"}},
                "required": ["request_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plan_approval",
            "description": "Approve/reject a teammate's plan",
            "parameters": {
                "type": "object",
                "properties": {
                    "request_id": {"type": "string"},
                    "approve": {"type": "boolean"},
                    "feedback": {"type": "string"},
                },
                "required": ["request_id", "approve"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spawn_teammate",
            "description": "Create a new teammate thread",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "role": {"type": "string"},
                    "prompt": {"type": "string"},
                },
                "required": ["name", "role", "prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_teammates",
            "description": "List all team members",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_message",
            "description": "Send message to a teammate",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "content": {"type": "string"},
                    "msg_type": {"type": "string", "enum": list(VALID_MSG_TYPES)},
                },
                "required": ["to", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_inbox",
            "description": "Read lead's inbox",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "broadcast",
            "description": "Send message to all teammates",
            "parameters": {
                "type": "object",
                "properties": {"content": {"type": "string"}},
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute shell command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read file from workspace",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write file to workspace",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Edit existing file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "todo",
            "description": "Update task progress list",
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "text": {"type": "string"},
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed"],
                                },
                            },
                            "required": ["id", "text", "status"],
                        },
                    }
                },
                "required": ["items"],
            },
        },
    },
]


# ===================== 主 Agent 循环（OpenAI 标准格式） =====================
def agent_loop(messages: list) -> None:
    """主控制器循环：严格遵循 OpenAI 标准"""
    while True:
        # 读取收件箱
        inbox = BUS.read_inbox("lead")
        if inbox:
            messages.append(
                {
                    "role": "user",
                    "content": f"<inbox>{json.dumps(inbox, indent=2, ensure_ascii=False)}</inbox>",
                }
            )

        # ✅ OpenAI 官方标准 API 调用
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=8000,
        )

        # 保存响应（OpenAI 标准格式）
        assistant_msg = response.choices[0].message
        messages.append(assistant_msg.model_dump())

        # 打印响应
        print("\n> Assistant:", assistant_msg.content or "<empty>")
        print("-" * 60)

        # 无工具调用则退出循环
        if response.choices[0].finish_reason != "tool_calls":
            break

        # 执行工具调用（OpenAI 标准）
        tool_results = []
        if response.choices[0].message.tool_calls:
            for tool_call in assistant_msg.tool_calls:
                tool_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)

                Logger(f"Executing tool: {tool_name}")
                try:
                    handler = TOOL_HANDLERS.get(tool_name)
                    output = (
                        handler(**args) if handler else f"Unknown tool: {tool_name}"
                    )
                except Exception as e:
                    output = f"Tool error: {str(e)}"

                Logger(output[:200])
                # ✅ OpenAI 标准工具结果
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": str(output),
                    }
                )

            messages.extend(tool_results)


# ===================== 主程序入口 =====================
if __name__ == "__main__":
    # 初始化对话上下文（OpenAI 标准）
    history = [{"role": "system", "content": SYSTEM}]

    Logger("Agent started (OpenAI Standard Format) | Type 'q' to exit")
    while True:
        try:
            query = input("\n\033[36ms02 >> \033[0m").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if query.lower() in ("q", "exit", ""):
            Logger("Exiting agent...")
            break

        # 添加用户消息（OpenAI 标准）
        history.append({"role": "user", "content": query})
        # 执行主循环
        agent_loop(history)
