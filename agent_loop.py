import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
import threading
import uuid
from openai import OpenAI
from dotenv import load_dotenv
import yaml
import time

load_dotenv()


def Loger(msg: str):
    print(msg)


api_key = os.environ["API_KEY"]
base_url = os.environ["BASE_URL"]
model_id = os.environ["MODEL"]

client = OpenAI(base_url=base_url, api_key=api_key)
WORKDIR = Path.cwd()
# SYSTEM = f"You are a coding agent at {WORKDIR}. Use the task tool to delegate exploration or subtasks."
SKILLS_DIR = WORKDIR / "skills"

THRESHOLD = 50000
KEEP_RECENT = 3
TRANSCRIPT_DIR = WORKDIR / ".transcripts"
PRESERVE_RESULT_TOOLS = {"read_file"}
TASKS_DIR = WORKDIR / ".tasks"
TEAM_DIR = WORKDIR / ".teams"
INBOX_DIR = TEAM_DIR / "inbox"

POLL_INTERVAL = 5
IDLE_TIMEOUT = 60
SYSTEM = f"You are a team lead at {WORKDIR}. Manage teammates with shutdown and plan approval protocols."

VALID_MSG_TYPES = {
    "message",
    "broadcast",
    "shutdown_request",
    "shutdown_response",
    "plan_approval_response",
}

shutdown_requests = {}
plan_requests = {}
_tracker_lock = threading.Lock()
_claim_lock = threading.Lock()


def scan_unclaimed_tasks() -> list:
    TASKS_DIR.mkdir(exist_ok=True)
    unclaimed = []
    for f in sorted(TASKS_DIR.glob("task_*.json")):
        task = json.loads(f.read_text())
        if (
            task.get("status") == "pending"
            and not task.get("owner")
            and not task.get("blockedBy")
        ):
            unclaimed.append(task)
    return unclaimed


def claim_task(task_id: int, owner: str) -> str:
    with _claim_lock:
        path = TASKS_DIR / f"task_{task_id}.json"
        if not path.exists():
            return f"Error: Task {task_id} not found"
        task = json.loads(path.read_text())
        if task.get("owner"):
            existing_owner = task.get("owner") or "someone else"
            return f"Error: Task {task_id} has already been claimed by {existing_owner}"
        if task.get("status") != "pending":
            status = task.get("status")
            return f"Error: Task {task_id} cannot be claimed because its status is '{status}'"
        if task.get("blockedBy"):
            return f"Error: Task {task_id} is blocked by other task(s) and cannot be claimed yet"
        task["owner"] = owner
        task["status"] = "in_progress"
        path.write_text(json.dumps(task, indent=2))
    return f"Claimed task #{task_id} for {owner}"


# -- Identity re-injection after compression --
def make_identity_block(name: str, role: str, team_name: str) -> dict:
    """
    上下文压缩后，重新身份注入
    """
    return {
        "role": "user",
        "content": f"<identity>You are '{name}', role: {role}, team: {team_name}. Continue your work.</identity>",
    }


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
        with open(inbox_path, "a") as f:
            f.write(json.dumps(msg) + "\n")

        # response
        print(f"{sender} -> {to}: {content} ({msg_type}")
        return f"Sent {msg_type} to {to}"

    def read_inbox(self, name: str) -> list:
        inbox_path = self.dir / f"{name}.jsonl"
        if not inbox_path.exists():
            return []
        messages = []
        for line in inbox_path.read_text().strip().splitlines():
            if line:
                messages.append(json.loads(line))
        inbox_path.write_text("")
        print(f"{name} is reading inbox,there are {len(messages)} messages")
        return messages

    def broadcast(self, sender: str, content: str, teammates: list) -> str:
        count = 0
        for name in teammates:
            if name != sender:
                self.send(sender, name, content, "broadcast")
                count += 1
        # response
        print(f"{sender} broadcasted to {count} teammates")
        return f"Broadcast to {count} teammates"


BUS = MessageBus(INBOX_DIR)


# =====================================================================================
class TeammateManager:
    def __init__(self, team_dir: Path):
        self.dir = team_dir
        self.dir.mkdir(exist_ok=True)
        self.config_path = self.dir / "config.json"
        self.config = self._load_config()
        self.threads = {}

    def _load_config(self) -> dict:
        if self.config_path.exists():
            return json.loads(self.config_path.read_text())
        return {"team_name": "default", "members": []}

    def _save_config(self):
        self.config_path.write_text(json.dumps(self.config, indent=2))

    def _find_member(self, name: str) -> dict:
        for m in self.config["members"]:
            if m["name"] == name:
                return m
        return None

    def _set_status(self, name: str, status: str):
        member = self._find_member(name=name)
        if member:
            member["status"] = status
            self._save_config()

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
        thread = threading.Thread(
            target=self._teammate_loop,
            args=(name, role, prompt),
            daemon=True,
        )
        self.threads[name] = thread
        thread.start()
        return f"Spawned '{name}' (role: {role})"

    def _teammate_loop(self, name: str, role: str, prompt: str):
        team_name = self.config["team_name"]
        sys_prompt = (
            f"You are '{name}', role: {role}, at {WORKDIR}. "
            f"Submit plans via plan_approval before major work. "
            f"Respond to shutdown_request with shutdown_reponse"
        )

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ]
        tools = self._teammate_tools()

        for _ in range(50):
            # 读取消息
            inbox = BUS.read_inbox(name)
            for msg in inbox:
                if msg.get("type") == "shutdown_request":
                    self._set_status(name=name, status="shutdown")
                    return
                messages.append({"role": "user", "content": json.dumps(msg)})
            try:
                # ✅ OpenAI 标准格式调用
                response = client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "system", "content": sys_prompt}, *messages],
                    tools=tools,
                    max_tokens=8000,
                )
            except Exception:
                break

            # 获取助手回复
            response_msg = response.choices[0].message
            messages.append(response_msg)

            # 判断是否停止（不是工具调用就退出）
            if response_msg.tool_calls is None:
                break
            results = []
            idle_requested = False
            # ✅ OpenAI 格式：tool_calls 遍历
            if response_msg.tool_calls:
                for tool_call in response_msg.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    if tool_name == "idle":
                        idle_requested = True
                        output = "Entering idle phase.Will poll for new task"
                    else:
                        # 执行工具
                        output = self._exec(name, tool_name, tool_args)
                    print(f"  [{name}] {tool_name}: {str(output)[:120]}")
                    # 组装 tool_result（OpenAI 格式）
                    results.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": str(output),
                        }
                    )
            # 把工具结果加入对话
            messages.extend(results)
            if idle_requested:
                break

            # idle阶段
        self._set_status(name, "idle")
        resume = False
        polls = IDLE_TIMEOUT // max(POLL_INTERVAL, 1)
        for _ in range(polls):
            time.sleep(POLL_INTERVAL)
            inbox = BUS.read_inbox(name=name)
            if inbox:
                for msg in inbox:
                    if msg.get("type") == "shutdown_request":
                        self._set_status(name, "shutdown")
                        return
                    messages.append({"role": "user", "content": json.dumps(msg)})
                resume = True
            # 检测空闲任务
            unclaimed = scan_unclaimed_tasks()
            if unclaimed:
                task = unclaimed[0]
                # 尝试领取任务
                result = claim_task(task["id"], name)
                if result.startswith("Error:"):
                    continue
                task_prompt = (
                    f"<auto-claimed>Task #{task['id']}: {task['subject']}\n"
                    f"{task.get('description', '')}</auto-claimed>"
                )

                if len(messages) <= 3:
                    messages.insert(0, make_identity_block(name, role, team_name))
                    messages.insert(
                        1, {"role": "assistant", "content": f"I am {name}. Continuing."}
                    )
                messages.append({"role": "user", "content": task_prompt})
                messages.append(
                    {
                        "role": "assistant",
                        "content": f"Claimed task #{task['id']}. Working on it.",
                    }
                )
                resume = True
                break
        if not resume:
            self._set_status(name, "shutdown")
            return
        self._set_status(name, "working")

    def _exec(self, sender: str, tool_name: str, args: dict) -> str:
        # these base tools are unchanged from s02
        if tool_name == "bash":
            return _run_bash(args["command"])
        if tool_name == "read_file":
            return _run_read(args["path"])
        if tool_name == "write_file":
            return _run_write(args["path"], args["content"])
        if tool_name == "edit_file":
            return _run_edit(args["path"], args["old_text"], args["new_text"])
        if tool_name == "send_message":
            return BUS.send(
                sender, args["to"], args["content"], args.get("msg_type", "message")
            )
        if tool_name == "read_inbox":
            return json.dumps(BUS.read_inbox(sender), indent=2)
        if tool_name == "shutdown_response":
            req_id = args["request_id"]
            approve = args["approve"]
            with _tracker_lock:
                if req_id in shutdown_requests:
                    shutdown_requests[req_id]["approve"] = (
                        "approved" if approve else "rejected"
                    )
            BUS.send(
                sender=sender,
                to="lead",
                content=args.get("reason", ""),
                msg_type="shutdown_response",
                extra={"request_id": req_id, "approve": approve},
            )
            return f"Shutdown {'approved' if approve else 'rejected'}"
        if tool_name == "plan_approval":
            plan_text = args.get("plan", "")
            req_id = str(uuid.uuid4())[:8]
            with _tracker_lock:
                plan_requests[req_id] = {
                    "from": sender,
                    "plan": plan_text,
                    "status": "pending",
                }
            BUS.send(
                sender=sender,
                to="lead",
                content=plan_text,
                msg_type="plan_approval_response",
                extra={"request_id": req_id, "plan": plan_text},
            )
            return f"Plan submitted (request_id={req_id}). Waiting for lead approval."
        if tool_name == "claim_task":
            return claim_task(args["task_id"], sender)
        return f"Unknown tool: {tool_name}"

    def _teammate_tools(self) -> list:
        return [
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "description": "Run a shell command.",
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
                    "description": "Read file contents.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to file.",
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
                    "description": "Replace exact text in file.",
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
                    "description": "Send message to a teammate.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "to": {"type": "string"},
                            "content": {"type": "string"},
                            "msg_type": {"type": "string", "enum": VALID_MSG_TYPES},
                        },
                        "required": ["to", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_inbox",
                    "description": "Read and drain your inbox.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "shutdown_response",
                    "description": "Respond to a shutdown request. Approve to shut down, reject to keep working.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "request_id": {
                                "type": "string",
                                "description": "The request ID of the shutdown request.",
                            },
                            "approve": {"type": "boolean"},
                            "reason": {"type": "string"},
                        },
                        "required": ["request_id", "approve"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "plan_approval",
                    "description": "Submit a plan for lead approval. Provide plan text.",
                    "parameters": {
                        "type": "object",
                        "properties": {"plan": {"type": "string"}},
                        "required": ["plan"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "idle",
                    "description": "Signal that you have no more work. Enters idle polling phase.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "claim_task",
                    "description": "Claim a task from the task board by ID",
                    "parameters": {
                        "type": "object",
                        "properties": {"task_id": {"type": "integer"}},
                        "required": ["task_id"],
                    },
                },
            },
        ]

    def list_all(self) -> str:
        if not self.config["members"]:
            return "No teammates."
        lines = [f"Team: {self.config['team_name']}"]
        for m in self.config["members"]:
            lines.append(f"  {m['name']} ({m['role']}): {m['status']}")
        return "\n".join(lines)

    def member_names(self) -> list:
        return [m["name"] for m in self.config["members"]]


TEAM = TeammateManager(TEAM_DIR)


# ===============================================================================
class TaskManager:
    def __init__(self, tasks_dir: Path):
        self.dir = tasks_dir
        self.dir.mkdir(exist_ok=True)
        self._next_id = self._max_id() + 1  # 启动时自动计算下一个可用的自增 ID

    def _max_id(self) -> int:
        """扫描本地硬盘，找出当前最大的任务 ID"""
        ids = [int(f.stem.split("_")[1]) for f in self.dir.glob("task_*.json")]
        return max(ids) if ids else 0

    def _load(self, task_id: int) -> dict:
        """根据 ID 物理读取 JSON 状态机"""
        path = self.dir / f"task_{task_id}.json"
        if not path.exists():
            raise ValueError(f"Task {task_id} not found")
        return json.loads(path.read_text())

    def _save(self, task: dict):
        """原子级落地：将任务状态写回硬盘。这里保留了 ensure_ascii=False 防乱码"""
        path = self.dir / f"task_{task['id']}.json"
        path.write_text(json.dumps(task, indent=2, ensure_ascii=False))

    def create(self, subject: str, description: str = "") -> str:
        """创建一个新任务，初始状态为 pending (待办)"""
        task = {
            "id": self._next_id,
            "subject": subject,
            "description": description,
            "status": "pending",
            "blockedBy": [],
            "owner": "",
        }
        self._save(task)
        self._next_id += 1
        return json.dumps(task, indent=2, ensure_ascii=False)

    def get(self, task_id: int) -> str:
        return json.dumps(self._load(task_id), indent=2, ensure_ascii=False)

    def update(
        self,
        task_id: int,
        status: str = None,
        add_blocked_by: list = None,
        remove_blocked_by: list = None,
    ) -> str:
        """更新任务的核心逻辑"""
        task = self._load(task_id)

        if status:
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Invalid status: {status}")
            task["status"] = status

            # 🎯 图联动的魔法：如果大模型把一个任务标为“已完成”，立刻触发全盘的依赖解除！
            if status == "completed":
                self._clear_dependency(task_id)

        if add_blocked_by:
            # 使用 set 去重，防止同一个前置任务被重复添加
            task["blockedBy"] = list(set(task["blockedBy"] + add_blocked_by))
        if remove_blocked_by:
            task["blockedBy"] = [
                x for x in task["blockedBy"] if x not in remove_blocked_by
            ]

        self._save(task)
        return json.dumps(task, indent=2, ensure_ascii=False)

    def _clear_dependency(self, completed_id: int):
        """
        依赖解绑器 (Dependency resolution)：
        遍历硬盘里的每一个任务，如果它们正在等这个刚完成的 completed_id，就把它从阻塞名单里踢掉。
        """
        for f in self.dir.glob("task_*.json"):
            task = json.loads(f.read_text())
            if completed_id in task.get("blockedBy", []):
                task["blockedBy"].remove(completed_id)
                self._save(task)

    def list_all(self) -> str:
        """
        生成给大模型看的宏观看板（把底层 JSON 渲染成 Markdown 风格的列表）。
        """
        tasks = []
        files = sorted(
            self.dir.glob("task_*.json"), key=lambda f: int(f.stem.split("_")[1])
        )
        for f in files:
            tasks.append(json.loads(f.read_text()))
        if not tasks:
            return "No tasks."

        lines = []
        for t in tasks:
            # 视觉化标识，方便大模型（和人类）一眼看清状态
            marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}.get(
                t["status"], "[?]"
            )
            blocked = f" (blocked by: {t['blockedBy']})" if t.get("blockedBy") else ""
            lines.append(f"{marker} #{t['id']}: {t['subject']}{blocked}")
        return "\n".join(lines)


# 实例化全局任务管理器
TASKS = TaskManager(TASKS_DIR)


# ================================================================================


def _safe_path(p: str):
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"path escapes workspace:{p}")
    return path


def _run_read(path: str, limit: int = None):
    text = _safe_path(path).read_text()
    lines = text.splitlines()
    if limit and limit < len(lines):
        lines = lines[:limit]
    return "\n".join(lines)[:50000]


def _run_write(path: str, content: str) -> str:
    try:
        fp = _safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def _run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = _safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


def _run_bash(command: str):
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(item in command for item in dangerous):
        return "Error:Dangerous command blocks"

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return "Error: Time out (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"

    output = (result.stdout + result.stderr).strip()
    return output[:50000] if output else "(no output)"


TOOL_HANDLERS = {
    "bash": lambda **kw: _run_bash(kw["command"]),
    "read_file": lambda **kw: _run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: _run_write(kw["path"], kw["content"]),
    "edit_file": lambda **kw: _run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "spawn_teammate": lambda **kw: TEAM.spawn(kw["name"], kw["role"], kw["prompt"]),
    "list_teammates": lambda **kw: TEAM.list_all(),
    "send_message": lambda **kw: BUS.send(
        "lead", kw["to"], kw["content"], kw.get("msg_type", "message")
    ),
    "read_inbox": lambda **kw: json.dumps(BUS.read_inbox("lead"), indent=2),
    "broadcast": lambda **kw: BUS.broadcast("lead", kw["content"], TEAM.member_names()),
    "shutdown_request": lambda **kw: handle_shutdown_request(kw["teammate"]),
    "shutdown_response": lambda **kw: _check_shutdown_status(kw.get("request_id", "")),
    "plan_approval": lambda **kw: handle_plan_review(
        kw["request_id"], kw["approve"], kw.get("feedback", "")
    ),
    "task_create": lambda **kw: TASKS.create(kw["subject"], kw.get("description", "")),
    "task_update": lambda **kw: TASKS.update(
        kw["task_id"],
        kw.get("status"),
        kw.get("addBlockedBy"),
        kw.get("removeBlockedBy"),
    ),
    "task_list": lambda **kw: TASKS.list_all(),
    "task_get": lambda **kw: TASKS.get(kw["task_id"]),
    "idle": lambda **kw: "Lead does not idle.",
    "claim_task": lambda **kw: claim_task(kw["task_id"], "lead"),
}
# +++++++++++++++++++++++++++++++工具定义+++++++++++++++++++++++++++++++++++++++++++++++++++++


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "task_create",
            "description": "创建一个新任务，初始状态为 pending（待办）",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": {
                        "type": "string",
                        "description": "任务标题/主题（必填）",
                    },
                    "description": {
                        "type": "string",
                        "description": "任务详细描述（可选）",
                    },
                },
                "required": ["subject"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task_update",
            "description": "更新任务状态、添加/移除依赖阻塞关系；状态完成时会自动解除依赖",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "integer",
                        "description": "要更新的任务 ID（必填）",
                    },
                    "status": {
                        "type": "string",
                        "description": "任务状态，只能是 pending / in_progress / completed",
                        "enum": ["pending", "in_progress", "completed"],
                    },
                    "addBlockedBy": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "要添加的依赖任务 ID 列表",
                    },
                    "removeBlockedBy": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "要移除的依赖任务 ID 列表",
                    },
                },
                "required": ["task_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task_get",
            "description": "根据任务 ID 获取单个任务的完整信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "integer",
                        "description": "要查询的任务 ID（必填）",
                    }
                },
                "required": ["task_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task_list",
            "description": "列出所有任务，包含状态、ID、标题、依赖关系",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "shutdown_request",
            "description": "Request a teammate to shut down gracefully. Returns a request_id for tracking.",
            "parameters": {
                "type": "object",
                "properties": {
                    "teammate": {
                        "type": "string",
                        "description": "Name or identifier of the teammate to shut down",
                    }
                },
                "required": ["teammate"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "shutdown_response",
            "description": "Check the status of a shutdown request by request_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "request_id": {
                        "type": "string",
                        "description": "ID of the shutdown request to check status for",
                    }
                },
                "required": ["request_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plan_approval",
            "description": "Approve or reject a teammate's plan. Provide request_id + approve + optional feedback.",
            "parameters": {
                "type": "object",
                "properties": {
                    "request_id": {
                        "type": "string",
                        "description": "ID of the plan request to approve or reject",
                    },
                    "approve": {
                        "type": "boolean",
                        "description": "Whether to approve the plan (true) or reject it (false)",
                    },
                    "feedback": {
                        "type": "string",
                        "description": "Optional feedback explaining the approval decision",
                    },
                },
                "required": ["request_id", "approve"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "shutdown_request",
            "description": "Request a teammate to shut down gracefully. Returns a request_id for tracking.",
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
            "description": "Check the status of a shutdown request by request_id.",
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
            "description": "Approve or reject a teammate's plan. Provide request_id + approve + optional feedback.",
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
            "description": "Spawn a persistent teammate that runs in its own thread.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name of the teammate"},
                    "role": {"type": "string", "description": "Role of the teammate"},
                    "prompt": {
                        "type": "string",
                        "description": "Initial prompt for the teammate",
                    },
                },
                "required": ["name", "role", "prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_teammates",
            "description": "List all teammates with name, role, status.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_message",
            "description": "Send a message to a teammate's inbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Recipient teammate name"},
                    "content": {"type": "string", "description": "Message content"},
                    "msg_type": {
                        "type": "string",
                        "enum": [
                            "message",
                            "broadcast",
                            "shutdown_request",
                            "shutdown_response",
                            "plan_approval_response",
                        ],
                        "description": "Type of the message",
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
            "description": "Read and drain the lead's inbox.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "broadcast",
            "description": "Send a message to all teammates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Broadcast message content",
                    }
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command in current workspace",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute",
                    }
                },
                "required": ["command"],  # 正确位置
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file in current workspace",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"},
                    "limit": {
                        "type": "integer",
                        "description": "Limit the number of lines to read",
                    },
                },
                "required": ["path", "limit"],  # ✅ 修复：和 properties 平级
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a file in current workspace",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to write"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Edit a file in current workspace",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to edit"},
                    "old_text": {"type": "string", "description": "Text to replace"},
                    "new_text": {
                        "type": "string",
                        "description": "Text to replace with",
                    },
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "todo",
            "description": "Update task list. Track progress on multi-step tasks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "description": "待办事项数组",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {
                                    "type": "string",
                                    "description": "待办事项唯一ID",
                                },
                                "text": {
                                    "type": "string",
                                    "description": "待办内容文本",
                                },
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed"],
                                    "description": "待办状态：待处理/进行中/已完成",
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
    {
        "type": "function",
        "function": {
            "name": "idle",
            "description": "Signal that you have no more work. Enters idle polling phase.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "claim_task",
            "description": "Claim a task from the task board by ID",
            "parameters": {
                "type": "object",
                "properties": {"task_id": {"type": "integer"}},
                "required": ["task_id"],
            },
        },
    },
]


def handle_shutdown_request(teammate: str) -> str:
    req_id = str(uuid.uuid4())[:8]
    with _tracker_lock:
        shutdown_requests[req_id] = {"target": teammate, "status": "pending"}
    BUS.send(
        sender="lead",
        to=teammate,
        content="Please shut down gracefully.",
        msg_type="shutdown_request",
        extra={"request_id": req_id},
    )
    return f"Shutdown request {req_id} sent to '{teammate}' (status: pending)"


def handle_plan_review(request_id: str, approve: bool, feedback: str = ""):
    with _tracker_lock:
        req = plan_requests.get(request_id)
    if not req:
        return f"Error: Unknown plan request_id '{request_id}"
    with _tracker_lock:
        req["status"] = "approved" if approve else "rejected"
    BUS.send(
        "lead",
        req["from"],
        feedback,
        "plan_approval_response",
        {"request_id": request_id, "approve": approve, "feedback": feedback},
    )
    return f"Plan {req['status']} for '{req['from']}'"


def _check_shutdown_status(request_id: str) -> str:
    with _tracker_lock:
        return json.dumps(shutdown_requests.get(request_id, {"error": "not found"}))


def agent_loop(messages: list) -> None:
    print()
    while True:
        inbox = BUS.read_inbox("lead")
        if inbox:
            messages.append(
                {
                    "role": "user",
                    "content": f"<inbox>{json.dumps(inbox, indent=2)}</inbox>",
                }
            )
        else:
            print("lead inbox is empty")

        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            tool_choice="auto",
            tools=TOOLS,
            max_tokens=8000,
        )
        messages.append(response.choices[0].message.model_dump())
        print()
        if response.choices[0].message.content.strip():
            print(f"> Assistant: {response.choices[0].message.content}")
        else:
            print("Assistant: <empty>")
        print(f"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        # 3.从返回内容中，寻找是否参数调用
        finish_reason = response.choices[0].finish_reason
        print(f"finish_reason: {finish_reason}")
        if finish_reason != "tool_calls":
            return
        results = []
        tool_calls = response.choices[0].message.tool_calls
        if tool_calls:
            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                handler = TOOL_HANDLERS.get(tool_name)
                print(f"工具调用: {tool_name}")
                try:
                    output = handler(**args) if handler else f"Unknow tool :{tool_name}"
                except Exception as e:
                    output = f"Error: {e}"

                print(output[:200])
                results.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": output,
                    }
                )
        messages.extend(results)


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms11 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        if query.strip() == "/team":
            print(TEAM.list_all())
            continue
        if query.strip() == "/inbox":
            print(json.dumps(BUS.read_inbox("lead"), indent=2))
            continue
        if query.strip() == "/tasks":
            TASKS_DIR.mkdir(exist_ok=True)
            for f in sorted(TASKS_DIR.glob("task_*.json")):
                t = json.loads(f.read_text())
                marker = {
                    "pending": "[ ]",
                    "in_progress": "[>]",
                    "completed": "[x]",
                }.get(t["status"], "[?]")
                owner = f" @{t['owner']}" if t.get("owner") else ""
                print(f"  {marker} #{t['id']}: {t['subject']}{owner}")
            continue
        history.append({"role": "user", "content": query})
        agent_loop(history)
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(block.text)
        print()
