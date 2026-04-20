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

SYSTEM = f"You are a coding agent at {WORKDIR}. Use background_run for long-running commands."


def estimate_tokens(messages: list):
    return len(str(messages)) // 4


def micro_compact(messages: list):
    """
    OpenAI 格式对话消息精简：
    自动压缩旧的 tool call 结果，只保留最近 KEEP_RECENT 条完整内容
    兼容 OpenAI Chat Completions tools / function calling 格式
    """
    # 1. 收集所有 tool_result（OpenAI  role=tool 的消息）
    tool_results = []
    for msg_idx, msg in enumerate(messages):
        # OpenAI 官方：工具返回用 role="tool"
        if msg.get("role") == "tool":
            tool_results.append((msg_idx, msg))

    # 如果工具结果太少，不精简
    if len(tool_results) < KEEP_RECENT:
        return messages

    # 2. 建立 tool_call_id -> function name 映射（从 assistant 调用记录）
    tool_name_map = {}
    for msg in messages:
        if msg.get("role") == "assistant":
            tool_calls = msg.get("tool_calls", [])
            if not tool_calls:
                continue
            for call in tool_calls:
                tool_id = call.get("id")
                tool_name = call.get("function", {}).get("name")
                if tool_id and tool_name:
                    tool_name_map[tool_id] = tool_name

    # 3. 只保留最近 KEEP_RECENT 条，更早的需要精简
    to_clear = tool_results[:-KEEP_RECENT]

    # 4. 精简旧的工具结果
    for msg_idx, result_msg in to_clear:
        content = result_msg.get("content", "")
        tool_call_id = result_msg.get("tool_call_id", "")

        # 过滤：短内容不精简
        if not isinstance(content, str) or len(content) < 100:
            continue

        # 获取工具名
        tool_name = tool_name_map.get(tool_call_id, "unknown_tool")

        # 白名单工具不精简
        if tool_name in PRESERVE_RESULT_TOOLS:
            continue

        # 替换为精简提示（OpenAI 格式直接改 content 即可）
        result_msg["content"] = f"[Previous result: {tool_name} called]"

    return messages


def auto_compact(messages: list):
    TRANSCRIPT_DIR.mkdir(exist_ok=True)
    transcript_path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    # 历史写入文件
    with open(transcript_path, "w", encoding="utf-8") as f:
        for msg in messages:
            f.write(json.dumps(msg, ensure_ascii=False, default=str) + "\n")
    print(f"[transcript saved: {transcript_path}]")

    conversation_text = json.dumps(messages, ensure_ascii=False, default=str)[-80000:]
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {
                "role": "user",
                "content": "Summarize this conversation for continuity. Include:\n"
                "1) What was accomplished\n"
                "2) Current state\n"
                "3) Key decisions made\n"
                "Be concise but keep critical details.\n\n"
                f"Conversation:\n{conversation_text}",
            }
        ],
        max_tokens=2000,
        temperature=0.1,  # 让总结更稳定
    )
    summary = response.choices[0].message.content.strip()
    if not summary:
        summary = "No summary generated."

    # ========== 5. 返回 OpenAI 格式的单条压缩消息 ==========
    return [
        {
            "role": "user",
            "content": f"[Conversation compressed. Transcript: {transcript_path}]\n\n{summary}",
        }
    ]


# ================================
class SkillLoader:
    def __init__(self, skill_dir: Path):
        self.skill_dir = skill_dir
        self.skills = {}
        self._load_all()

    def _load_all(self):
        if not self.skill_dir.exists():
            return
        for f in sorted(self.skill_dir.rglob("SKILL.md")):
            text = f.read_text()
            meta, body = self._parse_frontmatter(text)
            name = meta.get("name", f.parent.name)
            self.skills[name] = {"meta": meta, "body": body, "path": str(f)}

    def _parse_frontmatter(self, text):
        match = re.match(r"^---\n(.*?)\n---\n(.*)", text, re.DOTALL)

        if not match:
            return {}, text

        try:
            meta = yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError:
            meta = {}

        return meta, match.group(2).strip()

    def get_descriptions(self):
        """
        short description for the system prompt
        """
        if not self.skills:
            return "(no skills available)"
        lines = []
        for name, skill in self.skills.items():
            desc = skill["meta"].get("description", "No description")
            tags = skill["meta"].get("tags", "")
            line = f"  - {name}: {desc}"
            if tags:
                line += f" [{tags}]"
            lines.append(line)
        return "\n".join(lines)

    def get_content(self, name: str) -> str:
        """Layer 2: full skill body returned in tool_result."""
        skill = self.skills.get(name)
        if not skill:
            return f"Error: Unknown skill '{name}'. Available: {', '.join(self.skills.keys())}"
        return f"<skill name=\"{name}\">\n{skill['body']}\n</skill>"


SKILL_LOADER = SkillLoader(SKILLS_DIR)


# =======================================================================================


class TaskManager:
    def __init__(self, task_dir: Path):
        self.dir = task_dir
        self.dir.mkdir(exist_ok=True)
        self._next_id = self._max_id() + 1

    def _max_id(self):
        ids = [int(f.stem.split("_")[1]) for f in self.dir.glob("*task_*.json")]
        return max(ids) if ids else 0

    def _load(self, task_id: int):
        path = self.dir / f"task_{task_id}.json"
        if not path.exists():
            raise ValueError(f"Task {task_id} not found")
        return json.loads(path.read_text())

    def _save(self, task: dict):
        path = self.dir / f"task_{task['id']}.json"
        path.write_text(json.dumps(task, indent=2, ensure_ascii=False))

    def create(self, subject: str, description: str = ""):
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

    def get(self, task_id: int):
        return json.dumps(self._load(task_id), indent=2, ensure_ascii=False)

    def update(
        self,
        task_id: int,
        status: str = None,
        add_blocked_by: list = None,
        remove_blocked_by: list = None,
    ):
        """
        检查状态，完成则清理依赖
        """
        task = self._load(task_id)
        if status:
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Invalid status: {status}")
            task["status"] = status
            if status == "completed":
                self._clear_dependency(task_id)
        if add_blocked_by:
            task["blockedBy"] = list(set(task["blockedBy"] + add_blocked_by))
        if remove_blocked_by:
            task["blockedBy"] = [
                t for t in task["blockedBy"] if t not in remove_blocked_by
            ]
        self._save(task)
        return json.dumps(task, indent=2, ensure_ascii=False)

    def _clear_dependency(self, completed_id: int):
        """
        remove completed id from all other tasks' blockedBy lists
        """
        for f in self.dir.glob("*task_*.json"):
            task = json.loads(f.read_text())
            if completed_id in task["blockedBy"]:
                task["blockedBy"].remove(completed_id)
                self._save(task)

    def list_all(self):
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
            marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}.get(
                t["status"], "[?]"
            )
            blocked = f" (blocked by: {t['blockedBy']})" if t.get("blockedBy") else ""
            lines.append(f"{marker} #{t['id']}: {t['subject']}{blocked}")
        return "\n".join(lines)


TASKS = TaskManager(TASKS_DIR)

# =================================================================================


class BackgroundManager:
    """
    run 创建线程，建立任务id映射。
    exe 执行命令，写入通知队列
    check 检查tasks映射，返回所有后台任务
    """

    def __init__(self):
        self.tasks = {}  # task id->{status,result,command}
        self._notification_queue = []  # 已完成的结果
        self._lock = threading.Lock()

    def run(self, command: str):
        """
        start a background thread,return task_id immediately
        """
        task_id = str(uuid.uuid4())[:8]
        self.tasks[task_id] = {"status": "running", "result": None, "command": command}
        thread = threading.Thread(
            target=self._execute, args=(task_id, command), daemon=True
        )
        thread.start()
        return f"Background task {task_id} started : {command[:80]}"

    def _execute(self, task_id: str, command: str):
        """
        threading target: run subprocess,capture output push to queue
        """
        try:
            r = subprocess.run(
                command,
                shell=True,
                cwd=WORKDIR,
                capture_output=True,
                text=True,
                timeout=300,
            )
            output = (r.stdout + r.stderr).strip()[:50000]
            status = "completed"
        except Exception as e:
            output = "Error : Time out(300s)"
            status = "error"
        self.tasks[task_id]["status"] = status
        self.tasks[task_id]["result"] = output or "(no output)"

        with self._lock:
            self._notification_queue.append(
                {
                    "task_id": task_id,
                    "status": status,
                    "command": command[:80],
                    "result": (output or "(no output)")[:500],
                }
            )

    def check(self, task_id: str):
        if task_id:
            t = self.tasks.get(task_id)
            if not t:
                return f"Error Unknow task {task_id}"
            return (
                f"[{t['status']}] {t['command'][:60]}\n{t.get('result') or '(running)'}"
            )
        lines = []
        for tid, t in self.tasks.items():
            lines.append(f"{tid}: [{t['status']}] {t['command'][:60]}")
        return "\n".join(lines) if lines else "No background tasks."

    def drain_notifications(self) -> list:
        """Return and clear all pending completion notifications."""
        with self._lock:
            notifs = list(self._notification_queue)
            self._notification_queue.clear()
        return notifs


BG = BackgroundManager()


# ======================================================================================
def safe_path(p: str):
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"path escapes workspace:{p}")
    return path


def run_read(path: str, limit: int = None):
    text = safe_path(path).read_text()
    lines = text.splitlines()
    if limit and limit < len(lines):
        lines = lines[:limit]
    return "\n".join(lines)[:50000]


def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


def run_bash(command: str):
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


def run_subagent(prompt: str):
    sub_messages = [{"role": "user", "content": prompt}]
    for _ in range(30):
        response = client.chat.completions.create(
            model=model_id,
            messages=sub_messages,
            tool_choice="auto",
            tools=CHILD_TOOLS,
            max_tokens=8000,
        )

        sub_messages.append(
            {"role": "assistant", "content": response.choices[0].message.content}
        )

        if response.choices[0].finish_reason != "stop":
            break
        results = []
        tool_calls = response.choices[0].message.tool_calls
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            args = tool_call.function.arguments
            args = json.loads(args)
            handler = TOOL_HANDERS.get(tool_name)
            try:
                output = handler(**args) if handler else f"Unknow tool: {tool_name}"
            except Exception as e:
                output = f"Error: {e}"
            results.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": output,
                }
            )
        sub_messages.extend(results)
    final_content = response.choices[0].message.content

    return final_content.strip() if final_content.strip() else "(no output)"


# +++++++++++++++++++++++++++++++工具定义+++++++++++++++++++++++++++++++++++++++++++++++++++++

TOOL_HANDERS = {
    "bash": lambda **kw: run_bash(kw["command"]),
    "read_file": lambda **kw: run_read(kw["path"], kw["limit"]),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "load_skill": lambda **kw: SKILL_LOADER.get_content(kw["name"]),
    "compact": lambda **kw: "Manual compression requested.",
    "task_create": lambda **kw: TASKS.create(kw["subject"], kw.get("description", "")),
    "task_update": lambda **kw: TASKS.update(
        kw["task_id"],
        kw.get("status"),
        kw.get("addBlockedBy"),
        kw.get("removeBlockedBy"),
    ),
    "task_list": lambda **kw: TASKS.list_all(),
    "task_get": lambda **kw: TASKS.get(kw["task_id"]),
    "background_run": lambda **kw: BG.run(kw["command"]),
    "check_background": lambda **kw: BG.check(kw.get("task_id")),
}

CHILD_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "background_run",
            "description": "Run command in background thread. Returns task_id immediately.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command to run in the background",
                    }
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_background",
            "description": "Check background task status. Omit task_id to list all.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The ID of the background task to check (optional, omit to list all tasks)",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task_create",
            "description": "Create a new task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": {
                        "type": "string",
                        "description": "The subject or title of the task",
                    },
                    "description": {
                        "type": "string",
                        "description": "Detailed description of the task",
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
            "description": "Update a task's status or dependencies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "integer",
                        "description": "The ID of the task to update",
                    },
                    "status": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "completed"],
                        "description": "The new status of the task",
                    },
                    "addBlockedBy": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "List of task IDs to add as dependencies",
                    },
                    "removeBlockedBy": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "List of task IDs to remove from dependencies",
                    },
                },
                "required": ["task_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task_list",
            "description": "List all tasks with status summary.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task_get",
            "description": "Get full details of a task by ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "integer",
                        "description": "The ID of the task to retrieve",
                    }
                },
                "required": ["task_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compact",
            "description": "Trigger manual conversation compression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "focus": {
                        "type": "string",
                        "description": "What content or information to preserve in the summary",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_skill",
            "description": "Load specialized knowledge by name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Skill name to load"}
                },
                "required": ["name"],
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
]

# -- Parent tools: base tools + task dispatcher --
PARENT_TOOLS = CHILD_TOOLS + [
    {
        "type": "function",
        "function": {
            "name": "task",
            "description": "Spawn a subagent with fresh context. It shares the filesystem but not conversation history.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The main prompt/instruction for the subagent to execute",
                    },
                    "description": {
                        "type": "string",
                        "description": "Short description of the task",
                    },
                },
                "required": ["prompt"],
            },
        },
    }
]


def agent_loop(messages: list) -> None:
    while True:
        notifs = BG.drain_notifications()
        if notifs and messages:
            notif_text = "\n".join(
                f"[bg:{n['task_id']}] {n['status']}: {n['result']}" for n in notifs
            )
            messages.append(
                {
                    "role": "user",
                    "content": f"<background-results>\n{notif_text}\n</background-results>",
                }
            )
        micro_compact(messages)
        if estimate_tokens(messages) > THRESHOLD:
            print("auto compact triggered")
            messages[:] = auto_compact(messages)

        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            tool_choice="auto",
            tools=PARENT_TOOLS,
            max_tokens=8000,
        )
        messages.append(response.choices[0].message.model_dump())

        print(f"> Assistant: {response.choices[0].message.content}")
        print(f"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        # 3.从返回内容中，寻找是否参数调用
        if response.choices[0].finish_reason != "tool_calls":
            return
        results = []
        manual_compact = False
        tool_calls = response.choices[0].message.tool_calls
        if tool_calls:
            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                if tool_name == "task":
                    args = json.loads(tool_call.function.arguments)
                    desc = args.get("description", "")
                    prompt = args.get("prompt", "")
                    print(f">task: {desc} : prompt: {prompt}")
                    output = run_subagent(prompt)
                elif tool_name == "compact":
                    manual_compact = True
                    output = "Compressing"
                else:
                    args = json.loads(tool_call.function.arguments)
                    handler = TOOL_HANDERS.get(tool_name)
                    print(f"工具调用: {tool_name}")
                    try:
                        output = (
                            handler(**args) if handler else f"Unknow tool :{tool_name}"
                        )
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
                if tool_name == "todo":
                    used_todo = True
        # 三轮没有规划任务强制提醒
        # rounds_since_todo = 0 if used_todo else rounds_since_todo + 1
        # if rounds_since_todo > 3:
        #     results.append(
        #         {"role": "user", "content": "<reminder>Update your todos. </reminder>"}
        #     )
        messages.extend(results)
        if manual_compact:
            print("[manual compact]")
            messages[:] = auto_compact(messages)
            return


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms02 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(block.text)
        print()
