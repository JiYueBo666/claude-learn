import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import yaml


load_dotenv()


def Loger(msg: str):
    print(msg)


api_key = os.environ["API_KEY"]
base_url = os.environ["BASE_URL"]
model_id = os.environ["MODEL"]

client = OpenAI(base_url=base_url, api_key=api_key)
WORKDIR = Path.cwd()
SYSTEM = f"You are a coding agent at {WORKDIR}. Use the task tool to delegate exploration or subtasks."
SKILLS_DIR = WORKDIR / "skills"


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


# Layer 1: skill metadata injected into system prompt
SYSTEM = f"""You are a coding agent at {WORKDIR}.
Use load_skill to access specialized knowledge before tackling unfamiliar topics.

Skills available:
{SKILL_LOADER.get_descriptions()}"""

# =======================================================================================


class TodoManager:
    def __init__(self):
        self.items = []

    def update(self, items):
        if len(items) > 20:
            raise ValueError("Max 20 dotos allowed")
        validated = []
        in_progress_count = 0
        for i, item in enumerate(items):
            text = str(item.get("text", "")).strip()
            status = str(item.get("status", "")).strip()
            item_id = str(item.get("id", "")).strip()
            if not text:
                raise ValueError(f"Todo {i} text required")
            if not status:
                raise ValueError(f"Todo {i} status required")
            if status == "in_progress":
                in_progress_count += 1
            validated.append({"id": item_id, "text": text, "status": status})
        if in_progress_count > 1:
            raise ValueError("Only one todo can be in_progress")
        self.items = validated
        return self.render()

    def render(self):
        if not self.items:
            return "No todos"
        lines = []
        for item in self.items:
            marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}[
                item["status"]
            ]
            lines.append(f"{marker} #{item['id']} {item['text']}")
        done = sum(1 for t in self.items if t["status"] == "completed")
        lines.append(f"\n({done}/{len(self.items)} completed)")
        return "\n".join(lines)


TODO = TodoManager()


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
    "todo": lambda **kw: TODO.update(kw["items"]),
    "load_skill": lambda **kw: SKILL_LOADER.get_content(kw["name"]),
}

CHILD_TOOLS = [
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


def agent_loop(messages) -> None:
    rounds_since_todo = 0
    while True:
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
        used_todo = False
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
