import json
import subprocess
import os
import threading
import time
from config import TASKS_DIR, TRANSCRIPT_DIR, WORKDIR, MAX_CICLE
from openai import OpenAI
from functools import lru_cache
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


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


def make_identity_block(name: str, role: str, team_name: str) -> dict:
    """
    上下文压缩后，重新身份注入
    """
    return {
        "role": "user",
        "content": f"<identity>You are '{name}', role: {role}, team: {team_name}. Continue your work.</identity>",
    }


@lru_cache()
def get_ai_client() -> OpenAI:
    """
    AI 客户端工厂（单例模式）
    全局只会初始化一次，重复调用返回同一个实例
    """
    api_key = os.environ.get("API_KEY")
    base_url = os.environ.get("BASE_URL")
    model = os.environ.get("MODEL")

    # 增加健壮性，大型项目必备
    if not api_key or not base_url or not model:
        raise EnvironmentError("请配置环境变量 API_KEY / BASE_URL / MODEL")

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    return client, model


def _timer_log(func):
    """装饰器：自动记录函数执行时长"""

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = (time.time() - start) * 1000  # 转换为毫秒
        logger.info(
            f"{func.__name__} 执行耗时: {elapsed:.2f}ms (args: {args[0] if args else 'N/A'})"
        )
        return result

    return wrapper


def _safe_path(p: str):
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"path escapes workspace:{p}")
    return path


@_timer_log
def _run_read(path: str, limit: int = None):
    text = _safe_path(path).read_text()
    lines = text.splitlines()
    if limit and limit < len(lines):
        lines = lines[:limit]
    return "\n".join(lines)[:50000]


@_timer_log
def _run_write(path: str, content: str) -> str:
    try:
        fp = _safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


@_timer_log
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


@_timer_log
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


@_timer_log
def _run_subagent(prompt: str, agent_type: str = "Explore"):
    sub_tools = [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Run command.",
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
                "description": "Read file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path to read"}
                    },
                    "required": ["path"],
                },
            },
        },
    ]
    if agent_type == "Explore":
        sub_tools += [
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "File path to write",
                            },
                            "content": {
                                "type": "string",
                                "description": "Content to write into the file",
                            },
                        },
                        "required": ["path", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "edit_file",
                    "description": "Edit file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "File path to edit",
                            },
                            "old_text": {
                                "type": "string",
                                "description": "Text to be replaced",
                            },
                            "new_text": {
                                "type": "string",
                                "description": "New text to replace",
                            },
                        },
                        "required": ["path", "old_text", "new_text"],
                    },
                },
            },
        ]
    sub_handlers = {
        "bash": lambda **kw: _run_bash(kw["command"]),
        "read_file": lambda **kw: _run_read(kw["path"]),
        "write_file": lambda **kw: _run_write(kw["path"], kw["content"]),
        "edit_file": lambda **kw: _run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    }
    sub_msgs = [{"role": "user", "content": prompt}]
    resp = None
    client: OpenAI
    client, model = get_ai_client()
    for _ in range(MAX_CICLE):
        resp = client.chat.completions.create(
            model=model,
            messages=sub_msgs,
            tool_choice="auto",
            tools=sub_tools,
            max_tokens=8000,
        )
        # add history
        sub_msgs.append(
            {"role": "assistant", "content": resp.choices[0].message.content}
        )

        # check stop reason
        finish_reason = resp.choices[0].finish_reason
        if finish_reason != "tool_calls":
            return
        results = []
        tool_calls = resp.choices[0].message.tool_calls
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            handler = sub_handlers.get(tool_name)
            logger.info(f"Subagent 工具调用: {tool_name},tool_id: {tool_call.id}")
            try:
                output = handler(**args) if handler else f"Unknow tool :{tool_name}"
            except Exception as e:
                logger.warning(
                    f"Subagent 工具调用失败: {tool_name},tool_id: {tool_call.id}"
                )
                output = f"Error: {e}"
            results.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": output,
                }
            )
        sub_msgs.extend(results)
    if resp:
        return (
            "".join(b.text for b in resp.content if hasattr(b, "text"))
            or "(no summary)"
        )
    return "(subagent failed)"


# === SECTION: compression (s06) ===
def estimate_tokens(messages: list) -> int:
    return len(json.dumps(messages, default=str)) // 4


def microcompact(messages: list):
    indices = []
    for i, msg in enumerate(messages):
        if msg["role"] == "user" and isinstance(msg.get("content"), list):
            for part in msg["content"]:
                if isinstance(part, dict) and part.get("type") == "tool_result":
                    indices.append(part)
    if len(indices) <= 3:
        return
    for part in indices[:-3]:
        if isinstance(part.get("content"), str) and len(part["content"]) > 100:
            part["content"] = "[cleared]"


def auto_compact(messages: list) -> list:
    TRANSCRIPT_DIR.mkdir(exist_ok=True)
    path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with open(path, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg, default=str) + "\n")
    conv_text = json.dumps(messages, default=str)[-80000:]
    # ===================================================================
    client: OpenAI
    client, model = get_ai_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": f"Summarize for continuity:\n{conv_text}"}
        ],
        max_tokens=2000,
    )
    summary = resp.choices[0].message.content
    return [
        {"role": "user", "content": f"[Compressed. Transcript: {path}]\n{summary}"},
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


shutdown_requests = {}
plan_requests = {}

_tracker_lock = threading.Lock()
_claim_lock = threading.Lock()


from models import *
from models import BUS


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
