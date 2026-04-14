import os
import subprocess
from dataclasses import dataclass

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def Loger(msg:str):
    print(msg)


api_key=os.environ['API_KEY']
base_url=os.environ['BASE_URL']
model_id=os.environ['MODEL']

Loger(f'api key :{api_key}')

client=OpenAI(
    base_url=base_url,
    api_key=api_key
)

SYSTEM=(
    f'you are a coding agent at {os.getcwd()}'
    "use bash to inspect and change workspace.Act first,than report clearly"
)
TOOLS = [
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
                        "description": "Shell command to execute"
                    }
                },
                "required": ["command"]
            }
        }
    }
]
@dataclass
class LoopState:
    message:list
    turn_count:int=1
    transition_reason:str|None=None

def run_bash(command:str):
    dangerous=["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(item in command for item in dangerous):
        return 'Error:Dangerous command blocks'
    
    try:
        result=subprocess.run(
            command,
            shell=True,
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            timeout=120
        )
    except subprocess.TimeoutExpired:
        return 'Error: Time out (120s)'
    except (FileNotFoundError,OSError) as e:
        return f'Error: {e}'
    
    output=(result.stdout+result.stderr).strip()
    return output[:50000] if output else "(no output)"

def extract_text(content) -> str:
    if not isinstance(content, list):
        return ""
    texts = []
    for block in content:
        text = getattr(block, "text", None)
        if text:
            texts.append(text)
    return "\n".join(texts).strip()

def execute_tool_calls(tool_calls):
    print("执行工具调用")
    results=[]
    #列表中是一系列要调用的函数
    #遍历调用列表，获取函数，函数名，函数参数
    for tool_call in tool_calls:
        func=tool_call.function
        func_name=func.name
        #函数参数json，字符串格式,转为字典
        args=eval(func.arguments)

        if func_name=='bash':
            #取参数字段
            command=args['command']
            print(f"\033[33m$ {command}\033[0m")
            output = run_bash(command)
            print(output[:200])

            # OpenAI 格式返回 tool 消息
            results.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": output
            })
    return results


def run_one_turn(state:LoopState):
    #1.定义服务端，传入工具列表和历史消息
    response=client.chat.completions.create(
        model=model_id,
        messages=state.message,
        tool_choice='auto',
        tools=TOOLS,
        max_tokens=8000
    )
    #.2.获取返回内容，解析choices[0].message字段，存入历史
    choice=response.choices[0]
    ai_message=choice.message
    state.message.append(ai_message.model_dump())

    #3.从返回内容中，寻找是否参数调用
    if choice.finish_reason!='tool_calls':
        state.transition_reason=None
        return False
    #4. 如果有参数调用列表，获取该列表进行处理
    results=execute_tool_calls(ai_message.tool_calls)

    if not results:
        state.transition_reason=None
        return False
    state.message.extend(results)
    state.turn_count+=1
    state.transition_reason='tool_result'
    return True

def agent_loop(state: LoopState) -> None:
    while run_one_turn(state):
        pass

if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms01 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role":"system","content":SYSTEM})
        history.append({"role": "user", "content": query})
        state = LoopState(message=history)
        agent_loop(state)

        final_text = extract_text(history[-1]["content"])
        if final_text:
            print(final_text)
        print()