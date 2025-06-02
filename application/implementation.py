import sys
import chat
import json
import re
import random
import agent

from datetime import datetime
from typing_extensions import TypedDict
from stub import ManusAgent
from typing_extensions import Annotated, TypedDict
from langgraph.graph.message import add_messages

from template import get_prompt_template
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langgraph.constants import START, END
from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles

import logging
import sys
from queue import Queue

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("graph-implementation")

status_msg = []
def get_status_msg(status):
    global status_msg
    status_msg.append(status)

    if status != "end":
        status = " -> ".join(status_msg)
        return "[status]\n" + status + "..."
    else: 
        status = " -> ".join(status_msg)
        return "[status]\n" + status

response_msg = []

import utils
config = utils.load_config()

def get_mcp_tools(tools):
    mcp_tools = []
    for tool in tools:
        name = tool.name
        description = tool.description
        description = description.replace("\n", "")
        mcp_tools.append(f"{name}: {description}")
        # logger.info(f"mcp_tools: {mcp_tools}")

    return mcp_tools

s3_bucket = config["s3_bucket"] if "s3_bucket" in config else None
if s3_bucket is None:
    raise Exception ("No storage!")

message_queue = Queue()
def show_info(message: str):
    logger.info(message)
    if hasattr(show_info, 'callback'):
        message_queue.put(message)

class State(TypedDict):
    full_plan: str
    messages: Annotated[list, add_messages]
    appendix: list[str]
    final_response: str
    report: str

async def Coordinator(state: State, config: dict) -> dict:
    """Coordinator node that communicate with customers."""
    logger.info(f"###### Coordinator ######")

    question = state["messages"][0].content
    logger.info(f"question: {question}")

    prompt_name = "coordinator"

    status_container = config.get("configurable", {}).get("status_container", None)
    response_container = config.get("configurable", {}).get("response_container", None)
    
    logger.info(f"status_container: {status_container}")
    if chat.debug_mode == "Enable" and status_container is not None:
        try:
            logger.info(f"-----> status_container")
            status_container.info(get_status_msg(f"{prompt_name}"))
        except Exception as e:
            logger.error(f"Failed to update status container: {e}")

    system_prompt=get_prompt_template(prompt_name)
    logger.info(f"system_prompt: {system_prompt}")
    
    llm = chat.get_chat(extended_thinking="Disable")
    coordinator_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}"),
        ]
    )

    prompt = coordinator_prompt | llm 
    result = prompt.invoke({
        "question": question
    })
    logger.info(f"result of Coordinator: {result}")

    final_response = ""
    if result.content.find('to_planner') == -1:
        result.content = result.content.split('<next>')[0]

        logger.info(f"next: END")
        final_response = result.content    

    if chat.debug_mode == "Enable" and response_container is not None:
        try:
            response_container.info(result.content)
        except Exception as e:
            logger.error(f"Failed to update response container: {e}")
    
    return {
        "final_response": final_response
    }

async def to_planner(state: State) -> str:
    logger.info(f"###### to_planner ######")
    # logger.info(f"state: {state}")

    if "final_response" in state and state["final_response"] != "":
        next = END
    else:
        next ="Planner"

    return next

async def Planner(state: State, config: dict) -> dict:
    logger.info(f"###### Planner ######")
    # logger.info(f"state: {state}")

    request_id = config.get("configurable", {}).get("request_id", "")
    logger.info(f"request_id: {request_id}")

    status_container = config.get("configurable", {}).get("status_container", None)
    response_container = config.get("configurable", {}).get("response_container", None)
    key_container = config.get("configurable", {}).get("key_container", None)
    tools = config.get("configurable", {}).get("tools", None)

    mcp_tools = get_mcp_tools(tools)
    
    prompt_name = "planner"

    if chat.debug_mode == "Enable":
        status_container.info(get_status_msg(f"{prompt_name}"))

    system = get_prompt_template(prompt_name)
    # logger.info(f"system_prompt of planner: {system}")

    human = "{input}" 

    llm = chat.get_chat(extended_thinking="Disable")
    planner_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", human),
        ]
    )

    prompt = planner_prompt | llm 
    result = prompt.invoke({
        "mcp_tools": mcp_tools,
        "input": state
    })
    logger.info(f"Planner: {result.content}")

    if chat.debug_mode == "Enable":
        key_container.info(result.content)

    # Update the plan into s3
    key = f"artifacts/{request_id}_plan.md"
    time = f"## {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    chat.updata_object(key, time + result.content, 'prepend')

    output = result.content
    if output.find("<status>") != -1:
        status = output.split("<status>")[1].split("</status>")[0]
        logger.info(f"status: {status}")

        if status == "Completed":
            final_response = state["messages"][-1].content
            logger.info(f"final_response: {final_response}")

            return {
                "full_plan": result.content,
                "final_response": final_response                
            }

    return {
        "full_plan": result.content,
    }

async def to_operator(state: State, config: dict) -> str:
    logger.info(f"###### to_operator ######")
    # logger.info(f"state: {state}")

    request_id = config.get("configurable", {}).get("request_id", "")
    logger.info(f"request_id: {request_id}")

    if "final_response" in state and state["final_response"] != "":
        logger.info(f"Finished!!!")
        next = "Reporter"

        key = f"artifacts/{request_id}.md"
        body = f"# Final Response\n\n{state["final_response"]}\n\n"
        chat.updata_object(key, body, 'append')

    else:
        logger.info(f"go to Operator...")
        next = "Operator"

    return next

async def Operator(state: State, config: dict) -> dict:
    logger.info(f"###### Operator ######")
    # logger.info(f"state: {state}")
    appendix = state["appendix"] if "appendix" in state else []

    status_container = config.get("configurable", {}).get("status_container", None)
    response_container = config.get("configurable", {}).get("response_container", None)    
    key_container = config.get("configurable", {}).get("key_container", None)
    tools = config.get("configurable", {}).get("tools", None)

    mcp_tools = get_mcp_tools(tools)
    
    last_state = state["messages"][-1].content
    logger.info(f"last_state: {last_state}")

    full_plan = state["full_plan"]
    logger.info(f"full_plan: {full_plan}")

    request_id = config.get("configurable", {}).get("request_id", "")
    prompt_name = "operator"

    if chat.debug_mode == "Enable":
        status_container.info(get_status_msg(f"{prompt_name}"))

    system = get_prompt_template(prompt_name)
    # logger.info(f"system_prompt: {system}")

    human = (
        "<full_plan>{full_plan}</full_plan>\n"
        "<tools>{mcp_tools}</tools>\n"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", human),
        ]
    )

    logger.info(f"mcp_tools: {mcp_tools}")

    llm = chat.get_chat(extended_thinking="Disable")
    chain = prompt | llm 
    result = chain.invoke({
        "full_plan": full_plan,
        "mcp_tools": mcp_tools
    })
    logger.info(f"result: {result}")
    
    content = result.content
    # Remove control characters
    content = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', content)
    # Try to extract JSON string
    try:
        # Regular expression to find JSON object
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            content = json_match.group(0)
        result_dict = json.loads(content)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        logger.error(f"Problematic content: {content}")
        return {
            "messages": [
                HumanMessage(content="JSON parsing error occurred. Please try again.")
            ]
        }

    next = result_dict["next"]
    logger.info(f"next: {next}")

    task = result_dict["task"]
    logger.info(f"task: {task}")

    if chat.debug_mode == "Enable":
        response_container.info(f"{next}: {task}")

    if next == "FINISHED":
        return
    else:
        tool_info = []
        for tool in tools:
            if tool.name == next:
                tool_info.append(tool)
                logger.info(f"tool_info: {tool_info}")
        
        result, image_url = await agent.run_manus(task, tool_info, status_container, response_container, key_container, "Disable")
        
        logger.info(f"response of Operator: {result}, {image_url}")

        if image_url:
            output_images = ""
            for url in image_url:
                output_images += f"![{task}]({url})\n\n"
            body = f"# {task}\n\n{result}\n\n{output_images}"
            
            logger.info(f"output_images: {output_images}")
            appendix.append(f"{output_images}")
        
        else:
            body = f"# {task}\n\n{result}\n\n"

        key = f"artifacts/{request_id}_steps.md"
        time = f"## {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        chat.updata_object(key, time + body, 'append')
        
        # with open(key, "a", encoding="utf-8") as f:
        #     f.write(body)
        
        return {
            "messages": [
                HumanMessage(content=json.dumps(task)),
                AIMessage(content=body)
            ],
            "appendix": appendix
        }

async def Reporter(state: State, config: dict) -> dict:
    logger.info(f"###### Reporter ######")

    prompt_name = "Reporter"

    status_container = config.get("configurable", {}).get("status_container", None)
    response_container = config.get("configurable", {}).get("response_container", None)

    if chat.debug_mode == "Enable":
        status_container.info(get_status_msg(f"{prompt_name}"))

    request_id = config.get("configurable", {}).get("request_id", "")    
    
    key = f"artifacts/{request_id}_steps.md"
    context = chat.get_object(key)

    logger.info(f"context: {context}")

    system_prompt=get_prompt_template(prompt_name)
    # logger.info(f"system_prompt: {system_prompt}")
    
    llm = chat.get_chat(extended_thinking="Disable")

    human = (
        "다음의 context를 바탕으로 사용자의 질문에 대한 답변을 작성합니다.\n"
        "<question>{question}</question>\n"
        "<context>{context}</context>"
    )
    reporter_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", human)
        ]
    )

    question = state["messages"][0].content
    logger.info(f"question: {question}")

    prompt = reporter_prompt | llm 
    result = prompt.invoke({
        "context": context,
        "question": question
    })
    logger.info(f"result of Reporter: {result}")

    if chat.debug_mode == "Enable":
        response_container.info(result.content)

    key = f"artifacts/{request_id}_report.md"
    time = f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    
    appendix = state["appendix"] if "appendix" in state else []
    values = '\n\n'.join(appendix)
    logger.info(f"values: {values}")

    chat.create_object(key, time + result.content + values)

    if chat.debug_mode == "Enable":
        status_container.info(get_status_msg("end"))

    return {
        "report": result.content
    }

app = ManusAgent(
    state_schema=State,
    impl=[
        ("Coordinator", Coordinator),
        ("Planner", Planner),
        ("Operator", Operator),
        ("to_planner", to_planner),
        ("to_operator", to_operator),
        ("Reporter", Reporter),
    ]
)

manus_agent = app.compile()

async def run(question: str, tools: list[BaseTool], status_container, response_container, key_container, request_id):
    logger.info(f"request_id: {request_id}")

    if chat.debug_mode == "Enable":
        status_container.info(get_status_msg("start"))
        
    inputs = {
        "messages": [HumanMessage(content=question)],
        "final_response": ""
    }
    config = {
        "request_id": request_id,
        "recursion_limit": 50,
        "status_container": status_container,
        "response_container": response_container,
        "key_container": key_container,
        "tools": tools
    }

    # draw a graph
    graph_diagram = manus_agent.get_graph().draw_mermaid_png(
        draw_method=MermaidDrawMethod.API,
        curve_style=CurveStyle.LINEAR
    )    
    random_id = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=8))
    image_filename = f'workflow_{random_id}.png'
    url = chat.upload_to_s3(graph_diagram, image_filename)
    logger.info(f"url: {url}")

    # add plan to report
    key = f"artifacts/{request_id}_plan.md"
    task = "실행 계획"
    output_images = f"![{task}]({url})\n\n"
    body = f"## {task}\n\n{output_images}"
    chat.updata_object(key, body, 'prepend')

    value = None
    async for output in manus_agent.astream(inputs, config):
        for key, value in output.items():
            logger.info(f"Finished running: {key}")
    
    logger.info(f"value: {value}")

    if "full_plan" in value:
        show_info(f"{value['full_plan']}")

    if "report" in value:
        return value["report"]
    else:
        return value["final_response"]
