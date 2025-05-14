import sys
import chat
import json
import re
import random
import string

from datetime import datetime
from typing_extensions import TypedDict
from stub import ManusAgent
from langchain_core.prompts import PromptTemplate
from langgraph.prebuilt.chat_agent_executor import AgentState
from typing_extensions import Annotated, TypedDict
from langgraph.graph.message import add_messages
from typing import Literal

from template import get_prompt_template
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langgraph.constants import START, END
from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage, AIMessage

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

import utils
config = utils.load_config()

mcp_tools: list[str]
tool_list: list[BaseTool]

def update_mcp_tools(tools: list[BaseTool]):
    global mcp_tools, tool_list
    tool_list = tools

    mcp_tools = []
    for tool in tools:
        name = tool.name
        description = tool.description
        description = description.replace("\n", "")
        mcp_tools.append(f"{name}: {description}")
        # logger.info(f"mcp_tools: {mcp_tools}")

message_queue = Queue()
def show_info(message: str):
    logger.info(message)
    if hasattr(show_info, 'callback'):
        message_queue.put(message)

class State(TypedDict):
    full_plan: str
    messages: Annotated[list, add_messages]
    final_response: str
    report: str

def Coordinator(state: State) -> dict:
    """Coordinator node that communicate with customers."""
    logger.info(f"###### Coordinator ######")

    question = state["messages"][0].content
    logger.info(f"question: {question}")

    prompt_name = "coordinator"

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
    
    return {
        "final_response": final_response
    }

def to_planner(state: State) -> str:
    logger.info(f"###### to_planner ######")
    # logger.info(f"state: {state}")

    if "final_response" in state and state["final_response"] != "":
        next = END
    else:
        next ="Planner"

    return next

def Planner(state: State, config: dict) -> dict:
    logger.info(f"###### Planner ######")
    # logger.info(f"state: {state}")

    request_id = config.get("configurable", {}).get("request_id", "")
    logger.info(f"request_id: {request_id}")
    
    prompt_name = "planner"

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

    if "full_plan" in state and state["full_plan"] != "":
        # show_info(f"{result.content}") # show initial plan
        file = f"artifacts/{request_id}.md"
        with open(file, "a", encoding="utf-8") as f:
            f.write(f"{result.content}\n\n")

        logger.info(f"Plan saved to {file}")

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

def to_operator(state: State, config: dict) -> str:
    logger.info(f"###### to_operator ######")
    # logger.info(f"state: {state}")

    request_id = config.get("configurable", {}).get("request_id", "")
    logger.info(f"request_id: {request_id}")

    if "final_response" in state and state["final_response"] != "":
        logger.info(f"Finished!!!")
        next = "Reporter"

        file = f"artifacts/{request_id}.md"
        with open(file, "a", encoding="utf-8") as f:
            f.write(f"# Final Response\n\n{state["final_response"]}\n\n")
    else:
        logger.info(f"go to Operator...")
        next = "Operator"

    return next

async def Operator(state: State, config: dict) -> dict:
    logger.info(f"###### Operator ######")
    # logger.info(f"state: {state}")

    request_id = config.get("configurable", {}).get("request_id", "")
    prompt_name = "operator"

    system = get_prompt_template(prompt_name)
    # logger.info(f"system_prompt: {system}")

    human = "{input}" 

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
        "input": state,
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

    if next == "FINISHED":
        return
    else:
        tool_info = []
        for tool in tool_list:
            if tool.name == next:
                tool_info.append(tool)
                logger.info(f"tool_info: {tool_info}")
        
        # Agent
        agent, config = chat.create_agent(tool_info)

        messages = [HumanMessage(content=json.dumps(task))]
        response = await agent.ainvoke({"messages": messages}, config)
        # logger.info(f"response of Operator: {response}")

        logger.info(f"result: {response["messages"][-1].content}")
        output = response["messages"][-1].content

        file = f"artifacts/{request_id}.md"
        with open(file, "a", encoding="utf-8") as f:
            f.write(f"# {task}\n\n{output}\n\n")
        
        return {
            "messages": [
                HumanMessage(content=json.dumps(task)),
                AIMessage(content=output)
            ]
        }

def Reporter(state: State) -> dict:
    logger.info(f"###### Reporter ######")

    prompt_name = "Reporter"

    system_prompt=get_prompt_template(prompt_name)
    logger.info(f"system_prompt: {system_prompt}")
    
    llm = chat.get_chat(extended_thinking="Disable")

    human = "{messages}"
    reporter_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", human)
        ]
    )

    prompt = reporter_prompt | llm 
    result = prompt.invoke({
        "messages": state["messages"]
    })
    logger.info(f"result of Reporter: {result}")

    return {
        "report": result.content,
        "full_plan": state["full_plan"]
    }

agent = ManusAgent(
    state_schema=State,
    impl=[
        ("Coordinator", Coordinator),
        ("Planner", Planner),
        ("Operator", Operator),
        ("to_planner", to_planner),
        ("to_operator", to_operator),
        ("Reporter", Reporter),
    ],
)

manus_agent = agent.compile()

async def run(question: str):
    request_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    logger.info(f"request_id: {request_id}")

    file = f"artifacts/{request_id}.md"
    with open(file, "w", encoding="utf-8") as f:
        f.write(f"## {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    inputs = {
        "messages": [HumanMessage(content=question)],
        "final_response": ""
    }
    config = {
        "request_id": request_id,
        "recursion_limit": 50
    }

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
