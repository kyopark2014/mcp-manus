import sys
import os
import utils
import os
import re

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

import json
def load_config():
    config = None
    try:
        with open("/home/config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
            print(f"config: {config}")

    except Exception:
        print("use local configuration")
        with open("application/config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
    
    return config
config = load_config()

team_members: list[str]
tool_list: list[BaseTool]

def update_team_members(tools: list[BaseTool]):
    global team_members, tool_list
    tool_list = tools

    team_members = []
    for tool in tools:
        name = tool.name
        description = tool.description
        description = description.replace("\n", "")
        team_members.append(f"{name}: {description}")
        # logger.info(f"team_members: {team_members}")

class State(TypedDict):
    question : str    
    # Runtime Variables
    full_plan: str
    deep_thinking_mode: bool
    search_before_planning: bool
    final_response: str

    # Messages
    history: list[dict]

def Coordinator(state: State) -> dict:
    """Coordinator node that communicate with customers."""
    logger.info(f"###### Coordinator ######")
    logger.info(f"state: {state}")

    # chat 모듈을 함수 내부에서 임포트
    import chat

    question = state["question"]

    prompt_name = "coordinator"

    system_prompt=get_prompt_template(prompt_name)
    logger.info(f"system_prompt: {system_prompt}")
    
    llm = chat.get_chat(extended_thinking="Disable")
    coordinator_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Question: {question}"),
        ]
    )

    prompt = coordinator_prompt | llm 
    result = prompt.invoke({
        "question": question
    })
    logger.info(f"result of Coordinator: {result}")

    final_response = ""
    if result.content != 'to_planner':
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

message_queue = Queue()

def show_info(message: str):
    logger.info(message)
    if hasattr(show_info, 'callback'):
        message_queue.put(message)

def Planner(state: State) -> dict:
    logger.info(f"###### Planner ######")
    #logger.info(f"state: {state}")

    logger.info(f"team_members: {team_members}")    

    prompt_name = "planner"

    system = get_prompt_template(prompt_name)
    logger.info(f"system_prompt of planner: {system}")

    human = "{input}" 

    import chat
    llm = chat.get_chat(extended_thinking="Disable")
    planner_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", human),
        ]
    )

    prompt = planner_prompt | llm 
    result = prompt.invoke({
        "team_members": team_members,
        "input": state
    })
    show_info(f"Planner: {result.content}")

    return {
        "full_plan": result.content,
        "deep_thinking_mode": state.get("deep_thinking_mode", False),
        "search_before_planning": state.get("search_before_planning", False)
    }

async def Operator(state: State) -> dict:
    logger.info(f"###### Operator ######")
    # logger.info(f"state: {state}")

    prompt_name = "operator"

    system = get_prompt_template(prompt_name)
    logger.info(f"system_prompt: {system}")

    human = "{input}" 

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", human),
        ]
    )

    import chat
    llm = chat.get_chat(extended_thinking="Disable")
    chain = prompt | llm 
    result = chain.invoke({
        "input": state,
        "team_members": team_members
    })
    logger.info(f"result: {result}")
    show_info(f"Operator: {result.content}")

    import json
    result_dict = json.loads(result.content)

    next = result_dict["next"]
    print(f"next: {next}")

    task = result_dict["task"]
    print(f"task: {task}")

    tool_info = []
    for tool in tool_list:
        if tool.name == next:
            tool_info.append(tool)
    
    from langgraph.prebuilt import create_react_agent
    agent = create_react_agent(tools=tool_info, model=llm)

    response = await agent.ainvoke({"messages": task})

    result = response["messages"][-1].content
    logger.info(f"result: {result}")
    
    return {
        "final_response": result
    }

def to_operator(state: State) -> str:
    logger.info(f"###### to_operator ######")
    # logger.info(f"state: {state}")

    if "final_response" in state and state["final_response"] != "":
        return END
    else:
        return "Operator"

def should_end(state: State) -> str:
    logger.info(f"###### should_end ######")
    # logger.info(f"state: {state}")

    if "final_response" in state and state["final_response"] != "":
        return END
    else:
        return "Planner"

agent = ManusAgent(
    state_schema=State,
    impl=[
        ("Coordinator", Coordinator),
        ("Planner", Planner),
        ("Operator", Operator),
        ("to_planner", to_planner),
        ("to_operator", to_operator),
        ("should_end", should_end),
    ],
)

manus_agent = agent.compile()

async def run(question: str):
    inputs = {
        "question": question
    }
    config = {
        "recursion_limit": 50
    }

    value = None
    async for output in manus_agent.astream(inputs, config):
        for key, value in output.items():
            logger.info(f"Finished running: {key}")
    
    logger.info(f"value: {value}")

    return value["final_response"]