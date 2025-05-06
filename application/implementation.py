import sys
import chat

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
from langchain_core.messages import HumanMessage

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
            logger.info(f"config: {config}")

    except Exception:
        logger.info("use local configuration")
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

def Planner(state: State) -> dict:
    logger.info(f"###### Planner ######")
    # logger.info(f"state: {state}")

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
        "team_members": team_members,
        "input": state
    })
    logger.info(f"Planner: {result.content}")

    if "full_plan" in state and state["full_plan"] != "":
        show_info(f"{result.content}") # shoe initial plan

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

def to_operator(state: State) -> str:
    logger.info(f"###### to_operator ######")
    # logger.info(f"state: {state}")

    if "final_response" in state and state["final_response"] != "":
        logger.info(f"Finished!!!")
        next = "Reporter"
    else:
        logger.info(f"go to Operator...")
        next = "Operator"

    return next

# def write_result(result: str):    
#     file_path = "./artifacts/all_results.txt"
#     os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
#     with open(file_path, "a", encoding="utf-8") as f:
#         f.write(f"{result}\n\n\n")

async def Operator(state: State) -> dict:
    logger.info(f"###### Operator ######")
    # logger.info(f"state: {state}")

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

    llm = chat.get_chat(extended_thinking="Disable")
    chain = prompt | llm 
    result = chain.invoke({
        "input": state,
        "team_members": team_members
    })
    logger.info(f"result: {result}")

    import json
    result_dict = json.loads(result.content)

    next = result_dict["next"]
    logger.info(f"next: {next}")

    task = result_dict["task"]
    logger.info(f"task: {task}")

    if next == "FINISHED":
        next = END
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
        
    return {
        "messages": [response["messages"][-1]]
    }

def Reporter(state: State) -> dict:
    logger.info(f"###### Reporter ######")

    question = state["messages"][0].content
    logger.info(f"question: {question}")

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

    show_info(f"{state['full_plan']}")
    
    return {
        "report": result.content 
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
    inputs = {
        "messages": [HumanMessage(content=question)],
        "final_response": ""
    }
    config = {
        "recursion_limit": 50
    }

    value = None
    async for output in manus_agent.astream(inputs, config):
        for key, value in output.items():
            logger.info(f"Finished running: {key}")
    
    logger.info(f"value: {value}")

    if "report" in value:
        return value["report"]
    else:
        return value["final_response"]
