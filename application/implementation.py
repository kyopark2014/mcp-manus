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

config = utils.load_config()
logger = utils.CreateLogger("graph-implementation")

class State(TypedDict):
    question : str
    team_members : list[str]
    # Runtime Variables
    next: str
    full_plan: str
    deep_thinking_mode: bool
    search_before_planning: bool
    full_response: str

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
        "question": question,
        "CURRENT_TIME": datetime.now().strftime("%a %b %d %Y %H:%M:%S %z"),
    })
    logger.info(f"result of Coordinator: {result}")

    if result.content == 'to_planner()':
        logger.info(f"next: Planner")
        return {
            "next": "Planner"
        }
    else:
        logger.info(f"next: END")
        return {
            "next": END,
            "full_response": result.content
        }

def Planner(state: State) -> dict:
    logger.info(f"###### Planner ######")
    logger.info(f"state: {state}")

    team_members = state["team_members"]
    logger.info(f"team_members: {team_members}")

    prompt_name = "planner"

    system_prompt = get_prompt_template(prompt_name)
    logger.info(f"system_prompt: {system_prompt}")

    import chat
    llm = chat.get_chat(extended_thinking="Disable")
    planner_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt)
        ]
    )

    prompt = planner_prompt | llm 
    result = prompt.invoke({
        "CURRENT_TIME": datetime.now().strftime("%a %b %d %Y %H:%M:%S %z"),
        "team_members": team_members
    })
    logger.info(f"result: {result}")

    return {
        "next": "Operator",
        "full_plan": state.get("full_plan", ""),
        "deep_thinking_mode": state.get("deep_thinking_mode", False),
        "search_before_planning": state.get("search_before_planning", False)
    }

def Operator(state: State) -> dict:
    logger.info(f"###### Operator ######")
    logger.info(f"state: {state}")
    
    return {
        # Add your state update logic here
    }

def to_planner(state: State) -> str:
    logger.info(f"###### to_planner ######")
    logger.info(f"state: {state}")

    if state["next"] == "Planner":
        return "Planner"
    else:
        return END

def to_operator(state: State) -> str:
    logger.info(f"###### to_operator ######")
    logger.info(f"state: {state}")

    if state["next"] == "Operator":
        return "Operator"
    else:
        return "END"

def should_end(state: State) -> str:
    logger.info(f"###### should_end ######")
    logger.info(f"state: {state}")

    if state["next"] == "END":
        return "END"
    else:
        return "Operator"

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

def run(question: str, toolList: list):
    inputs = {
        "question": question,
        "team_members": toolList
    }
    config = {
        "recursion_limit": 50
    }

    for output in manus_agent.stream(inputs, config):   
        for key, value in output.items():
            logger.info(f"Finished running: {key}")
    
    logger.info(f"value: {value}")

    return value["full_response"]