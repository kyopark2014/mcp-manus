import re
import base64
import logging
import traceback
import chat 
import sys
import uuid

from urllib import parse
from langchain_experimental.tools import PythonAstREPLTool
from io import BytesIO
from template import get_prompt_template
from langchain_core.prompts import ChatPromptTemplate

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("manus")

repl = PythonAstREPLTool()

def reporter(state):
    """
    Write final report.
    code: the Python code was written in English
    """
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

    import chat
    llm = chat.get_chat(extended_thinking="Disable")
    chain = prompt | llm 
    result = chain.invoke({
        "input": state,
        "team_members": team_members
    })
    logger.info(f"result: {result}")
    return result.content

