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

def reporter(user_request: str):
    """
    Write final report.
    user_request: the user's request
    """
    prompt_name = "reporter"

    system = get_prompt_template(prompt_name)
    # logger.info(f"system_prompt: {system}")

    human = "{input}" 

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", human)
        ]
    )

    import chat
    llm = chat.get_chat(extended_thinking="Disable")
    chain = prompt | llm 
    result = chain.invoke({
        "user_request": user_request,
        "input": "Question에 맞는 적절한 답변을 리포트로 작성해주세요."
    })
    logger.info(f"result: {result}")

    return result.content

# async def get_tool_info():
#     import mcp_config
#     mcp_selections = {"tavily", "ArXiv", "wikipedia", "aws document"}
#     mcp_json = mcp_config.load_selected_config(mcp_selections)
#     logger.info(f"mcp_json: {mcp_json}")

#     import mcp_client
#     server_params = mcp_client.load_multiple_mcp_server_parameters(mcp_json)
#     logger.info(f"server_params: {server_params}")

#     from langchain_mcp_adapters.client import MultiServerMCPClient
#     async with MultiServerMCPClient(server_params) as client:
#         tools = client.get_tools()
#         logger.info(f"tools: {tools}")

#         return tools

