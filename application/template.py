import os
import re
import utils as utils
from datetime import datetime

from langchain_core.prompts import PromptTemplate
from langgraph.prebuilt.chat_agent_executor import AgentState

logger = utils.CreateLogger("graph-implementation")

def get_prompt_template(prompt_name: str) -> str:
    template = open(os.path.join(os.path.dirname(__file__), f"{prompt_name}.md")).read()
    logger.info(f"template: {template}")

    return template
