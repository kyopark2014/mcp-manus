import os
import re
import utils as utils
from datetime import datetime

from langchain_core.prompts import PromptTemplate
from langgraph.prebuilt.chat_agent_executor import AgentState

import logging
import sys

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("template")

def get_prompt_template(prompt_name: str) -> str:
    template = open(os.path.join(os.path.dirname(__file__), f"{prompt_name}.md")).read()
    logger.info(f"template: {template}")

    return template
