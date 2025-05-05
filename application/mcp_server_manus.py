import logging
import sys
import mcp_manus as manus

from typing import Dict, Optional, Any
from mcp.server.fastmcp import FastMCP 

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("manus")

try:
    mcp = FastMCP(
        name = "rag",
        instructions=(
            "You are a helpful assistant. "
            "You generate report or code."
        ),
    )
    logger.info("MCP server initialized successfully")
except Exception as e:
        err_msg = f"Error: {str(e)}"
        logger.info(f"{err_msg}")

######################################
# reporter
######################################
@mcp.tool()
def reporter(user_request: str):
    """
    Write final report.
    user_request: the user's request
    """
    
    return manus.reporter(user_request)

if __name__ =="__main__":
    print(f"###### main ######")
    mcp.run(transport="stdio")


