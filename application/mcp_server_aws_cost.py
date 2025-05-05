import logging
import sys
import mcp_cost as cost

from mcp.server.fastmcp import FastMCP 

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("mcp-server-aws-cost")

try:
    mcp = FastMCP(
        name = "AWS_Cost",
        instructions=(
            "You are a helpful assistant. "
            "You can retrieve AWS Cost and provide insights."
        ),
    )
    logger.info("MCP server initialized successfully")
except Exception as e:
        err_msg = f"Error: {str(e)}"
        logger.info(f"{err_msg}")

######################################
# AWS Cost
######################################
@mcp.tool()
def aws_cost_loader(days: int=30, region: str='us-west-2') -> list:
    """
    load aws cost data
    days: the number of days looking for cost data
    region: the name of aws region
    return: cost data during days
    """

    return cost.get_cost_analysis(days=days, region=region)

@mcp.tool()
def create_cost_visualizations() -> list:
    """
    create aws cost visualizations
    """

    return cost.create_cost_visualizations()

@mcp.tool()
def generate_cost_insights() -> str:
    """
    generate cost insights
    """

    return cost.generate_cost_insights()

@mcp.tool()
def generate_cost_insights(question: str) -> str:
    """
    generate cost report only when the user clearly requests
    question: the question to ask
    """

    return cost.ask_cost_insights(question)

######################################
# AWS Logs
######################################

if __name__ =="__main__":
    print(f"###### main ######")
    mcp.run(transport="stdio")


