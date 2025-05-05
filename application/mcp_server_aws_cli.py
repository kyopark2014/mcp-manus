import logging
import sys
import mcp_basic
import subprocess

from mcp.server.fastmcp import FastMCP 

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("aws-cli")

try:
    mcp = FastMCP(
        name = "tools",
        instructions=(
            "You are a helpful assistant. "
            "You can use tools for the user's question and provide the answer."
        ),
    )
    logger.info("MCP server initialized successfully")
except Exception as e:
        err_msg = f"Error: {str(e)}"
        logger.info(f"{err_msg}")


######################################
# AWS CLI
######################################

@mcp.tool()    
def run_aws_cli(command: str, subcommand: str, options: str) -> str:
    """
    run aws command using aws cli and then return the result
    command: AWS CLI command (e.g., s3, ec2, dynamodb)
    subcommand: subcommand for the AWS CLI command (e.g., ls, cp, get-object)
    options: additional options for the command (e.g., --bucket mybucket)
    return: command output as string
    """   
    logger.info(f"run_aws_cli_ommand --> command: {command}, subcommand: {subcommand}, options: {options}")
    
    # 명령어 구성
    cmd = ['aws', command, subcommand]
    logger.info(f"run_aws_cli_ommand --> cmd: {cmd}")
    
    if options:
        options_list = options.split()
        cmd.extend(options_list)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        error_message = f"execution error: {e.stderr}"
        logger.error(error_message)
        return error_message
    except Exception as e:
        error_message = f"{str(e)}"
        logger.error(error_message)
        return error_message


if __name__ =="__main__":
    print(f"###### main ######")
    mcp.run(transport="stdio")


