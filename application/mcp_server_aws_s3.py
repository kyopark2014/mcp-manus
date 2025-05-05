import logging
import sys
import mcp_s3 as storage

from typing import Dict, Optional, Any
from mcp.server.fastmcp import FastMCP 

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("aws-s3")

try:
    mcp = FastMCP(
        name = "tools",
        instructions=(
            "You are a helpful assistant. "
            "You can check the status of Amazon S3 and retrieve insights."
        )
    )
    logger.info("MCP server initialized successfully")
except Exception as e:
        err_msg = f"Error: {str(e)}"
        logger.info(f"{err_msg}")

######################################
# AWS S3
######################################
from typing import List, Optional
from mcp.types import Resource

@mcp.tool()
async def list_buckets(
    start_after: Optional[str] = None,
    max_buckets: Optional[int] = 10,
    region: Optional[str] = "us-west-2"
) -> List[dict]:
    """
    List S3 buckets using async client with pagination
    """
    logger.info(f"list_buckets --> start_after: {start_after}, max_buckets: {max_buckets}, region: {region}")

    return await storage.list_buckets(start_after, max_buckets, region)
    
@mcp.tool()  
async def list_objects(
    bucket_name: str, 
    prefix: Optional[str] = "", 
    max_keys: Optional[int] = 1000,
    region: Optional[str] = "us-west-2"
) -> List[dict]:
    """
    List objects in a specific bucket using async client with pagination
    Args:
        bucket_name: Name of the S3 bucket
        prefix: Object prefix for filtering
        max_keys: Maximum number of keys to return,
        region: Name of the aws region
    """
    logger.info(f"list_objects --> bucket_name: {bucket_name}, prefix: {prefix}, max_keys: {max_keys}, region: {region}")

    return await storage.list_objects(bucket_name, prefix, max_keys, region)

@mcp.tool()    
async def list_resources(
    start_after: Optional[str] = None,
    max_buckets: Optional[int] = 10,
    region: Optional[str] = "us-west-2"
) -> List[Resource]:
    """
    List S3 buckets and their contents as resources with pagination
    Args:
        start_after: Start listing after this bucket name
    """
    logger.info(f"list_resources --> start_after: {start_after}, max_buckets: {max_buckets}, region: {region}")
    
    return await storage.list_resources(start_after, max_buckets, region)
    
######################################
# AWS Logs
######################################

if __name__ =="__main__":
    print(f"###### main ######")
    mcp.run(transport="stdio")


