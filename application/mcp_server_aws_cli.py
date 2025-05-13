import logging
import sys
import mcp_basic
import subprocess
import json

from mcp.server.fastmcp import FastMCP 
from typing import List, Optional

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
    logger.info(f"run_aws_cli_command --> command: {command}, subcommand: {subcommand}, options: {options}")
    
    # Build command list
    cmd = ['aws', command, subcommand]
    logger.info(f"run_aws_cli_command --> cmd: {cmd}")
    
    if options:
        options_list = options.split()
        cmd.extend(options_list)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            error_message = f"execution error: {result.stderr}"
            logger.error(error_message)
            return error_message
        return result.stdout
    except Exception as e:
        error_message = f"{str(e)}"
        logger.error(error_message)
        return error_message

@mcp.tool()    
def retrieve_aws_cost(start_date: str, end_date: str, granularity: str = "MONTHLY") -> str:
    """
    Retrieve AWS costs using AWS Cost Explorer for the specified period.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        granularity: Data grouping unit (DAILY, MONTHLY, HOURLY)
    
    Returns:
        AWS cost information in JSON format
    """
    logger.info(f"retrieve_aws_cost --> start_date: {start_date}, end_date: {end_date}, granularity: {granularity}")
    
    cmd = [
        'aws', 'ce', 'get-cost-and-usage',
        '--time-period', f'Start={start_date},End={end_date}',
        '--granularity', granularity,
        '--metrics', 'UnblendedCost',
        '--group-by', 'Type=DIMENSION,Key=SERVICE'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        error_message = f"Cost retrieval execution error: {e.stderr}"
        logger.error(error_message)
        return error_message
    except Exception as e:
        error_message = f"{str(e)}"
        logger.error(error_message)
        return error_message

@mcp.tool()    
def list_objects_by_cli(bucket_name: str, prefix: Optional[str] = None, delimiter: Optional[str] = None, region: str = "us-west-2") -> str:
    """
    List objects in an S3 bucket using list-objects-v2 command.
    
    Args:
        bucket_name: Name of the S3 bucket
        prefix: Optional prefix to filter objects (e.g., 'folder/'). Default is None.
        delimiter: Optional delimiter to group objects (e.g., '/'). Default is None.
        region: AWS region (e.g., 'us-west-2'). Default is us-west-2.
    
    Returns:
        List of objects in the bucket in JSON format
    """
    logger.info(f"list_objects --> bucket_name: {bucket_name}, prefix: {prefix}, delimiter: {delimiter}, region: {region}")
    
    # First check if bucket exists
    check_cmd = ['aws', 's3api', 'head-bucket', '--bucket', bucket_name, '--region', region]
    try:
        subprocess.run(check_cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        error_message = f"Bucket {bucket_name} does not exist or is not accessible: {e.stderr}"
        logger.error(error_message)
        return error_message
    
    # If bucket exists, list objects
    cmd = ['aws', 's3api', 'list-objects-v2', '--bucket', bucket_name, '--region', region]
    
    if prefix:
        cmd.extend(['--prefix', prefix])
    if delimiter:
        cmd.extend(['--delimiter', delimiter])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            error_message = f"Failed to list objects: {result.stderr}"
            logger.error(error_message)
            return error_message
        return result.stdout
    except Exception as e:
        error_message = f"Error executing S3 command: {str(e)}"
        logger.error(error_message)
        return error_message

@mcp.tool()    
def get_ec2_instances(region: str = "us-west-2") -> str:
    """
    Retrieve EC2 instance information using AWS CLI.
    
    Args:
        region: AWS region (e.g., 'us-west-2'). Default is us-west-2.
    
    Returns:
        EC2 instance information in JSON format
    """
    logger.info(f"get_ec2_instances --> region: {region}")
    
    cmd = ['aws', 'ec2', 'describe-instances', '--region', region]
    logger.info(f"get_ec2_instances --> cmd: {cmd}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        error_message = f"Failed to get EC2 instances: {e.stderr}"
        logger.error(error_message)
        return error_message
    except Exception as e:
        error_message = f"Error executing EC2 command: {str(e)}"
        logger.error(error_message)
        return error_message

@mcp.tool()    
def list_secrets(region: str = "us-west-2") -> str:
    """
    List secrets in AWS Secrets Manager.
    
    Args:
        region: AWS region (e.g., 'us-west-2'). Default is us-west-2.
    
    Returns:
        List of secrets in JSON format
    """
    logger.info(f"list_secrets --> region: {region}")
    
    cmd = ['aws', 'secretsmanager', 'list-secrets', '--region', region]
    logger.info(f"list_secrets --> cmd: {cmd}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        error_message = f"Failed to list secrets: {e.stderr}"
        logger.error(error_message)
        return error_message
    except Exception as e:
        error_message = f"Error executing Secrets Manager command: {str(e)}"
        logger.error(error_message)
        return error_message

@mcp.tool()    
def create_secret(name: str, secret_value: str, description: str = "", region: str = "us-west-2") -> str:
    """
    Create a new secret in AWS Secrets Manager.
    
    Args:
        name: Name of the secret (e.g., 'my-db-password')
        secret_value: The secret value to store (e.g., '{"username":"admin","password":"secret123"}')
        description: Optional description for the secret
        region: AWS region (e.g., 'us-west-2'). Default is us-west-2.
    
    Returns:
        Result of the create-secret operation in JSON format
    """
    logger.info(f"create_secret --> name: {name}, region: {region}")
    
    cmd = [
        'aws', 'secretsmanager', 'create-secret',
        '--name', name,
        '--secret-string', secret_value,
        '--region', region
    ]
    
    if description:
        cmd.extend(['--description', description])
    
    logger.info(f"create_secret --> cmd: {cmd}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        error_message = f"Failed to create secret: {e.stderr}"
        logger.error(error_message)
        return error_message
    except Exception as e:
        error_message = f"Error executing Secrets Manager command: {str(e)}"
        logger.error(error_message)
        return error_message

@mcp.tool()    
def list_s3_buckets(region: str = "us-west-2") -> str:
    """
    List all S3 buckets in the specified region.
    
    Args:
        region: AWS region (e.g., 'us-west-2'). Default is us-west-2.
    
    Returns:
        List of S3 buckets in JSON format
    """
    logger.info(f"list_s3_buckets --> region: {region}")
    
    cmd = ['aws', 's3api', 'list-buckets', '--region', region]
    logger.info(f"list_s3_buckets --> cmd: {cmd}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            error_message = f"Failed to list S3 buckets: {result.stderr}"
            logger.error(error_message)
            return error_message
        return result.stdout
    except Exception as e:
        error_message = f"Error executing S3 command: {str(e)}"
        logger.error(error_message)
        return error_message

@mcp.tool()    
def check_bucket_contents(bucket_name: str, region: str = "us-west-2") -> str:
    """
    Check if a bucket is empty by listing its objects.
    
    Args:
        bucket_name: Name of the S3 bucket
        region: AWS region (e.g., 'us-west-2'). Default is us-west-2.
    
    Returns:
        List of objects in the bucket in JSON format
    """
    logger.info(f"check_bucket_contents --> bucket_name: {bucket_name}, region: {region}")
    
    cmd = [
        'aws', 's3api', 'list-objects-v2',
        '--bucket', bucket_name,
        '--region', region
    ]
    logger.info(f"check_bucket_contents --> cmd: {cmd}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            error_message = f"Failed to check bucket contents: {result.stderr}"
            logger.error(error_message)
            return error_message
        return result.stdout
    except Exception as e:
        error_message = f"Error executing S3 command: {str(e)}"
        logger.error(error_message)
        return error_message

@mcp.tool()    
def delete_bucket_object(bucket_name: str, object_key: str, region: str = "us-west-2") -> str:
    """
    Delete an object from an S3 bucket.
    
    Args:
        bucket_name: Name of the S3 bucket
        object_key: Key (path) of the object to delete
        region: AWS region (e.g., 'us-west-2'). Default is us-west-2.
    
    Returns:
        Result of the delete operation in JSON format
    """
    logger.info(f"delete_bucket_object --> bucket_name: {bucket_name}, object_key: {object_key}, region: {region}")
    
    cmd = [
        'aws', 's3api', 'delete-object',
        '--bucket', bucket_name,
        '--key', object_key,
        '--region', region
    ]
    logger.info(f"delete_bucket_object --> cmd: {cmd}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            error_message = f"Failed to delete object: {result.stderr}"
            logger.error(error_message)
            return error_message
        return result.stdout
    except Exception as e:
        error_message = f"Error executing S3 command: {str(e)}"
        logger.error(error_message)
        return error_message

@mcp.tool()    
def delete_bucket(bucket_name: str, region: str = "us-west-2") -> str:
    """
    Delete an empty S3 bucket.
    
    Args:
        bucket_name: Name of the S3 bucket to delete
        region: AWS region (e.g., 'us-west-2'). Default is us-west-2.
    
    Returns:
        Result of the delete operation in JSON format
    """
    logger.info(f"delete_bucket --> bucket_name: {bucket_name}, region: {region}")
    
    cmd = [
        'aws', 's3api', 'delete-bucket',
        '--bucket', bucket_name,
        '--region', region
    ]
    logger.info(f"delete_bucket --> cmd: {cmd}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            error_message = f"Failed to delete bucket: {result.stderr}"
            logger.error(error_message)
            return error_message
        return result.stdout
    except Exception as e:
        error_message = f"Error executing S3 command: {str(e)}"
        logger.error(error_message)
        return error_message

if __name__ =="__main__":
    print(f"###### main ######")
    mcp.run(transport="stdio")


