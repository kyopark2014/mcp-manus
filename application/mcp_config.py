import chat
import logging
import sys

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("mcp-config")

mcp_user_config = {}    
def load_config(mcp_type):
    if mcp_type == "default":
        return {
            "mcpServers": {
                "search": {
                    "command": "python",
                    "args": [
                        "application/mcp_server_basic.py"
                    ]
                }
            }
        }
    elif mcp_type == "image_generation":
        return {
            "mcpServers": {
                "imageGeneration": {
                    "command": "python",
                    "args": [
                        "application/mcp_server_image_generation.py"
                    ]
                }
            }
        }    
    elif mcp_type == "airbnb":
        return {
            "mcpServers": {
                "airbnb": {
                    "command": "npx",
                    "args": [
                        "-y",
                        "@openbnb/mcp-server-airbnb",
                        "--ignore-robots-txt"
                    ]
                }
            }
        }
    elif mcp_type == "playwright":
        return {
            "mcpServers": {
                "playwright": {
                    "command": "npx",
                    "args": [
                        "@playwright/mcp@latest"
                    ]
                }
            }
        }
    elif mcp_type == "obsidian":
        return {
            "mcpServers": {
                "mcp-obsidian": {
                "command": "npx",
                "args": [
                    "-y",
                    "@smithery/cli@latest",
                    "run",
                    "mcp-obsidian",
                    "--config",
                    "{\"vaultPath\":\"/\"}"
                ]
                }
            }
        }
    elif mcp_type == "aws_diagram":
        return {
            "mcpServers": {
                "awslabs.aws-diagram-mcp-server": {
                    "command": "uvx",
                    "args": ["awslabs.aws-diagram-mcp-server"],
                    "env": {
                        "FASTMCP_LOG_LEVEL": "ERROR"
                    },
                }
            }
        }
    
    elif mcp_type == "aws_documentation":
        return {
            "mcpServers": {
                "awslabs.aws-documentation-mcp-server": {
                    "command": "uvx",
                    "args": ["awslabs.aws-documentation-mcp-server@latest"],
                    "env": {
                        "FASTMCP_LOG_LEVEL": "ERROR"
                    }
                }
            }
        }
    
    elif mcp_type == "aws_cost":
        return {
            "mcpServers": {
                "aws_cost": {
                    "command": "python",
                    "args": [
                        "application/mcp_server_aws_cost.py"
                    ]
                }
            }
        }    
    elif mcp_type == "aws_cloudwatch":
        return {
            "mcpServers": {
                "aws_cloudwatch_log": {
                    "command": "python",
                    "args": [
                        "application/mcp_server_aws_log.py"
                    ]
                }
            }
        }    
    
    elif mcp_type == "aws_storage":
        return {
            "mcpServers": {
                "aws_storage": {
                    "command": "python",
                    "args": [
                        "application/mcp_server_aws_storage.py"
                    ]
                }
            }
        }    
        
    elif mcp_type == "arxiv":
        return {
            "mcpServers": {
                "arxiv-mcp-server": {
                    "command": "npx",
                    "args": [
                        "-y",
                        "@smithery/cli@latest",
                        "run",
                        "arxiv-mcp-server",
                        "--config",
                        "{\"storagePath\":\"/Users/ksdyb/Downloads/ArXiv\"}"
                    ]
                }
            }
        }
    
    elif mcp_type == "firecrawl":
        return {
            "mcpServers": {
                "firecrawl-mcp": {
                    "command": "npx",
                    "args": ["-y", "firecrawl-mcp"],
                    "env": {
                        "FIRECRAWL_API_KEY": chat.firecrawl_key
                    }
                }
            }
        }
    
    elif mcp_type == "aws_rag":
        return {
            "mcpServers": {
                "aws_storage": {
                    "command": "python",
                    "args": [
                        "application/mcp_server_rag.py"
                    ]
                }
            }
        }    

    elif mcp_type == "manus":
        return {
            "mcpServers": {
                "manus": {
                    "command": "python",
                    "args": [
                        "application/mcp_server_manus.py"
                    ]
                }
            }
        }
        
    elif mcp_type == "code_interpreter":
        return {
            "mcpServers": {
                "aws_storage": {
                    "command": "python",
                    "args": [
                        "application/mcp_server_coder.py"
                    ]
                }
            }
        }    
    
    elif mcp_type == "aws_cli":
        return {
            "mcpServers": {
                "aw-cli": {
                    "command": "python",
                    "args": [
                        "application/mcp_server_aws_cli.py"
                    ]
                }
            }
        }    
    
    elif mcp_type == "tavily":
        return {
            "mcpServers": {
                "tavily-mcp": {
                    "command": "npx",
                    "args": ["-y", "tavily-mcp@0.1.4"],
                    "env": {
                        "TAVILY_API_KEY": chat.tavily_key
                    },
                }
            }
        }
    elif mcp_type == "wikipedia":
        return {
            "mcpServers": {
                "wikipedia": {
                    "command": "python",
                    "args": [
                        "application/mcp_server_wikipedia.py"
                    ]
                }
            }
        }    
    elif mcp_type == "terminal":
        return {
            "mcpServers": {
                "iterm-mcp": {
                    "command": "npx",
                    "args": [
                        "-y",
                        "iterm-mcp"
                    ]
                }
            }
        }    
    elif mcp_type == "filesystem":
        return {
            "mcpServers": {
                "filesystem": {
                    "command": "npx",
                    "args": [
                        "@modelcontextprotocol/server-filesystem",
                        "~/"
                    ]
                }
            }
        }
    
    elif mcp_type == "puppeteer":
        return {
            "mcpServers": {
                "puppeteer": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-puppeteer"]
                }
            }
        }
    
    elif mcp_type == "사용자 설정":
        return mcp_user_config

def load_selected_config(mcp_selections: dict[str, bool]):
    #logger.info(f"mcp_selections: {mcp_selections}")
    loaded_config = {}

    # True 값만 가진 키들을 리스트로 변환
    selected_servers = [server for server, is_selected in mcp_selections.items() if is_selected]
    logger.info(f"selected_servers: {selected_servers}")

    for server in selected_servers:
        logger.info(f"server: {server}")

        if server == "image generation":
            config = load_config('image_generation')
        elif server == "aws diagram":
            config = load_config('aws_diagram')
        elif server == "aws document":
            config = load_config('aws_documentation')
        elif server == "aws cost":
            config = load_config('aws_cost')
        elif server == "ArXiv":
            config = load_config('arxiv')
        elif server == "aws cloudwatch":
            config = load_config('aws_cloudwatch')
        elif server == "aws storage":
            config = load_config('aws_storage')
        elif server == "knowledge base":
            config = load_config('aws_rag')
        elif server == "code interpreter":
            config = load_config('code_interpreter')
        elif server == "aws cli":
            config = load_config('aws_cli')
        else:
            config = load_config(server)
        logger.info(f"config: {config}")
        
        if config:
            loaded_config.update(config["mcpServers"])

    logger.info(f"loaded_config: {loaded_config}")
        
    return {
        "mcpServers": loaded_config
    }
