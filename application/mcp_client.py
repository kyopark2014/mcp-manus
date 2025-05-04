import utils
import info
import boto3
import traceback
import uuid
import asyncio
import json

from botocore.config import Config
from langchain_aws import ChatBedrock
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

import logging
import sys

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("mcp-client")

def load_multiple_mcp_server_parameters(mcp_json):
    logger.info(f"mcp_json: {mcp_json}")

    mcpServers = mcp_json.get("mcpServers")
    logger.info(f"mcpServers: {mcpServers}")
  
    server_info = {}
    if mcpServers is not None:
        command = ""
        args = []
        for server in mcpServers:
            logger.info(f"server: {server}")

            config = mcpServers.get(server)
            logger.info(f"config: {config}")

            if "command" in config:
                command = config["command"]
            if "args" in config:
                args = config["args"]
            if "env" in config:
                env = config["env"]

                server_info[server] = {
                    "command": command,
                    "args": args,
                    "env": env,
                    "transport": "stdio"
                }
            else:
                server_info[server] = {
                    "command": command,
                    "args": args,
                    "transport": "stdio"
                }
    logger.info(f"server_info: {server_info}")

    return server_info



def show_status_message(response, st, debug_mode):
    image_url = []
    references = []
    for i, re in enumerate(response):
        logger.info(f"message[{i}]: {re}")

        if i==len(response)-1:
            break

        if isinstance(re, AIMessage):
            logger.info(f"AIMessage: {re}")
            if re.content:
                logger.info(f"content: {re.content}")
                content = re.content
                if len(content) > 500:
                    content = content[:500] + "..."       
                if debug_mode == "Enable": 
                    st.info(f"{content}")
            if hasattr(re, 'tool_calls') and re.tool_calls:
                logger.info(f"Tool name: {re.tool_calls[0]['name']}")
                
                if 'args' in re.tool_calls[0]:
                    logger.info(f"Tool args: {re.tool_calls[0]['args']}")
                    
                    args = re.tool_calls[0]['args']
                    if 'code' in args:
                        logger.info(f"code: {args['code']}")
                        if debug_mode == "Enable": 
                            st.code(args['code'])
                    elif re.tool_calls[0]['args']:
                        if debug_mode == "Enable": 
                            st.info(f"Tool name: {re.tool_calls[0]['name']}  \nTool args: {re.tool_calls[0]['args']}")
            # else:
            #     st.info(f"Tool name: {re.tool_calls[0]['name']}")

        elif isinstance(re, ToolMessage):            
            if re.name:
                logger.info(f"Tool name: {re.name}")
                
                if re.content:                
                    content = re.content
                    if len(content) > 500:
                        content = content[:500] + "..."
                    logger.info(f"Tool result: {content}")
                    
                    if debug_mode == "Enable": 
                        st.info(f"Tool name: {re.name}  \nTool result: {content}")                    
                else:
                    if debug_mode == "Enable": 
                        st.info(f"Tool name: {re.name}")
            try: 
                # tavily
                if isinstance(re.content, str) and "Title:" in re.content and "URL:" in re.content and "Content:" in re.content:
                    logger.info("Tavily parsing...")                    
                    items = re.content.split("\n\n")
                    for i, item in enumerate(items):
                        logger.info(f"item[{i}]: {item}")
                        if "Title:" in item and "URL:" in item and "Content:" in item:
                            try:
                                # 정규식 대신 문자열 분할 방법 사용
                                title_part = item.split("Title:")[1].split("URL:")[0].strip()
                                url_part = item.split("URL:")[1].split("Content:")[0].strip()
                                content_part = item.split("Content:")[1].strip()
                                
                                logger.info(f"title_part: {title_part}")
                                logger.info(f"url_part: {url_part}")
                                logger.info(f"content_part: {content_part}")
                                
                                references.append({
                                    "url": url_part,
                                    "title": title_part,
                                    "content": content_part[:100] + "..." if len(content_part) > 100 else content_part
                                })
                            except Exception as e:
                                logger.info(f"파싱 오류: {str(e)}")
                                continue
                
                # check json format
                if isinstance(re.content, str) and (re.content.strip().startswith('{') or re.content.strip().startswith('[')):
                    tool_result = json.loads(re.content)
                    logger.info(f"tool_result: {tool_result}")
                else:
                    tool_result = re.content
                    logger.info(f"tool_result (not JSON): {tool_result}")

                if "path" in tool_result:
                    logger.info(f"Path: {tool_result['path']}")

                    path = tool_result['path']
                    if isinstance(path, list):
                        for p in path:
                            logger.info(f"image: {p}")
                            if p.startswith('http') or p.startswith('https'):
                                st.image(p)
                                image_url.append(p)
                            else:
                                with open(p, 'rb') as f:
                                    image_data = f.read()
                                    st.image(image_data)
                                    image_url.append(p)
                    else:
                        logger.info(f"image: {path}")
                        try:
                            if path.startswith('http') or path.startswith('https'):
                                st.image(path)
                                image_url.append(path)
                            else:
                                with open(path, 'rb') as f:
                                    image_data = f.read()
                                    st.image(image_data)
                                    image_url.append(path)
                        except Exception as e:
                            logger.error(f"이미지 표시 오류: {str(e)}")
                            st.error(f"이미지를 표시할 수 없습니다: {str(e)}")

                # ArXiv
                if "papers" in tool_result:
                    logger.info(f"size of papers: {len(tool_result['papers'])}")

                    papers = tool_result['papers']
                    for paper in papers:
                        url = paper['url']
                        title = paper['title']
                        content = paper['abstract'][:100]
                        logger.info(f"url: {url}, title: {title}, content: {content}")

                        references.append({
                            "url": url,
                            "title": title,
                            "content": content
                        })
                                
                if isinstance(tool_result, list):
                    logger.info(f"size of tool_result: {len(tool_result)}")
                    for i, item in enumerate(tool_result):
                        logger.info(f'item[{i}]: {item}')
                        
                        # RAG
                        if "reference" in item:
                            logger.info(f"reference: {item['reference']}")

                            infos = item['reference']
                            url = infos['url']
                            title = infos['title']
                            source = infos['from']
                            logger.info(f"url: {url}, title: {title}, source: {source}")

                            references.append({
                                "url": url,
                                "title": title,
                                "content": item['contents'][:100]
                            })

                        # Others               
                        if isinstance(item, str):
                            try:
                                item = json.loads(item)

                                # AWS Document
                                if "rank_order" in item:
                                    references.append({
                                        "url": item['url'],
                                        "title": item['title'],
                                        "content": item['context'][:100]
                                    })
                            except json.JSONDecodeError:
                                logger.info(f"JSON parsing error: {item}")
                                continue

            except:
                logger.info(f"fail to parsing..")
                pass
    return image_url, references

def extract_thinking_tag(response, st, debug_mode):
    if response.find('<thinking>') != -1:
        status = response[response.find('<thinking>')+10:response.find('</thinking>')]
        logger.info(f"gent_thinking: {status}")
        
        if debug_mode=="Enable":
            st.info(status)

        if response.find('<thinking>') == 0:
            msg = response[response.find('</thinking>')+12:]
        else:
            msg = response[:response.find('<thinking>')]
        logger.info(f"msg: {msg}")
    else:
        msg = response

    return msg

