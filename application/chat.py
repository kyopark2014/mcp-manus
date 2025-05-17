import utils
import info
import boto3
import traceback
import uuid
import asyncio
import json
import mcp_client
import re
import implementation
import random
import string
import os

from botocore.config import Config
from langchain_aws import ChatBedrock
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_mcp_adapters.client import MultiServerMCPClient
from io import BytesIO
from PIL import Image
from urllib import parse

import logging
import sys

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("chat")

model_name = "Claude 3.5 Sonnet"
model_type = "claude"
models = info.get_model_info(model_name)
number_of_models = len(models)
model_id = models[0]["model_id"]
debug_mode = "Enable"
multi_region = "Disable"

models = info.get_model_info(model_name)
reasoning_mode = 'Disable'
s3_prefix = 'docs'
s3_image_prefix = 'images'

mcp_json = ""

config = utils.load_config()

bedrock_region = config["region"] if "region" in config else "us-west-2"
projectName = config["projectName"] if "projectName" in config else "mcp-rag"
accountId = config["accountId"] if "accountId" in config else None
if accountId is None:
    raise Exception ("No accountId")

region = config["region"] if "region" in config else "us-west-2"
logger.info(f"region: {region}")

path = config["sharing_url"] if "sharing_url" in config else None
if path is None:
    raise Exception ("No Sharing URL")

s3_bucket = config["s3_bucket"] if "s3_bucket" in config else None
if s3_bucket is None:
    raise Exception ("No storage!")

# api key to get weather information in agent
secretsmanager = boto3.client(
    service_name='secretsmanager',
    region_name=bedrock_region
)

# api key to use Tavily Search
tavily_key = tavily_api_wrapper = ""
try:
    get_tavily_api_secret = secretsmanager.get_secret_value(
        SecretId=f"tavilyapikey-{projectName}"
    )
    #print('get_tavily_api_secret: ', get_tavily_api_secret)
    secret = json.loads(get_tavily_api_secret['SecretString'])
    #print('secret: ', secret)

    if "tavily_api_key" in secret:
        tavily_key = secret['tavily_api_key']
        #print('tavily_api_key: ', tavily_api_key)

        if tavily_key:
            tavily_api_wrapper = TavilySearchAPIWrapper(tavily_api_key=tavily_key)
            #     os.environ["TAVILY_API_KEY"] = tavily_key

        else:
            logger.info(f"tavily_key is required.")
except Exception as e: 
    logger.info(f"Tavily credential is required: {e}")
    raise e

MSG_LENGTH = 100  
def save_chat_history(text, msg):
    memory_chain.chat_memory.add_user_message(text)
    if len(msg) > MSG_LENGTH:
        memory_chain.chat_memory.add_ai_message(msg[:MSG_LENGTH])                          
    else:
        memory_chain.chat_memory.add_ai_message(msg) 


def update(modelName, reasoningMode, debugMode, multiRegion, mcp):    
    global model_name, model_id, model_type, debug_mode, multi_region
    global models, mcp_json
    
    if model_name != modelName:
        model_name = modelName
        logger.info(f"model_name: {model_name}")
        
        models = info.get_model_info(model_name)
        model_id = models[0]["model_id"]
        model_type = models[0]["model_type"]
                                
    if debug_mode != debugMode:
        debug_mode = debugMode
        logger.info(f"debug_mode: {debug_mode}")

    if multi_region != multiRegion:
        multi_region = multiRegion
        logger.info(f"multi_region: {multi_region}")

    mcp_json = mcp
    logger.info(f"mcp_json: {mcp_json}")


selected_chat = 0
def get_chat(extended_thinking):
    global selected_chat, model_type

    logger.info(f"models: {models}")
    logger.info(f"selected_chat: {selected_chat}")
    
    profile = models[selected_chat]
    # print('profile: ', profile)
        
    bedrock_region =  profile['bedrock_region']
    modelId = profile['model_id']
    model_type = profile['model_type']
    if model_type == 'claude':
        maxOutputTokens = 4096 # 4k
    else:
        maxOutputTokens = 5120 # 5k
    number_of_models = len(models)

    logger.info(f"LLM: {selected_chat}, bedrock_region: {bedrock_region}, modelId: {modelId}, model_type: {model_type}")

    if profile['model_type'] == 'nova':
        STOP_SEQUENCE = '"\n\n<thinking>", "\n<thinking>", " <thinking>"'
    elif profile['model_type'] == 'claude':
        STOP_SEQUENCE = "\n\nHuman:" 
                          
    # bedrock   
    boto3_bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name=bedrock_region,
        config=Config(
            retries = {
                'max_attempts': 30
            }
        )
    )
    if extended_thinking=='Enable':
        maxReasoningOutputTokens=64000
        logger.info(f"extended_thinking: {extended_thinking}")
        thinking_budget = min(maxOutputTokens, maxReasoningOutputTokens-1000)

        parameters = {
            "max_tokens":maxReasoningOutputTokens,
            "temperature":1,            
            "thinking": {
                "type": "enabled",
                "budget_tokens": thinking_budget
            },
            "stop_sequences": [STOP_SEQUENCE]
        }
    else:
        parameters = {
            "max_tokens":maxOutputTokens,     
            "temperature":0.1,
            "top_k":250,
            "top_p":0.9,
            "stop_sequences": [STOP_SEQUENCE]
        }

    chat = ChatBedrock(   # new chat model
        model_id=modelId,
        client=boto3_bedrock, 
        model_kwargs=parameters,
        region_name=bedrock_region
    )    
    
    if multi_region=='Enable':
        selected_chat = selected_chat + 1
        if selected_chat == number_of_models:
            selected_chat = 0
    else:
        selected_chat = 0

    return chat

map_chain = dict() # general conversation
def initiate():
    global userId
    global memory_chain
    
    userId = uuid.uuid4().hex
    logger.info(f"userId: {userId}")

    if userId in map_chain:  
        memory_chain = map_chain[userId]
    else: 
        memory_chain = ConversationBufferWindowMemory(memory_key="chat_history", output_key='answer', return_messages=True, k=5)
        map_chain[userId] = memory_chain

initiate()

def create_object(key, body):
    """
    Create an object in S3 and return the URL. If the file already exists, append the new content.
    """
    s3_client = boto3.client(
        service_name='s3',
        region_name=bedrock_region
    )
    s3_client.put_object(Bucket=s3_bucket, Key=key, Body=body)
    
def updata_object(key, body):
    """
    Create an object in S3 and return the URL. If the file already exists, append the new content.
    """
    s3_client = boto3.client(
        service_name='s3',
        region_name=bedrock_region
    )
    
    try:
        # Check if file exists
        try:
            response = s3_client.get_object(Bucket=s3_bucket, Key=key)
            existing_body = response['Body'].read().decode('utf-8')
            # Append new content to existing content
            updated_body = existing_body + '\n' + body
        except s3_client.exceptions.NoSuchKey:
            # File doesn't exist, use new body as is
            updated_body = body
            
        # Upload the updated content
        s3_client.put_object(Bucket=s3_bucket, Key=key, Body=updated_body)
        
    except Exception as e:
        logger.error(f"Error updating object in S3: {str(e)}")
        raise e

def get_object(key):
    """
    Get an object from S3 and return the content
    """
    s3_client = boto3.client(
        service_name='s3',
        region_name=bedrock_region
    )
    return s3_client.get_object(Bucket=s3_bucket, Key=key)

def upload_to_s3(file_bytes, key):
    """
    Upload a file to S3 and return the URL
    """
    try:
        s3_client = boto3.client(
            service_name='s3',
            region_name=bedrock_region
        )
        # Generate a unique file name to avoid collisions
        #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        #unique_id = str(uuid.uuid4())[:8]
        #s3_key = f"uploaded_images/{timestamp}_{unique_id}_{key}"

        content_type = utils.get_contents_type(key)       
        logger.info(f"content_type: {content_type}") 

        if content_type == "image/jpeg" or content_type == "image/png":
            s3_key = f"{s3_image_prefix}/{key}"
        else:
            s3_key = f"{s3_prefix}/{key}"
        
        user_meta = {  # user-defined metadata
            "content_type": content_type,
            "model_name": model_name
        }
        
        response = s3_client.put_object(
            Bucket=s3_bucket, 
            Key=s3_key, 
            ContentType=content_type,
            Metadata = user_meta,
            Body=file_bytes            
        )
        logger.info(f"upload response: {response}")

        #url = f"https://{s3_bucket}.s3.amazonaws.com/{s3_key}"
        url = path+'/'+s3_image_prefix+'/'+parse.quote(key)
        return url
    
    except Exception as e:
        err_msg = f"Error uploading to S3: {str(e)}"
        logger.info(f"{err_msg}")
        return None

def extract_and_display_s3_images(text, s3_client):
    """
    Extract S3 URLs from text, download images, and return them for display
    """
    s3_pattern = r"https://[\w\-\.]+\.s3\.amazonaws\.com/[\w\-\./]+"
    s3_urls = re.findall(s3_pattern, text)

    images = []
    for url in s3_urls:
        try:
            bucket = url.split(".s3.amazonaws.com/")[0].split("//")[1]
            key = url.split(".s3.amazonaws.com/")[1]

            response = s3_client.get_object(Bucket=bucket, Key=key)
            image_data = response["Body"].read()

            image = Image.open(BytesIO(image_data))
            images.append(image)

        except Exception as e:
            err_msg = f"Error downloading image from S3: {str(e)}"
            logger.info(f"{err_msg}")
            continue

    return images

####################### LangChain #######################
# General Conversation
#########################################################
def general_conversation(query):
    llm = get_chat(reasoning_mode)

    system = (
        "당신의 이름은 서연이고, 질문에 대해 친절하게 답변하는 사려깊은 인공지능 도우미입니다."
        "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다." 
        "모르는 질문을 받으면 솔직히 모른다고 말합니다."
    )
    
    human = "Question: {input}"
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system), 
        MessagesPlaceholder(variable_name="history"), 
        ("human", human)
    ])
                
    history = memory_chain.load_memory_variables({})["chat_history"]

    try: 
        if reasoning_mode == "Disable":
            chain = prompt | llm | StrOutputParser()
            output = chain.stream(
                {
                    "history": history,
                    "input": query,
                }
            )  
            response = output
        else:
            # output = llm.invoke(query)
            # logger.info(f"output: {output}")
            # response = output.content
            chain = prompt | llm
            output = chain.invoke(
                {
                    "history": history,
                    "input": query,
                }
            )
            logger.info(f"output: {output}")
            response = output
            
    except Exception:
        err_msg = traceback.format_exc()
        logger.info(f"error message: {err_msg}")        
        raise Exception ("Not able to request to LLM: "+err_msg)
        
    return response

#########################################################
# Agent 
#########################################################
import re
from langgraph.prebuilt import ToolNode
from typing_extensions import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from typing import Literal
from langgraph.graph import START, END, StateGraph

def create_agent(tools):
    tool_node = ToolNode(tools)

    chatModel = get_chat(extended_thinking="Disable")
    model = chatModel.bind_tools(tools)

    class State(TypedDict):
        messages: Annotated[list, add_messages]

    def call_model(state: State, config):
        logger.info(f"###### call_model ######")
        # logger.info(f"state: {state['messages']}")

        last_message = state['messages'][-1]
        content = last_message.content.encode().decode('unicode_escape')
        logger.info(f"last message: {content}")

        system = (
            "당신의 이름은 서연이고, 질문에 친근한 방식으로 대답하도록 설계된 대화형 AI입니다."
            "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다."
            "모르는 질문을 받으면 솔직히 모른다고 말합니다."
            "한국어로 답변하세요."
        )

        try:
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            )
            chain = prompt | model
                
            response = chain.invoke(state["messages"])
            # logger.info(f"call_model response: {response}")
            logger.info(f"call_model: {response.content}")

        except Exception:
            response = AIMessage(content="답변을 찾지 못하였습니다.")

            err_msg = traceback.format_exc()
            logger.info(f"error message: {err_msg}")
            # raise Exception ("Not able to request to LLM")

        return {"messages": [response]}

    def should_continue(state: State) -> Literal["continue", "end"]:
        logger.info(f"###### should_continue ######")

        messages = state["messages"]    
        last_message = messages[-1]
        
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            tool_name = last_message.tool_calls[-1]['name']
            logger.info(f"--- CONTINUE: {tool_name} ---")
            return "continue"
        else:
            logger.info(f"--- END ---")
            return "end"

    def buildChatAgent():
        workflow = StateGraph(State)

        workflow.add_node("agent", call_model)
        workflow.add_node("action", tool_node)
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "continue": "action",
                "end": END,
            },
        )
        workflow.add_edge("action", "agent")

        return workflow.compile() 

    app = buildChatAgent()
    config = {
        "recursion_limit": 50
    }

    return app, config


#########################################################
# MCP RAG Agent
#########################################################
def load_multiple_mcp_server_parameters():
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

def tool_info(tools, st):
    tool_info = ""
    tool_list = []
    st.info("Tool 정보를 가져옵니다.")
    for tool in tools:
        tool_info += f"name: {tool.name}\n"    
        if hasattr(tool, 'description'):
            tool_info += f"description: {tool.description}\n"
        tool_info += f"args_schema: {tool.args_schema}\n\n"
        tool_list.append(tool.name)
    # st.info(f"{tool_info}")
    st.info(f"Tools: {tool_list}")

debug_messages = []

def get_debug_messages():
    global debug_messages
    messages = debug_messages.copy()
    debug_messages = []  # Clear messages after returning
    return messages

def push_debug_messages(type, contents):
    global debug_messages
    debug_messages.append({
        type: contents
    })

image_url = []
references = []

def status_messages(message):
    global image_url, references

    # type of message
    if isinstance(message, AIMessage):
        logger.info(f"status_messages (AIMessage): {message}")
    elif isinstance(message, ToolMessage):
        logger.info(f"status_messages (ToolMessage): {message}")
    elif isinstance(message, HumanMessage):
        logger.info(f"status_messages (HumanMessage): {message}")

    if isinstance(message, AIMessage):
        if message.content:
            logger.info(f"content: {message.content}")
            content = message.content
            if len(content) > 500:
                content = content[:500] + "..."       
            push_debug_messages("text", content)
        if hasattr(message, 'tool_calls') and message.tool_calls:
            logger.info(f"Tool name: {message.tool_calls[0]['name']}")
                
            if 'args' in message.tool_calls[0]:
                logger.info(f"Tool args: {message.tool_calls[0]['args']}")
                    
                args = message.tool_calls[0]['args']
                if 'code' in args:
                    logger.info(f"code: {args['code']}")
                    push_debug_messages("text", args['code'])
                elif message.tool_calls[0]['args']:
                    status = f"Tool name: {message.tool_calls[0]['name']}  \nTool args: {message.tool_calls[0]['args']}"
                    logger.info(f"status: {status}")
                    push_debug_messages("text", status)

    elif isinstance(message, ToolMessage):
        if message.name:
            logger.info(f"Tool name: {message.name}")
            
            if message.content:                
                content = message.content
                if len(content) > 500:
                    content = content[:500] + "..."
                logger.info(f"Tool result: {content}")                
                status = f"Tool name: {message.name}  \nTool result: {content}"
            else:
                status = f"Tool name: {message.name}"

            logger.info(f"status: {status}")
            push_debug_messages("text", status)
        try: 
            # Parse Tavily search results
            if isinstance(message.content, str) and "Title:" in message.content and "URL:" in message.content and "Content:" in message.content:
                logger.info("Tavily parsing...")                    
                items = message.content.split("\n\n")
                for i, item in enumerate(items):
                    logger.info(f"item[{i}]: {item}")
                    if "Title:" in item and "URL:" in item and "Content:" in item:
                        try:
                            # Use string splitting instead of regex
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
                            logger.info(f"Parsing error: {str(e)}")
                            continue
            logger.info(f"references: {references}")
            
            # Check JSON format
            if isinstance(message.content, str) and (message.content.strip().startswith('{') or message.content.strip().startswith('[')):
                tool_result = json.loads(message.content)
                logger.info(f"tool_result: {tool_result}")
            else:
                tool_result = message.content
                logger.info(f"tool_result (not JSON): {tool_result}")

            if "path" in tool_result:
                logger.info(f"Path: {tool_result['path']}")

                path = tool_result['path']
                if isinstance(path, list):
                    for p in path:
                        logger.info(f"image: {p}")
                        if p.startswith('http') or p.startswith('https'):
                            push_debug_messages("image", p)
                            image_url.append(p)
                        else:
                            with open(p, 'rb') as f:
                                image_data = f.read()
                                push_debug_messages("image", image_data)
                                image_url.append(p)
                else:
                    logger.info(f"image: {path}")
                    try:
                        if path.startswith('http') or path.startswith('https'):
                            push_debug_messages("image", path)
                            image_url.append(path)
                        else:
                            with open(path, 'rb') as f:
                                image_data = f.read()
                                push_debug_messages("image", image_data)
                                image_url.append(path)
                    except Exception as e:
                        logger.error(f"Image display error: {str(e)}")
                        status = f"Cannot display image: {str(e)}"
                        push_debug_messages("text", image_data)

            # Parse ArXiv papers
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
                    
                    # Parse RAG references
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

                    # Parse other types of references
                    if isinstance(item, str):
                        try:
                            item = json.loads(item)

                            # Parse AWS Document references
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
            logger.info(f"Failed to parse message")
            pass

def extract_thinking_tag(response, st):
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

async def mcp_rag_agent_multiple(query, historyMode, st):
    server_params = load_multiple_mcp_server_parameters()
    logger.info(f"server_params: {server_params}")

    async with MultiServerMCPClient(server_params) as client:
        ref = ""
        with st.status("thinking...", expanded=True, state="running") as status:
            tools = client.get_tools()
            if debug_mode == "Enable":
                tool_info(tools, st)
                logger.info(f"tools: {tools}")

            # react agent
            # model = get_chat(extended_thinking="Disable")
            # agent = create_react_agent(model, client.get_tools())

            # langgraph agent
            agent, config = create_agent(tools)

            try:
                response = await agent.ainvoke({"messages": query}, config)
                logger.info(f"response: {response}")

                result = response["messages"][-1].content
                # logger.info(f"result: {result}")

                debug_msgs = get_debug_messages()
                for msg in debug_msgs:
                    logger.info(f"debug_msg: {msg}")
                    if "image" in msg:
                        st.image(msg["image"])
                    elif "text" in msg:
                        st.info(msg["text"])

                #logger.info(f"references: {references}")
                #image_url, references = show_status_message(response["messages"], st)     
                
                if references:
                    ref = "\n\n### Reference\n"
                for i, reference in enumerate(references):
                    ref += f"{i+1}. [{reference['title']}]({reference['url']}), {reference['content']}...\n"    

                result += ref

                if model_type == "nova":
                    result = extract_thinking_tag(result, st) # for nova

                st.markdown(result)

                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": result,
                    "images": image_url if image_url else []
                })

                return result
            except Exception as e:
                logger.error(f"Error during agent invocation: {str(e)}")
                raise Exception(f"Agent invocation failed: {str(e)}")


def run_mcp_agent(query, historyMode, st):
    result = asyncio.run(mcp_rag_agent_multiple(query, historyMode, st))
    #result = asyncio.run(mcp_rag_agent_single(query, historyMode, st))

    logger.info(f"result: {result}")
    
    return result

#########################################################
# Manus
#########################################################

def get_tool_info(tools, st):    
    toolList = []
    for tool in tools:
        name = tool.name
        toolList.append(name)
    
    toolmsg = ', '.join(toolList)
    st.info(f"Tools: {toolmsg}")

def show_implementation_info(message: str, st):
    st.info(message)
    logger.info(f"Implementation Info: {message}")

async def manus(query, model_type, historyMode, st, mcp_json, debug_mode):
    # implementation 모듈을 함수 내부에서 임포트
    implementation.show_info.callback = lambda msg: show_implementation_info(msg, st)
    
    server_params = load_multiple_mcp_server_parameters()
    logger.info(f"server_params: {server_params}")
    
    async with MultiServerMCPClient(server_params) as client:
        response = ""
        with st.status("thinking...", expanded=True, state="running") as status:            
            tools = client.get_tools()
            # logger.info(f"tools: {tools}")

            if debug_mode == "Enable":
                get_tool_info(tools, st)

            # langgraph agent
            implementation.update_mcp_tools(tools)

            # request id
            request_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
            template = open(os.path.join(os.path.dirname(__file__), f"report.html")).read()
            template = template.replace("{request_id}", request_id)
            key = f"artifacts/{request_id}.html"
            create_object(key, template)

            report_url = path + "/artifacts/" + request_id + ".html"
            logger.info(f"report_url: {report_url}")
            st.info(f"report_url: {report_url}")
                                            
            response = await implementation.run(query, request_id)
            logger.info(f"response: {response}")

            # message queue
            while not implementation.message_queue.empty():
                message = implementation.message_queue.get()
                st.info(message)
                # st.session_state.messages.append({
                #     "role": "assistant", 
                #     "content": message,
                #     "images": []
                # })

            st.markdown(response)

            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "images": image_url if image_url else []
            })

        return response

def run_manus(query, historyMode, st):
    # mcp_client 모듈을 함수 내부에서 임포트
    result = asyncio.run(manus(query, model_type, historyMode, st, mcp_json, debug_mode))
    logger.info(f"result: {result}")
    
    return result