import utils
import info
import boto3
import traceback
import uuid
import asyncio
import json
import mcp_client
import re

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

def upload_to_s3(file_bytes, file_name):
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
        #s3_key = f"uploaded_images/{timestamp}_{unique_id}_{file_name}"

        content_type = utils.get_contents_type(file_name)       
        logger.info(f"content_type: {content_type}") 

        if content_type == "image/jpeg" or content_type == "image/png":
            s3_key = f"{s3_image_prefix}/{file_name}"
        else:
            s3_key = f"{s3_prefix}/{file_name}"
        
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
        url = path+'/'+s3_image_prefix+'/'+parse.quote(file_name)
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

def run_agent(query, historyMode, st):
    # mcp_client 모듈을 함수 내부에서 임포트
    import mcp_client
    result = asyncio.run(mcp_agent(query, model_type, historyMode, st, mcp_json, debug_mode))
    logger.info(f"result: {result}")
    
    return result

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

async def mcp_agent(query, model_type, historyMode, st, mcp_json, debug_mode):
    # implementation 모듈을 함수 내부에서 임포트
    import implementation
    implementation.show_info.callback = lambda msg: show_implementation_info(msg, st)
    
    server_params = mcp_client.load_multiple_mcp_server_parameters(mcp_json)
    logger.info(f"server_params: {server_params}")

    response = ""
    async with MultiServerMCPClient(server_params) as client:
        tools = client.get_tools()
        # logger.info(f"tools: {tools}")

        if debug_mode == "Enable":
            get_tool_info(tools, st)

        # langgraph agent
        implementation.update_team_members(tools)
                            
        response = await implementation.run(query)
        logger.info(f"response: {response}")

        # 메시지 큐 처리
        while not implementation.message_queue.empty():
            message = implementation.message_queue.get()
            st.info(message)
            st.session_state.messages.append({
                "role": "assistant", 
                "content": message,
                "images": []
            })

    return response

####################### Agent #######################
# Agent 
#####################################################
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
        logger.info(f"last message: {last_message}")

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
