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
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

logger = utils.CreateLogger("chat")

model_name = "Claude 3.5 Sonnet"
model_type = "claude"
models = info.get_model_info(model_name)
number_of_models = len(models)
model_id = models[0]["model_id"]
debug_mode = "Enable"
multi_region = "Disable"

models = info.get_model_info(model_name)
reasoning_mode = 'Disable'

mcp_json = ""

config = utils.load_config()

bedrock_region = config["region"] if "region" in config else "us-west-2"
projectName = config["projectName"] if "projectName" in config else "mcp-rag"
accountId = config["accountId"] if "accountId" in config else None
if accountId is None:
    raise Exception ("No accountId")

region = config["region"] if "region" in config else "us-west-2"
logger.info(f"region: {region}")

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

def show_status_message(response, st):
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

async def mcp_rag_agent_multiple(query, historyMode, st):
    server_params = load_multiple_mcp_server_parameters()
    logger.info(f"server_params: {server_params}")

    async with  MultiServerMCPClient(server_params) as client:
        references = []
        ref = ""
        with st.status("thinking...", expanded=True, state="running") as status:                       
            tools = client.get_tools()
            logger.info(f"tools: {tools}")

            tool_info = []
            for tool in tools:
                name = tool.name
                description = tool.description
                args_schema = tool.args_schema
                tool_info.append(f"name: {name}, description: {description}, args_schema: {args_schema}")

            logger.info(f"tool_info: {tool_info}")

            if debug_mode == "Enable":
                tool_info(tools, st)

            # langgraph agent
            response = manus.run(query)
            logger.info(f"response: {response}")

            result = response["messages"][-1].content
            logger.info(f"result: {result}")

            image_url, references = show_status_message(response["messages"], st)     
            
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

def run_agent(query, historyMode, st):
    result = asyncio.run(mcp_rag_agent_multiple(query, historyMode, st))
    logger.info(f"result: {result}")
    
    return result
