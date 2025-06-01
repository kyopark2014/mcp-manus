import logging
import sys
import json
import traceback
import chat

from langgraph.prebuilt import ToolNode
from typing import Literal
from langgraph.graph import START, END, StateGraph
from typing_extensions import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

logging.basicConfig(
    level=logging.INFO,  
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("agent")

status_msg = []
def get_status_msg(status):
    global status_msg
    status_msg.append(status)

    if status != "end":
        status = " -> ".join(status_msg)
        return "[status]\n" + status + "..."
    else: 
        status = " -> ".join(status_msg)
        return "[status]\n" + status

response_msg = []

class State(TypedDict):
    messages: Annotated[list, add_messages]
    image_url: list

async def call_model(state: State, config):
    logger.info(f"###### call_model ######")

    last_message = state['messages'][-1]
    logger.info(f"last message: {last_message}")
    
    image_url = state['image_url'] if 'image_url' in state else []

    status_container = config.get("configurable", {}).get("status_container", None)
    response_container = config.get("configurable", {}).get("response_container", None)
    tools = config.get("configurable", {}).get("tools", None)
    
    if isinstance(last_message, ToolMessage):
        tool_name = last_message.name
        tool_content = last_message.content
        logger.info(f"tool_name: {tool_name}, content: {tool_content[:800]}")

        try:
            json_data = json.loads(tool_content)
            logger.info(f"json_data: {json_data}")
            if isinstance(json_data, dict) and "path" in json_data:
                path = json_data["path"]
                if isinstance(path, list):
                    for url in path:
                        image_url.append(url)
                else:
                    image_url.append(path)

                logger.info(f"image_url: {image_url}")
                if chat.debug_mode == "Enable":
                    response_container.info(f"Added path to image_url: {json_data['path']}")
                    response_msg.append(f"Added path to image_url: {json_data['path']}")

        except json.JSONDecodeError:
            pass

        if chat.debug_mode == "Enable":
            response_container.info(f"{tool_name}: {tool_content[:800]}")
            response_msg.append(f"{tool_name}: {tool_content[:800]}")

    if isinstance(last_message, AIMessage) and last_message.content:
        if chat.debug_mode == "Enable":
            status_container.info(get_status_msg(f"{last_message.name}"))
            response_container.info(f"{last_message.content[:800]}")
            response_msg.append(last_message.content[:800])    
        
    system = (
        "당신의 이름은 서연이고, 질문에 친근한 방식으로 대답하도록 설계된 대화형 AI입니다."
        "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다."
        "모르는 질문을 받으면 솔직히 모른다고 말합니다."
        "한국어로 답변하세요."
    )

    chatModel = chat.get_chat(extended_thinking=chat.reasoning_mode)
    model = chatModel.bind_tools(tools)

    try:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        chain = prompt | model
            
        response = await chain.ainvoke(state["messages"])

    except Exception:
        response = AIMessage(content="답변을 찾지 못하였습니다.")

        err_msg = traceback.format_exc()
        logger.info(f"error message: {err_msg}")

    return {"messages": [response], "image_url": image_url}

async def should_continue(state: State, config) -> Literal["continue", "end"]:
    logger.info(f"###### should_continue ######")

    messages = state["messages"]    
    last_message = messages[-1]

    status_container = config.get("configurable", {}).get("status_container", None)
    response_container = config.get("configurable", {}).get("response_container", None)
    key_container = config.get("configurable", {}).get("key_container", None)
    
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        tool_name = last_message.tool_calls[-1]['name']
        logger.info(f"--- CONTINUE: {tool_name} ---")

        tool_args = last_message.tool_calls[-1]['args']

        if last_message.content:
            logger.info(f"last_message: {last_message.content}")
            if chat.debug_mode == "Enable":
                response_container.info(f"{last_message.content}")
                response_msg.append(last_message.content)

        logger.info(f"tool_name: {tool_name}, tool_args: {tool_args}")
        if chat.debug_mode == "Enable":
            status_container.info(get_status_msg(f"{tool_name}"))
            if "code" in tool_args:
                logger.info(f"code: {tool_args['code']}")
                key_container.code(tool_args['code'])
                response_msg.append(f"{tool_args['code']}")

        return "continue"
    else:
        if chat.debug_mode == "Enable":
            status_container.info(get_status_msg("end"))

        logger.info(f"--- END ---")
        return "end"

def buildChatAgent(tools):
    tool_node = ToolNode(tools)

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

def buildChatAgentWithHistory(tools):
    tool_node = ToolNode(tools)

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

    return workflow.compile(
        checkpointer=chat.checkpointer,
        store=chat.memorystore
    )

def extract_reference(response):
    references = []
    for i, re in enumerate(response):
        if isinstance(re, ToolMessage):
            logger.info(f"###### extract_reference ######")
            try: 
                # tavily
                if isinstance(re.content, str) and "Title:" in re.content and "URL:" in re.content and "Content:" in re.content:
                    logger.info("Tavily parsing...")
                    items = re.content.split("\n\n")
                    for i, item in enumerate(items):
                        # logger.info(f"item[{i}]: {item}")
                        if "Title:" in item and "URL:" in item and "Content:" in item:
                            try:
                                # 정규식 대신 문자열 분할 방법 사용
                                title_part = item.split("Title:")[1].split("URL:")[0].strip()
                                url_part = item.split("URL:")[1].split("Content:")[0].strip()
                                content_part = item.split("Content:")[1].strip().replace("\n", "")
                                
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
                    # logger.info(f"tool_result: {tool_result}")
                else:
                    tool_result = re.content
                    # logger.info(f"tool_result (not JSON): {tool_result[:200]}")

                # ArXiv
                if "papers" in tool_result:
                    logger.info(f"size of papers: {len(tool_result['papers'])}")

                    papers = tool_result['papers']
                    for paper in papers:
                        url = paper['url']
                        title = paper['title']
                        content = paper['abstract'][:100].replace("\n", "")
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
                                "content": item['contents'][:100].replace("\n", "")
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
                                        "content": item['context'][:100].replace("\n", "")
                                    })
                            except json.JSONDecodeError:
                                logger.info(f"JSON parsing error: {item}")
                                continue

            except:
                logger.info(f"fail to parsing..")
                pass
    return references

async def run(question, tools, status_container, response_container, key_container, historyMode):
    global status_msg, response_msg
    status_msg = []
    response_msg = []

    if chat.debug_mode == "Enable":
        status_container.info(get_status_msg("start"))

    if historyMode == "Enable":
        app = buildChatAgentWithHistory(tools)
        config = {
            "recursion_limit": 50,
            "configurable": {"thread_id": chat.userId},
            "status_container": status_container,
            "response_container": response_container,
            "key_container": key_container,
            "tools": tools
        }
    else:
        app = buildChatAgent(tools)
        config = {
            "recursion_limit": 50,
            "status_container": status_container,
            "response_container": response_container,
            "key_container": key_container,
            "tools": tools
        }

    value = None
    inputs = {
        "messages": [HumanMessage(content=question)]
    }

    references = []
    async for output in app.astream(inputs, config):
        for key, value in output.items():
            logger.info(f"--> key: {key}, value: {value}")

            refs = extract_reference(value["messages"])
            if refs:
                for r in refs:
                    references.append(r)
                    logger.info(f"r: {r}")
                
    result = value["messages"][-1].content

    logger.info(f"references: {references}")
    if references:
        ref = "\n\n### Reference\n"
        for i, reference in enumerate(references):
            ref += f"{i+1}. [{reference['title']}]({reference['url']}), {reference['content']}...\n"    
        result += ref

    image_url = value["image_url"] if "image_url" in value else []

    return result, image_url