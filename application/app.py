import streamlit as st 
import chat
import utils
import json
import mcp_config 
import os

import logging
import sys

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("streamlit")

# title
st.set_page_config(page_title='Agentic RAG', page_icon=None, layout="centered", initial_sidebar_state="auto", menu_items=None)

mode_descriptions = {
    "일상적인 대화": [
        "대화이력을 바탕으로 챗봇과 일상의 대화를 편안히 즐길수 있습니다."
    ],
    "MANUS": [
        "MANUS Agent를 이용해 문제를 해결합니다."
    ]
}

with st.sidebar:
    st.title("🔮 Menu")
    
    st.markdown(
        "Amazon Bedrock을 이용해 다양한 형태의 대화를 구현합니다." 
        "여기에서는 MCP를 이용해 Manus와 유사한 agent 동작을 구현합니다." 
        "주요 코드는 LangChain과 LangGraph를 이용해 구현되었습니다.\n"
        "상세한 코드는 [Github](https://github.com/kyopark2014/mcp-manus)을 참조하세요."
    )

    st.subheader("🐱 대화 형태")
    
    # radio selection
    mode = st.radio(
        label="원하는 대화 형태를 선택하세요. ",options=["일상적인 대화", "MANUS"], index=1
    )   
    st.info(mode_descriptions[mode][0])    
    # print('mode: ', mode)

    # mcp selection
    mcp = ""
    if mode=='MANUS':
        # MCP Config JSON input
        st.subheader("⚙️ MCP Config")

        # Change radio to checkbox
        mcp_options = ["default", "code interpreter", "aws document", "aws cost", "aws cli", "aws cloudwatch", "aws storage", "image generation", "aws diagram","knowledge base", "tavily", "ArXiv", "wikipedia", "filesystem", "terminal", "puppeteer", "playwright", "firecrawl", "obsidian", "airbnb", "사용자 설정"]
        mcp_selections = {}
        default_selections = ["tavily", "playwright", "code interpreter"]

        with st.expander("MCP 옵션 선택", expanded=True):
            for option in mcp_options:
                default_value = option in default_selections
                mcp_selections[option] = st.checkbox(option, key=f"mcp_{option}", value=default_value)
        
        if not any(mcp_selections.values()):
            mcp_selections["default"] = True

        if mcp_selections["사용자 설정"]:
            mcp_info = st.text_area(
                "MCP 설정을 JSON 형식으로 입력하세요",
                value=mcp,
                height=150
            )
            logger.info(f"mcp_info: {mcp_info}")

            if mcp_info:
                mcp_config.mcp_user_config = json.loads(mcp_info)
                logger.info(f"mcp_user_config: {mcp_config.mcp_user_config}")
        
        mcp = mcp_config.load_selected_config(mcp_selections)
        logger.info(f"mcp: {mcp}")


    # model selection box
    index = 3
    modelName = st.selectbox(
        '🖊️ 사용 모델을 선택하세요',
        ('Nova Pro', 'Nova Lite', 'Claude 3.7 Sonnet', 'Claude 3.5 Sonnet', 'Claude 3.0 Sonnet', 'Claude 3.5 Haiku'), index=index
    )
    
    # extended thinking of claude 3.7 sonnet
    select_reasoning = st.checkbox('Reasonking (only Claude 3.7 Sonnet)', value=False)
    reasoningMode = 'Enable' if select_reasoning and modelName=='Claude 3.7 Sonnet' else 'Disable'
    logger.info(f"reasoningMode: {reasoningMode}")

    chat.update(modelName, reasoningMode, debugMode="Enable", multiRegion="Disable", mcp=mcp)
    
    st.success(f"Connected to {modelName}", icon="💚")
    clear_button = st.button("대화 초기화", key="clear")
    # print('clear_button: ', clear_button)

st.title('🔮 '+ mode)  

if clear_button==True:
    chat.initiate()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.greetings = False

# Display chat messages from history on app rerun
def display_chat_messages():
    """Print message history
    @returns None
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if "images" in message:                
                for url in message["images"]:
                    logger.info(f"url: {url}")

                    file_name = url[url.rfind('/')+1:]
                    st.image(url, caption=file_name, use_container_width=True)
            st.markdown(message["content"])

display_chat_messages()

# Greet user
if not st.session_state.greetings:
    with st.chat_message("assistant"):
        intro = "아마존 베드락을 이용하여 주셔서 감사합니다. 편안한 대화를 즐기실수 있으며, 파일을 업로드하면 요약을 할 수 있습니다."
        st.markdown(intro)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": intro})
        st.session_state.greetings = True

if clear_button or "messages" not in st.session_state:
    st.session_state.messages = []        
    
    st.session_state.greetings = False
    st.rerun()

    chat.clear_chat_history()
            
# Always show the chat input
if prompt := st.chat_input("메시지를 입력하세요."):
    # Delete all_results.txt if exists
    results_file = "./artifacts/all_results.txt"
    if os.path.exists(results_file):
        os.remove(results_file)
        logger.info(f"Deleted existing {results_file}")

    with st.chat_message("user"):  # display user message in chat message container
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})  # add user message to chat history
    prompt = prompt.replace('"', "").replace("'", "")
    logger.info(f"prompt: {prompt}")

    with st.chat_message("assistant"):
        if mode == '일상적인 대화':
            output = chat.general_conversation(prompt)            
            if reasoningMode=="Enable":
                with st.status("thinking...", expanded=True, state="running") as status:    
                    # extended thinking
                    response = output.content
                    st.write(response)
                
            else:
                response = st.write_stream(output)
            
            logger.info(f"response: {response}")
            st.session_state.messages.append({"role": "assistant", "content": response})
            chat.save_chat_history(prompt, response)

        elif mode == 'MANUS':
            with st.status("thinking...", expanded=True, state="running") as status:
                # response = manus.run(prompt)
                response = chat.run_agent(prompt, "Disable", st)
                logger.info(f"response: {response}")

                if response.find('<thinking>') != -1:
                    logger.info(f"Remove <thinking> tag.")
                    response = response[response.find('</thinking>')+12:]
                    logger.info(f"response without tag: {response}")

                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

                chat.save_chat_history(prompt, response)
            
        else:
            stream = chat.general_conversation(prompt)

            response = st.write_stream(stream)
            logger.info(f"response: {response}")

            st.session_state.messages.append({"role": "assistant", "content": response})
            chat.save_chat_history(prompt, response)