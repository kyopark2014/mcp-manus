import streamlit as st 
import chat
import json
import knowledge_base as kb
import mcp_config 
import asyncio

import logging
import sys
import agent
import manus.implementation as manus
import os
import pwd

# title
st.set_page_config(page_title='MCP Manus', page_icon=None, layout="centered", initial_sidebar_state="auto", menu_items=None)

# Add CSS style for better checkbox visibility
st.markdown("""
    <style>
    /* Improve checkbox style */
    .stAlert input[type="checkbox"] {
        width: 20px !important;
        height: 20px !important;
        margin: 0 5px 0 0 !important;
        border: 2px solid #666 !important;
        border-radius: 3px !important;
        appearance: none !important;
        -webkit-appearance: none !important;
        background-color: white !important;
    }
    
    .stAlert input[type="checkbox"]:checked {
        background-color: white !important;
        border-color: #666 !important;
        position: relative;
    }
    
    .stAlert input[type="checkbox"]:checked::after {
        content: "✓";
        color: #ff0000;
        position: absolute;
        left: 4px;
        top: -3px;
        font-size: 16px;
        font-weight: bold;
        text-shadow: 1px 1px 1px rgba(0,0,0,0.1);
    }
    
    /* Checkbox label style */
    .stAlert label {
        font-size: 14px !important;
        color: #333 !important;
        margin-left: 5px !important;
        font-weight: 500 !important;
    }
    </style>
    """, unsafe_allow_html=True)

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("streamlit")

try:
    user_info = pwd.getpwuid(os.getuid())
    username = user_info.pw_name
    home_dir = user_info.pw_dir
    logger.info(f"Username: {username}")
    logger.info(f"Home directory: {home_dir}")
except (ImportError, KeyError):
    username = "root"
    logger.info(f"Username: {username}")
    pass  

if username == "root":
    environment = "system"
else:
    environment = "user"
logger.info(f"environment: {environment}")

mode_descriptions = {
    "일상적인 대화": [
        "대화이력을 바탕으로 챗봇과 일상의 대화를 편안히 즐길수 있습니다."
    ],
    "RAG": [
        "Bedrock Knowledge Base를 이용해 구현한 RAG로 필요한 정보를 검색합니다."
    ],
    "Agent": [
        "MCP를 활용한 Agent를 이용합니다. 왼쪽 메뉴에서 필요한 MCP를 선택하세요."
    ],
    "Agent (Chat)": [
        "MCP를 활용한 Agent를 이용합니다. 채팅 히스토리를 이용해 interative한 대화를 즐길 수 있습니다."
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
        label="원하는 대화 형태를 선택하세요. ",options=["일상적인 대화", "RAG", "Agent", "Agent (Chat)", "MANUS"], index=2
    )   
    st.info(mode_descriptions[mode][0])    
    # print('mode: ', mode)

    # mcp selection
    mcp = ""
    if mode=='Agent' or mode=='Agent (Chat)' or mode=='MANUS':
        # MCP Config JSON input
        st.subheader("⚙️ MCP Config")

        # Change radio to checkbox
        if environment == "user":        
            mcp_options = [
                "default", "code interpreter", "aws document", "aws cost", "aws cli", 
                "use_aws","aws cloudwatch", "aws storage", "image generation", "aws diagram",
                "knowledge base", "tavily", "perplexity", "ArXiv", "wikipedia", 
                "filesystem", "terminal", "text editor", "context7", "puppeteer", 
                "playwright", "firecrawl", "obsidian", "airbnb", 
                "pubmed", "chembl", "clinicaltrial", "arxiv-manual", "tavily-manual",
                "사용자 설정"
            ]
        else:
            mcp_options = [ 
                "default", "code interpreter", "aws document", "aws cost", "aws cli", 
                "use_aws", "aws cloudwatch", "aws storage", "image generation", "aws diagram",
                "knowledge base", "tavily", "ArXiv", "wikipedia", "pubmed", 
                "filesystem", "terminal", "text editor", "playwright", "airbnb",
                "pubmed", "chembl", "clinicaltrial", "arxiv-manual", "tavily-manual",
                "사용자 설정"
            ]

        mcp_selections = {}
        default_selections = ["default", "tavily", "use_aws", "code interpreter"]

        with st.expander("MCP 옵션 선택", expanded=True):            
            # 2개의 컬럼 생성
            col1, col2 = st.columns(2)
            
            # 옵션을 두 그룹으로 나누기
            mid_point = len(mcp_options) // 2
            first_half = mcp_options[:mid_point]
            second_half = mcp_options[mid_point:]
            
            # 첫 번째 컬럼에 첫 번째 그룹 표시
            with col1:
                for option in first_half:
                    default_value = option in default_selections
                    mcp_selections[option] = st.checkbox(option, key=f"mcp_{option}", value=default_value)
            
            # 두 번째 컬럼에 두 번째 그룹 표시
            with col2:
                for option in second_half:
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
    
    # debug checkbox
    select_debugMode = st.checkbox('Debug Mode', value=True)
    debugMode = 'Enable' if select_debugMode else 'Disable'
    #print('debugMode: ', debugMode)

    # multi region check box
    select_multiRegion = st.checkbox('Multi Region', value=False)
    multiRegion = 'Enable' if select_multiRegion else 'Disable'
    #print('multiRegion: ', multiRegion)

    # RAG grading
    select_grading = st.checkbox('Grading', value=False)
    gradingMode = 'Enable' if select_grading else 'Disable'
    # logger.info(f"gradingMode: {gradingMode}")

    uploaded_file = None
    if mode=='RAG' or mode=="Agent" or mode=="Agent (Chat)":
        st.subheader("📋 문서 업로드")
        # print('fileId: ', chat.fileId)
        uploaded_file = st.file_uploader("RAG를 위한 파일을 선택합니다.", type=["pdf", "txt", "py", "md", "csv", "json"], key=chat.fileId)

    # extended thinking 
    select_reasoning = st.checkbox('Reasoning', value=False)
    reasoningMode = 'Enable' if select_reasoning else 'Disable'
    logger.info(f"reasoningMode: {reasoningMode}")

    chat.update(modelName, reasoningMode, debugMode, multiRegion, gradingMode, mcp=mcp)
    
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

def show_references(reference_docs):
    if debugMode == "Enable" and reference_docs:
        with st.expander(f"답변에서 참조한 {len(reference_docs)}개의 문서입니다."):
            for i, doc in enumerate(reference_docs):
                st.markdown(f"**{doc.metadata['name']}**: {doc.page_content}")
                st.markdown("---")

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
    uploaded_file = None
    
    st.session_state.greetings = False
    chat.clear_chat_history()
    st.rerun()

# Preview the uploaded image in the sidebar
file_name = ""
state_of_code_interpreter = False
if uploaded_file is not None and clear_button==False:
    logger.info(f"uploaded_file.name: {uploaded_file.name}")
    if uploaded_file.name:
        logger.info(f"csv type? {uploaded_file.name.lower().endswith(('.csv'))}")

    if uploaded_file.name and not mode == '이미지 분석':
        chat.initiate()

        if debugMode=='Enable':
            status = '선택한 파일을 업로드합니다.'
            logger.info(f"status: {status}")
            st.info(status)

        file_name = uploaded_file.name
        logger.info(f"uploading... file_name: {file_name}")
        file_url = chat.upload_to_s3(uploaded_file.getvalue(), file_name)
        logger.info(f"file_url: {file_url}")

        kb.sync_data_source()  # sync uploaded files
            
        status = f'선택한 "{file_name}"의 내용을 요약합니다.'
        # my_bar = st.sidebar.progress(0, text=status)
        
        # for percent_complete in range(100):
        #     time.sleep(0.2)
        #     my_bar.progress(percent_complete + 1, text=status)
        if debugMode=='Enable':
            logger.info(f"status: {status}")
            st.info(status)
    
        msg = chat.get_summary_of_uploaded_file(file_name, st)
        st.session_state.messages.append({"role": "assistant", "content": f"선택한 문서({file_name})를 요약하면 아래와 같습니다.\n\n{msg}"})    
        logger.info(f"msg: {msg}")

        st.write(msg)

    if uploaded_file and clear_button==False and mode == '이미지 분석':
        st.image(uploaded_file, caption="이미지 미리보기", use_container_width=True)

        file_name = uploaded_file.name
        url = chat.upload_to_s3(uploaded_file.getvalue(), file_name)
        logger.info(f"url: {url}")

# Always show the chat input
if prompt := st.chat_input("메시지를 입력하세요."):
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

        elif mode == 'RAG':
            with st.status("running...", expanded=True, state="running") as status:
                response, reference_docs = chat.run_rag_with_knowledge_base(prompt, st)                           
                st.write(response)
                logger.info(f"response: {response}")

                st.session_state.messages.append({"role": "assistant", "content": response})

                chat.save_chat_history(prompt, response)
            
            show_references(reference_docs) 

        elif mode == 'Agent' or mode == 'Agent (Chat)':
            sessionState = ""

            if mode == 'Agent':
                history_mode = "Disable"
            else:
                history_mode = "Enable"

            with st.status("thinking...", expanded=True, state="running") as status:
                containers = {
                    "tools": st.empty(),
                    "status": st.empty(),
                    "notification": [st.empty() for _ in range(100)]
                }
                response, image_url = asyncio.run(agent.run_agent(prompt, history_mode, containers))
            
            if agent.response_msg:
                with st.expander(f"수행 결과"):
                    st.markdown('\n\n'.join(agent.response_msg))

            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "images": image_url if image_url else []
            })

            st.write(response)
            for url in image_url:
                    logger.info(f"url: {url}")
                    file_name = url[url.rfind('/')+1:]
                    st.image(url, caption=file_name, use_container_width=True)

        elif mode == 'MANUS':
            sessionState = ""
            chat.references = []
            chat.image_url = []

            response, image_url, urls = asyncio.run(manus.run_manus(prompt, "Enable", st))
            logger.info(f"response: {response}")

            st.markdown(response)            
            
            for url in image_url:
                logger.info(f"url: {url}")
                file_name = url[url.rfind('/')+1:]
                st.image(url, caption=file_name, use_container_width=True)           

            if urls:
                with st.expander(f"최종 결과"):
                    url_msg = '\n\n'.join(urls)
                    st.markdown(url_msg)

            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "images": image_url if image_url else []
            })
            
        else:
            stream = chat.general_conversation(prompt)

            response = st.write_stream(stream)
            logger.info(f"response: {response}")

            st.session_state.messages.append({"role": "assistant", "content": response})
            chat.save_chat_history(prompt, response)