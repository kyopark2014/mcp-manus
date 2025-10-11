import utils
import info
import boto3
import traceback
import uuid
import json
import re
import os
import PyPDF2
import csv
import base64
import re

from botocore.config import Config
from langchain_aws import ChatBedrock
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from io import BytesIO
from PIL import Image
from urllib import parse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

# Simple memory class to replace ConversationBufferWindowMemory
class SimpleMemory:
    def __init__(self, k=5):
        self.k = k
        self.chat_memory = SimpleChatMemory()
    
    def load_memory_variables(self, inputs):
        return {"chat_history": self.chat_memory.messages[-self.k:] if len(self.chat_memory.messages) > self.k else self.chat_memory.messages}

class SimpleChatMemory:
    def __init__(self):
        self.messages = []
    
    def add_user_message(self, message):
        self.messages.append(HumanMessage(content=message))
    
    def add_ai_message(self, message):
        self.messages.append(AIMessage(content=message))
    
    def clear(self):
        self.messages = []

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

s3_prefix = 'docs'
s3_image_prefix = 'images'

doc_prefix = s3_prefix+'/'

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

knowledge_base_role = config["knowledge_base_role"] if "knowledge_base_role" in config else None
if knowledge_base_role is None:
    raise Exception ("No Knowledge Base Role")

collectionArn = config["collectionArn"] if "collectionArn" in config else None
if collectionArn is None:
    raise Exception ("No collectionArn")

vectorIndexName = projectName

opensearch_url = config["opensearch_url"] if "opensearch_url" in config else None
if opensearch_url is None:
    raise Exception ("No OpenSearch URL")

path = config["sharing_url"] if "sharing_url" in config else None
if path is None:
    raise Exception ("No Sharing URL")

s3_arn = config["s3_arn"] if "s3_arn" in config else None
if s3_arn is None:
    raise Exception ("No S3 ARN")

s3_bucket = config["s3_bucket"] if "s3_bucket" in config else None
if s3_bucket is None:
    raise Exception ("No storage!")

knowledge_base_name = projectName
numberOfDocs = 4

MSG_LENGTH = 100    

def save_chat_history(text, msg):
    global memory_chain
    # Initialize memory_chain if it doesn't exist
    if memory_chain is None:
        initiate()
    
    if memory_chain and hasattr(memory_chain, 'chat_memory'):
        memory_chain.chat_memory.add_user_message(text)
        if len(msg) > MSG_LENGTH:
            memory_chain.chat_memory.add_ai_message(msg[:MSG_LENGTH])                          
        else:
            memory_chain.chat_memory.add_ai_message(msg) 

grading_mode = 'Disable'
def update(modelName, reasoningMode, debugMode, multiRegion, gradingMode, mcp):    
    global model_name, model_id, model_type, debug_mode, multi_region
    global models, mcp_json, grading_mode

    # load mcp.env    
    mcp_env = utils.load_mcp_env()
    logger.info(f"mcp_env: {mcp_env}")
    if not mcp_env:
        mcp_env = {
            'multi_region': 'Disable',
            'grading_mode': 'Disable'
        }
    
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
        mcp_env['multi_region'] = multi_region

    if grading_mode != gradingMode:
        grading_mode = gradingMode
        logger.info(f"grading_mode: {grading_mode}")
        mcp_env['grading_mode'] = grading_mode

    mcp_json = mcp
    logger.info(f"mcp_json: {mcp_json}")

    # update mcp.env    
    utils.save_mcp_env(mcp_env)
    logger.info(f"mcp.env updated: {mcp_env}")

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

userId = uuid.uuid4().hex
map_chain = dict() 

checkpointers = dict() 
memorystores = dict() 

checkpointer = MemorySaver()
memorystore = InMemoryStore()
memory_chain = None  # Initialize memory_chain as global variable

def clear_chat_history():
    global memory_chain
    # Initialize memory_chain if it doesn't exist
    if memory_chain is None:
        initiate()
    
    if memory_chain and hasattr(memory_chain, 'chat_memory'):
        memory_chain.chat_memory.clear()
    else:
        memory_chain = SimpleMemory(k=5)
    map_chain[userId] = memory_chain

map_chain = dict() # general conversation
def initiate():
    global userId
    global memory_chain, checkpointer, memorystore, checkpointers, memorystores
    
    userId = uuid.uuid4().hex
    logger.info(f"userId: {userId}")

    if userId in map_chain:  
        logger.info(f"memory exist. reuse it!")
        memory_chain = map_chain[userId]

        checkpointer = checkpointers[userId]
        memorystore = memorystores[userId]
    else: 
        logger.info(f"memory not exist. create new memory!")
        memory_chain = SimpleMemory(k=5)
        map_chain[userId] = memory_chain

        checkpointer = MemorySaver()
        memorystore = InMemoryStore()

        checkpointers[userId] = checkpointer
        memorystores[userId] = memorystore        

initiate()


def isKorean(text):
    # check korean
    pattern_hangul = re.compile('[\u3131-\u3163\uac00-\ud7a3]+')
    word_kor = pattern_hangul.search(str(text))
    # print('word_kor: ', word_kor)

    if word_kor and word_kor != 'None':
        # logger.info(f"Korean: {word_kor}")
        return True
    else:
        # logger.info(f"Not Korean:: {word_kor}")
        return False
    
# load csv documents from s3
def load_csv_document(s3_file_name):
    s3r = boto3.resource("s3")
    doc = s3r.Object(s3_bucket, s3_prefix+'/'+s3_file_name)

    lines = doc.get()['Body'].read().decode('utf-8').split('\n')   # read csv per line
    logger.info(f"prelinspare: {len(lines)}")
        
    columns = lines[0].split(',')  # get columns
    #columns = ["Category", "Information"]  
    #columns_to_metadata = ["type","Source"]
    logger.info(f"columns: {columns}")
    
    docs = []
    n = 0
    for row in csv.DictReader(lines, delimiter=',',quotechar='"'):
        # print('row: ', row)
        #to_metadata = {col: row[col] for col in columns_to_metadata if col in row}
        values = {k: row[k] for k in columns if k in row}
        content = "\n".join(f"{k.strip()}: {v.strip()}" for k, v in values.items())
        doc = Document(
            page_content=content,
            metadata={
                'name': s3_file_name,
                'row': n+1,
            }
            #metadata=to_metadata
        )
        docs.append(doc)
        n = n+1
    logger.info(f"docs[0]: {docs[0]}")

    return docs

def get_summary(docs):    
    llm = get_chat(extended_thinking="Disable")

    text = ""
    for doc in docs:
        text = text + doc
    
    if isKorean(text)==True:
        system = (
            "다음의 <article> tag안의 문장을 요약해서 500자 이내로 설명하세오."
        )
    else: 
        system = (
            "Here is pieces of article, contained in <article> tags. Write a concise summary within 500 characters."
        )
    
    human = "<article>{text}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    # print('prompt: ', prompt)
    
    chain = prompt | llm    
    try: 
        result = chain.invoke(
            {
                "text": text
            }
        )
        
        summary = result.content
        logger.info(f"esult of summarization: {summary}")
    except Exception:
        err_msg = traceback.format_exc()
        logger.info(f"error message: {err_msg}") 
        raise Exception ("Not able to request to LLM")
    
    return summary

# load documents from s3 for pdf and txt
def load_document(file_type, s3_file_name):
    s3r = boto3.resource("s3")
    doc = s3r.Object(s3_bucket, s3_prefix+'/'+s3_file_name)
    logger.info(f"s3_bucket: {s3_bucket}, s3_prefix: {s3_prefix}, s3_file_name: {s3_file_name}")
    
    contents = ""
    if file_type == 'pdf':
        contents = doc.get()['Body'].read()
        reader = PyPDF2.PdfReader(BytesIO(contents))
        
        raw_text = []
        for page in reader.pages:
            raw_text.append(page.extract_text())
        contents = '\n'.join(raw_text)    
        
    elif file_type == 'txt' or file_type == 'md':        
        contents = doc.get()['Body'].read().decode('utf-8')
        
    logger.info(f"contents: {contents}")
    new_contents = str(contents).replace("\n"," ") 
    logger.info(f"length: {len(new_contents)}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function = len,
    ) 
    texts = text_splitter.split_text(new_contents) 
    if texts:
        logger.info(f"exts[0]: {texts[0]}")
    
    return texts

def summary_of_code(code, mode):
    if mode == 'py':
        system = (
            "다음의 <article> tag에는 python code가 있습니다."
            "code의 전반적인 목적에 대해 설명하고, 각 함수의 기능과 역할을 자세하게 한국어 500자 이내로 설명하세요."
        )
    elif mode == 'js':
        system = (
            "다음의 <article> tag에는 node.js code가 있습니다." 
            "code의 전반적인 목적에 대해 설명하고, 각 함수의 기능과 역할을 자세하게 한국어 500자 이내로 설명하세요."
        )
    else:
        system = (
            "다음의 <article> tag에는 code가 있습니다."
            "code의 전반적인 목적에 대해 설명하고, 각 함수의 기능과 역할을 자세하게 한국어 500자 이내로 설명하세요."
        )
    
    human = "<article>{code}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    # print('prompt: ', prompt)
    
    llm = get_chat(extended_thinking="Disable")

    chain = prompt | llm    
    try: 
        result = chain.invoke(
            {
                "code": code
            }
        )
        
        summary = result.content
        logger.info(f"result of code summarization: {summary}")
    except Exception:
        err_msg = traceback.format_exc()
        logger.info(f"error message: {err_msg}")        
        raise Exception ("Not able to request to LLM")
    
    return summary

def summary_image(img_base64, instruction):      
    llm = get_chat(extended_thinking="Disable")

    if instruction:
        logger.info(f"instruction: {instruction}")
        query = f"{instruction}. <result> tag를 붙여주세요."
        
    else:
        query = "이미지가 의미하는 내용을 풀어서 자세히 알려주세요. markdown 포맷으로 답변을 작성합니다."
    
    messages = [
        HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}", 
                    },
                },
                {
                    "type": "text", "text": query
                },
            ]
        )
    ]
    
    for attempt in range(5):
        logger.info(f"attempt: {attempt}")
        try: 
            result = llm.invoke(messages)
            
            extracted_text = result.content
            # print('summary from an image: ', extracted_text)
            break
        except Exception:
            err_msg = traceback.format_exc()
            logger.info(f"error message: {err_msg}")                    
            raise Exception ("Not able to request to LLM")
        
    return extracted_text

def traslation(chat, text, input_language, output_language):
    system = (
        "You are a helpful assistant that translates {input_language} to {output_language} in <article> tags." 
        "Put it in <result> tags."
    )
    human = "<article>{text}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    # print('prompt: ', prompt)
    
    chain = prompt | chat    
    try: 
        result = chain.invoke(
            {
                "input_language": input_language,
                "output_language": output_language,
                "text": text,
            }
        )
        
        msg = result.content
        # print('translated text: ', msg)
    except Exception:
        err_msg = traceback.format_exc()
        logger.info(f"error message: {err_msg}")     
        raise Exception ("Not able to request to LLM")

    return msg[msg.find('<result>')+8:len(msg)-9] # remove <result> tag

def extract_text(img_base64):    
    multimodal = get_chat(extended_thinking="Disable")
    query = "텍스트를 추출해서 markdown 포맷으로 변환하세요. <result> tag를 붙여주세요."
    
    extracted_text = ""
    messages = [
        HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}", 
                    },
                },
                {
                    "type": "text", "text": query
                },
            ]
        )
    ]
    
    for attempt in range(5):
        logger.info(f"attempt: {attempt}")
        try: 
            result = multimodal.invoke(messages)
            
            extracted_text = result.content
            # print('result of text extraction from an image: ', extracted_text)
            break
        except Exception:
            err_msg = traceback.format_exc()
            logger.info(f"error message: {err_msg}")                    
            # raise Exception ("Not able to request to LLM")
    
    logger.info(f"xtracted_text: {extracted_text}")
    if len(extracted_text)<10:
        extracted_text = "텍스트를 추출하지 못하였습니다."    

    return extracted_text

fileId = uuid.uuid4().hex
# print('fileId: ', fileId)
def get_summary_of_uploaded_file(file_name, st):
    file_type = file_name[file_name.rfind('.')+1:len(file_name)]            
    logger.info(f"file_type: {file_type}")
    
    if file_type == 'csv':
        docs = load_csv_document(file_name)
        contexts = []
        for doc in docs:
            contexts.append(doc.page_content)
        logger.info(f"contexts: {contexts}")
    
        msg = get_summary(contexts)

    elif file_type == 'pdf' or file_type == 'txt' or file_type == 'md' or file_type == 'pptx' or file_type == 'docx':
        texts = load_document(file_type, file_name)

        if len(texts):
            docs = []
            for i in range(len(texts)):
                docs.append(
                    Document(
                        page_content=texts[i],
                        metadata={
                            'name': file_name,
                            # 'page':i+1,
                            'url': path+'/'+doc_prefix+parse.quote(file_name)
                        }
                    )
                )
            logger.info(f"docs[0]: {docs[0]}") 
            logger.info(f"docs size: {len(docs)}")

            contexts = []
            for doc in docs:
                contexts.append(doc.page_content)
            logger.info(f"contexts: {contexts}")

            msg = get_summary(contexts)
        else:
            msg = "문서 로딩에 실패하였습니다."
        
    elif file_type == 'py' or file_type == 'js':
        s3r = boto3.resource("s3")
        doc = s3r.Object(s3_bucket, s3_prefix+'/'+file_name)
        
        contents = doc.get()['Body'].read().decode('utf-8')
        
        msg = summary_of_code(contents, file_type)                  
        
    elif file_type == 'png' or file_type == 'jpeg' or file_type == 'jpg':
        logger.info(f"multimodal: {file_name}")
        
        s3_client = boto3.client(
            service_name='s3',
            region_name=bedrock_region
        )             
        if debug_mode=="Enable":
            status = "이미지를 가져옵니다."
            logger.info(f"status: {status}")
            st.info(status)
            
        image_obj = s3_client.get_object(Bucket=s3_bucket, Key=s3_prefix+'/'+file_name)
        
        image_content = image_obj['Body'].read()
        img = Image.open(BytesIO(image_content))
        
        width, height = img.size 
        logger.info(f"width: {width}, height: {height}, size: {width*height}")
        
        isResized = False
        while(width*height > 5242880):                    
            width = int(width/2)
            height = int(height/2)
            isResized = True
            logger.info(f"width: {width}, height: {height}, size: {width*height}")
        
        if isResized:
            img = img.resize((width, height))
        
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
               
        # extract text from the image
        if debug_mode=="Enable":
            status = "이미지에서 텍스트를 추출합니다."
            logger.info(f"status: {status}")
            st.info(status)
        
        text = extract_text(img_base64)
        # print('extracted text: ', text)

        if text.find('<result>') != -1:
            extracted_text = text[text.find('<result>')+8:text.find('</result>')] # remove <result> tag
            # print('extracted_text: ', extracted_text)
        else:
            extracted_text = text

        if debug_mode=="Enable":
            logger.info(f"### 추출된 텍스트\n\n{extracted_text}")
            print('status: ', status)
            st.info(status)
    
        if debug_mode=="Enable":
            status = "이미지의 내용을 분석합니다."
            logger.info(f"status: {status}")
            st.info(status)

        image_summary = summary_image(img_base64, "")
        logger.info(f"image summary: {image_summary}")
            
        if len(extracted_text) > 10:
            contents = f"## 이미지 분석\n\n{image_summary}\n\n## 추출된 텍스트\n\n{extracted_text}"
        else:
            contents = f"## 이미지 분석\n\n{image_summary}"
        logger.info(f"image content: {contents}")

        msg = contents

    global fileId
    fileId = uuid.uuid4().hex

    return msg

def create_object(key, body):
    """
    Create an object in S3 and return the URL. If the file already exists, append the new content.
    """
    s3_client = boto3.client(
        service_name='s3',
        region_name=bedrock_region
    )
    
    # Content-Type based on file extension
    content_type = 'application/octet-stream'  # default value
    if key.endswith('.html'):
        content_type = 'text/html'
    elif key.endswith('.md'):
        content_type = 'text/markdown'
    
    s3_client.put_object(
        Bucket=s3_bucket,
        Key=key,
        Body=body,
        ContentType=content_type
    )
    
def updata_object(key, body, direction):
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

            if direction == 'append':
                updated_body = existing_body + '\n' + body
            else: # prepend
                updated_body = body + '\n' + existing_body
        except s3_client.exceptions.NoSuchKey:
            # File doesn't exist, use new body as is
            updated_body = body
            
        # Content-Type based on file extension
        content_type = 'application/octet-stream'  # default value
        if key.endswith('.html'):
            content_type = 'text/html'
        elif key.endswith('.md'):
            content_type = 'text/markdown'
            
        # Upload the updated content
        s3_client.put_object(
            Bucket=s3_bucket,
            Key=key,
            Body=updated_body,
            ContentType=content_type
        )
        
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
    response = s3_client.get_object(Bucket=s3_bucket, Key=key)
    return response['Body'].read().decode('utf-8')

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

def upload_to_s3_artifacts(file_bytes, file_name):
    """
    Upload a file to S3 and return the URL
    """
    try:
        s3_client = boto3.client(
            service_name='s3',
            region_name=bedrock_region
        )

        content_type = utils.get_contents_type(file_name)       
        logger.info(f"content_type: {content_type}") 

        s3_key = f"artifacts/{file_name}"
        
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

        url = path+'/artifacts/'+parse.quote(file_name)
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
    global memory_chain
    initiate()  # Initialize memory_chain
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
                
    if memory_chain and hasattr(memory_chain, 'load_memory_variables'):
        history = memory_chain.load_memory_variables({})["chat_history"]
    else:
        history = []

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

####################### Bedrock Agent #######################
# RAG using Lambda
############################################################# 
def get_rag_prompt(text):
    # print("###### get_rag_prompt ######")
    llm = get_chat(extended_thinking="Disable")
    # print('model_type: ', model_type)
    
    if model_type == "nova":
        if isKorean(text)==True:
            system = (
                "당신의 이름은 서연이고, 질문에 대해 친절하게 답변하는 사려깊은 인공지능 도우미입니다."
                "다음의 Reference texts을 이용하여 user의 질문에 답변합니다."
                "모르는 질문을 받으면 솔직히 모른다고 말합니다."
                "답변의 이유를 풀어서 명확하게 설명합니다."
            )
        else: 
            system = (
                "You will be acting as a thoughtful advisor."
                "Provide a concise answer to the question at the end using reference texts." 
                "If you don't know the answer, just say that you don't know, don't try to make up an answer."
                "You will only answer in text format, using markdown format is not allowed."
            )    
    
        human = (
            "Question: {question}"

            "Reference texts: "
            "{context}"
        ) 
        
    elif model_type == "claude":
        if isKorean(text)==True:
            system = (
                "당신의 이름은 서연이고, 질문에 대해 친절하게 답변하는 사려깊은 인공지능 도우미입니다."
                "다음의 <context> tag안의 참고자료를 이용하여 상황에 맞는 구체적인 세부 정보를 충분히 제공합니다." 
                "모르는 질문을 받으면 솔직히 모른다고 말합니다."
                "답변의 이유를 풀어서 명확하게 설명합니다."
                "결과는 <result> tag를 붙여주세요."
            )
        else: 
            system = (
                "You will be acting as a thoughtful advisor."
                "Here is pieces of context, contained in <context> tags." 
                "If you don't know the answer, just say that you don't know, don't try to make up an answer."
                "You will only answer in text format, using markdown format is not allowed."
                "Put it in <result> tags."
            )    

        human = (
            "<question>"
            "{question}"
            "</question>"

            "<context>"
            "{context}"
            "</context>"
        )

    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    # print('prompt: ', prompt)
    
    rag_chain = prompt | llm

    return rag_chain
 
def retrieve_knowledge_base(query):
    lambda_client = boto3.client(
        service_name='lambda',
        region_name=bedrock_region
    )

    functionName = f"lambda-rag-for-{projectName}"
    logger.info(f"functionName: {functionName}")

    try:
        payload = {
            'function': 'search_rag',
            'knowledge_base_name': knowledge_base_name,
            'keyword': query,
            'top_k': numberOfDocs,
            'grading': "Enable",
            'model_name': model_name,
            'multi_region': multi_region
        }
        logger.info(f"payload: {payload}")

        output = lambda_client.invoke(
            FunctionName=functionName,
            Payload=json.dumps(payload),
        )
        payload = json.load(output['Payload'])
        logger.info(f"response: {payload['response']}")
        
    except Exception:
        err_msg = traceback.format_exc()
        logger.info(f"error message: {err_msg}")       

    return payload['response']

def get_reference_docs(docs):    
    reference_docs = []
    for doc in docs:
        reference = doc.get("reference")
        reference_docs.append(
            Document(
                page_content=doc.get("contents"),
                metadata={
                    'name': reference.get("title"),
                    'url': reference.get("url"),
                    'from': reference.get("from")
                },
            )
    )     
    return reference_docs

def run_rag_with_knowledge_base(query, st):
    global reference_docs, contentList
    reference_docs = []
    contentList = []

    # retrieve
    if debug_mode == "Enable":
        st.info(f"RAG 검색을 수행합니다. 검색어: {query}")  

    relevant_context = retrieve_knowledge_base(query)    
    logger.info(f"relevant_context: {relevant_context}")
    
    # change format to document
    reference_docs = get_reference_docs(json.loads(relevant_context))
    st.info(f"{len(reference_docs)}개의 관련된 문서를 얻었습니다.")

    rag_chain = get_rag_prompt(query)
                       
    msg = ""    
    try: 
        result = rag_chain.invoke(
            {
                "question": query,
                "context": relevant_context                
            }
        )
        logger.info(f"result: {result}")

        msg = result.content        
        if msg.find('<result>')!=-1:
            msg = msg[msg.find('<result>')+8:msg.find('</result>')]        
               
    except Exception:
        err_msg = traceback.format_exc()
        logger.info(f"error message: {err_msg}")                    
        raise Exception ("Not able to request to LLM")
    
    if reference_docs:
        logger.info(f"reference_docs: {reference_docs}")
        ref = "\n\n### Reference\n"
        for i, reference in enumerate(reference_docs):
            ref += f"{i+1}. [{reference.metadata['name']}]({reference.metadata['url']}), {reference.page_content[:100]}...\n"    
        logger.info(f"ref: {ref}")
        msg += ref
    
    return msg, reference_docs
