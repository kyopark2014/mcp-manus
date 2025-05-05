import json
import boto3
import logging
import sys
import base64
import chat
import pandas as pd
import plotly.express as px
import plotly.io as pio
import random
import traceback

from typing import Dict, Optional, Any
from datetime import datetime, timedelta
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("mcp-cost")

cost_data = {}
def get_cost_analysis(days: int=30, region: str="us-west-2"):
    """
    Cost analysis data collection
    Parameters:
        days: the period of the data, e.g., 30
        region: The region of aws infrastructure, e.g., us-west-2
    """   
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # cost explorer
        ce = boto3.client(
            service_name='ce',
            region_name=region
        )

        # service cost
        service_response = ce.get_cost_and_usage(
            TimePeriod={
                'Start': start_date.strftime('%Y-%m-%d'),
                'End': end_date.strftime('%Y-%m-%d')
            },
            Granularity='MONTHLY',
            Metrics=['UnblendedCost'],
            GroupBy=[{'Type': 'DIMENSION', 'Key': 'SERVICE'}]
        )
        
        service_costs = pd.DataFrame([
            {
                'SERVICE': group['Keys'][0],
                'cost': float(group['Metrics']['UnblendedCost']['Amount'])
            }
            for group in service_response['ResultsByTime'][0]['Groups']
        ])
        logger.info(f"Service Cost: {service_response}")
        
        # region cost
        region_response = ce.get_cost_and_usage(
            TimePeriod={
                'Start': start_date.strftime('%Y-%m-%d'),
                'End': end_date.strftime('%Y-%m-%d')
            },
            Granularity='MONTHLY',
            Metrics=['UnblendedCost'],
            GroupBy=[{'Type': 'DIMENSION', 'Key': 'REGION'}]
        )
        logger.info(f"Region Cost: {region_response}")
        
        region_costs = pd.DataFrame([
            {
                'REGION': group['Keys'][0],
                'cost': float(group['Metrics']['UnblendedCost']['Amount'])
            }
            for group in region_response['ResultsByTime'][0]['Groups']
        ])
        
        # Daily Cost
        daily_response = ce.get_cost_and_usage(
            TimePeriod={
                'Start': start_date.strftime('%Y-%m-%d'),
                'End': end_date.strftime('%Y-%m-%d')
            },
            Granularity='DAILY',
            Metrics=['UnblendedCost'],
            GroupBy=[{'Type': 'DIMENSION', 'Key': 'SERVICE'}]
        )
        logger.info(f"Daily Cost: {daily_response}")
        
        daily_costs = []
        for time_period in daily_response['ResultsByTime']:
            date = time_period['TimePeriod']['Start']
            for group in time_period['Groups']:
                daily_costs.append({
                    'date': date,
                    'SERVICE': group['Keys'][0],
                    'cost': float(group['Metrics']['UnblendedCost']['Amount'])
                })
        
        daily_costs_df = pd.DataFrame(daily_costs)
        logger.info(f"Daily Cost (df): {daily_costs_df}")
        
        global cost_data
        cost_data = {
            'service_costs': service_costs,
            'region_costs': region_costs,
            'daily_costs': daily_costs_df
        }
        return cost_data
        
    except Exception as e:
        logger.info(f"Error in cost analysis: {str(e)}")
        return None

def get_url(figure, prefix):
    # Convert fig_pie to base64 image
    img_bytes = pio.to_image(figure, format="png")
    base64_image = base64.b64encode(img_bytes).decode('utf-8')

    random_id = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=8))
    image_filename = f'{prefix}_{random_id}.png'
    
    # Convert base64 string back to bytes for S3 upload
    image_bytes = base64.b64decode(base64_image)
    url = chat.upload_to_s3(image_bytes, image_filename)
    logger.info(f"Uploaded image to S3: {url}")
    
    return url

def create_cost_visualizations():
    """Cost Visualization"""
    logger.info("Creating cost visualizations...")

    if not cost_data:
        logger.info("No cost data available")
        return None
        
    paths = []
    
    # service cost (pie chart)
    fig_pie = px.pie(
        cost_data['service_costs'],
        values='cost',
        names='SERVICE',
        title='Service Cost'
    )    
    paths.append(get_url(fig_pie, "service_costs"))
            
    # daily trend cost (line chart)
    fig_line = px.line(
        cost_data['daily_costs'],
        x='date',
        y='cost',
        color='SERVICE',
        title='Daily Cost Trend'
    )
    paths.append(get_url(fig_line, "daily_costs"))
    
    # region cost (bar chart)
    fig_bar = px.bar(
        cost_data['region_costs'],
        x='REGION',
        y='cost',
        title='Region Cost'
    )
    paths.append(get_url(fig_bar, "region_costs"))
    
    logger.info(f"paths: {paths}")

    return {
        "path": paths
    }

def generate_cost_insights():
    if cost_data:
        cost_data_dict = {
            'service_costs': cost_data['service_costs'].to_dict(orient='records'),
            'region_costs': cost_data['region_costs'].to_dict(orient='records'),
            'daily_costs': cost_data['daily_costs'].to_dict(orient='records') if 'daily_costs' in cost_data else []
        }
    else:
        return "Not available"

    system = (
        "당신의 AWS solutions architect입니다."
        "다음의 Cost Data을 이용하여 user의 질문에 답변합니다."
        "모르는 질문을 받으면 솔직히 모른다고 말합니다."
        "답변의 이유를 풀어서 명확하게 설명합니다."
    )
    human = (
        "다음 AWS 비용 데이터를 분석하여 상세한 인사이트를 제공해주세요:"
        "Cost Data:"
        "{raw_cost}"
        
        "다음 항목들에 대해 분석해주세요:"
        "1. 주요 비용 발생 요인"
        "2. 비정상적인 패턴이나 급격한 비용 증가"
        "3. 비용 최적화가 가능한 영역"
        "4. 전반적인 비용 추세와 향후 예측"
        
        "분석 결과를 다음과 같은 형식으로 제공해주세요:"

        "### 주요 비용 발생 요인"
        "- [구체적인 분석 내용]"

        "### 이상 패턴 분석"
        "- [비정상적인 비용 패턴 설명]"

        "### 최적화 기회"
        "- [구체적인 최적화 방안]"

        "### 비용 추세"
        "- [추세 분석 및 예측]"
    ) 

    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    # logger.info('prompt: ', prompt)    

    llm = chat.get_chat(extended_thinking="Disable")
    chain = prompt | llm

    raw_cost = json.dumps(cost_data_dict)

    try:
        response = chain.invoke(
            {
                "raw_cost": raw_cost
            }
        )
        logger.info(f"response: {response.content}")

    except Exception:
        err_msg = traceback.format_exc()
        logger.debug(f"error message: {err_msg}")                    
        raise Exception ("Not able to request to LLM")
    
    return response.content

def ask_cost_insights(question):
    if cost_data:
        cost_data_dict = {
            'service_costs': cost_data['service_costs'].to_dict(orient='records'),
            'region_costs': cost_data['region_costs'].to_dict(orient='records'),
            'daily_costs': cost_data['daily_costs'].to_dict(orient='records') if 'daily_costs' in cost_data else []
        }
    else:
        return "Cost 데이터를 가져오는데 실패하였습니다."

    system = (
        "당신의 AWS solutions architect입니다."
        "다음의 Cost Data을 이용하여 user의 질문에 답변합니다."
        "모르는 질문을 받으면 솔직히 모른다고 말합니다."
        "답변의 이유를 풀어서 명확하게 설명합니다."
    )
    human = (
        "Question: {question}"

        "Cost Data:"
        "{raw_cost}"        
    ) 

    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    # logger.info('prompt: ', prompt)    

    llm = chat.get_chat()
    chain = prompt | llm

    raw_cost = json.dumps(cost_data_dict)

    try:
        response = chain.invoke(
            {
                "question": question,
                "raw_cost": raw_cost
            }
        )
        logger.info(f"response: {response.content}")

    except Exception:
        err_msg = traceback.format_exc()
        logger.debug(f"error message: {err_msg}")                    
        raise Exception ("Not able to request to LLM")
    
    return response.content
