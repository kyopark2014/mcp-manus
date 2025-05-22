# MCP Manus

여기에서는 MCP를 이용하여 Manus와 같은 완전 자동화 에이전트(Autonomous AI Agent Redefining)를 구현합니다. 이것은 [Bedrock-Manus: AI Automation Framework Based on Amazon Bedrock](https://github.com/aws-samples/aws-ai-ml-workshop-kr/tree/master/genai/aws-gen-ai-kr/20_applications/08_bedrock_manus)와 [LangManus](https://github.com/Darwin-lfl/langmanus)의 동작 방식과 [Prompt](https://github.com/aws-samples/aws-ai-ml-workshop-kr/tree/master/genai/aws-gen-ai-kr/20_applications/08_bedrock_manus/src/prompts)를 참조하고 있습니다. Bedrock Manus와 LangManus는 Manus의 특징인 완전 자동화 멀티 에이전트의 특징을 가지고 있어서, 복잡한 요청에도 좋은 리포트를 생성할 수 있습니다. MCP는 다양한 데이터 소스를 쉽게 연결할 수 있으므로 Manus 결합시 다양한 애플리케이션을 손쉽게 개발할 수 있습니다.

전체적인 Architecture는 아래와 같습니다. 

<img width="398" alt="image" src="https://github.com/user-attachments/assets/9f1cbe01-efd4-43a4-a28e-672abd7a1c8b" />


## 상세 구현

### LangBuilder를 이용해 Workflow 구현

LangBuilder를 이용해 Manus Workflow를 정의합니다. [LangBuilder를](https://build.langchain.com/)에 접속하여 아래와 같이 workflow를 그린 후, 오른쪽의 code generator 버튼을 이용해 LangGraph 코드를 생성합니다. 이후 생성한 stub.py, spec.yml, implementation.py을 visual studio나 cursor로 다운로드 한 후 필요한 노드를 구현합니다. 상세 내용은 [LangGraph Builder로 Agent 개발하기](https://github.com/kyopark2014/langgraph-builder)을 참조합니다.

<img src="./contents/flow_mcp_manus_final.gif" width="500">

이를 LangGraph Studio를 이용해 graph 형태로 그리면 아래와 같습니다.

<img src="https://github.com/user-attachments/assets/07beb69d-aaf2-4fc3-bb4b-ddddbec72743" width="500">

### State 구현

Manus는 완전 자동화된 agent를 구현하기 위하여 Planning을 활용하고 있습니다. 이를 위해 State에는 아래와 같이 full_plan을 이용하여 LLM이 수행할 계획을 지정합니다. 또한 연속되는 동작의 결과를 저장하기 위하여 messages를 활용합니다. 최종 결과와 생성된 report는 final_response와 report를 활용해 저장되고 채팅창에 표시됩니다.

```python
class State(TypedDict):
    full_plan: str
    messages: Annotated[list, add_messages]
    final_response: str
    report: str
```

### Multi-agent 구조

여기서 LangGraph로 구현된 Agent 구조는 아래와 같습니다. 상세한 코드는 [stub.py](./application/stub.py)을 참조합니다. Coordinate는 사용자의 요청이 planning이 필요한 지 판단하여, planning이 필요한 경우에만 라우팅을 수행합니다. planning은 prompt를 [planner.md](./application/planner.md)을 이용해 system prompt를 생성하고, 적절한 plan을 생성합니다. 이후 Operator는 plan에서 지정한 tool을 수행하게 됩니다. tool 실행시 하나의 agent를 생성하므로 전체적인 구조는 multi agent 형태로 동작합니다. 

```python
# Add nodes
builder.add_node("Coordinator", nodes_by_name["Coordinator"])
builder.add_node("Planner", nodes_by_name["Planner"])
builder.add_node("Operator", nodes_by_name["Operator"])
builder.add_node("Reporter", nodes_by_name["Reporter"])
# Add edges
builder.add_edge(START, "Coordinator")    
builder.add_conditional_edges(
    "Coordinator",
    nodes_by_name["to_planner"],
    [
        END,
        "Planner",
    ],
)
builder.add_conditional_edges(
    "Planner",
    nodes_by_name["to_operator"],
    [
        "Operator",
        "Reporter",
    ],
)    
builder.add_edge("Operator", "Planner")
builder.add_edge("Reporter", END)
```

### Coodinater의 구현

Coordinatr는 [coordinator.md](./application/coordinator.md)을 이용하여 system prompt를 생성한 후에 사용자의 입력을 처리합니다. 이때 prompt에 의해서 to_planner로 라우팅이 필욯다마녀 final_response을 to_planner로 설정하여 전달합니다. 

```python
def Coordinator(state: State) -> dict:
    question = state["messages"][0].content
    prompt_name = "coordinator"
    system_prompt=get_prompt_template(prompt_name)
    
    llm = chat.get_chat(extended_thinking="Disable")
    coordinator_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}"),
        ]
    )

    prompt = coordinator_prompt | llm 
    result = prompt.invoke({
        "question": question
    })

    final_response = ""
    if result.content.find('to_planner') == -1:
        result.content = result.content.split('<next>')[0]
        final_response = result.content    
    
    return {
        "final_response": final_response
    }
```

### Planner의 구현

Planner는 [planner.md](./application/planner.md)을 이용해 system prompt를 구성합니다. 이때 planner의 input으로 state 전체를 전달하는데, full_plan과 messages를 이용해 plan을 업데이트 할 수 있습니다. planner.md의 prompt에 의해서 LLM이 plan을 완료하였다고 판단이 되면 <status> tag에 "Completed"라고 전달합니다. 이를 확인하여 messages의 마지막 항목을 final_response를 설정할 수 있습니다. state가 "Procedding"이라면 생성된 plan을 리턴합니다. 

```python
def Planner(state: State) -> dict:
    prompt_name = "planner"
    system = get_prompt_template(prompt_name)
    human = "{input}" 

    llm = chat.get_chat(extended_thinking="Disable")
    planner_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", human),
        ]
    )

    prompt = planner_prompt | llm 
    result = prompt.invoke({
        "team_members": team_members,
        "input": state
    })

    output = result.content
    if output.find("<status>") != -1:
        status = output.split("<status>")[1].split("</status>")[0]
        if status == "Completed":
            final_response = state["messages"][-1].content
            return {
                "full_plan": result.content,
                "final_response": final_response                
            }

    return {
        "full_plan": result.content,
    }
```

### Operator의 구현

생성된 plan에 따라서 Operator는 적절한 tool을 실행합니다. 이때 tool은 MCP로 부터 얻은 tool 정보를 활용하여 독립된 agent로 동작합니다. 여기에서는 [operator.md](./application/operator.md)의 system prompt를 이용하여, 수행할 tool과 동작을 next와 task로 얻습니다. 사용 가능한 tool의 정보는 [MultiServerMCPClient](https://github.com/langchain-ai/langchain-mcp-adapters)로 부터 얻어옵니다. 

```python
async def Operator(state: State) -> dict:
    prompt_name = "operator"
    system = get_prompt_template(prompt_name)
    human = "{input}" 

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", human),
        ]
    )

    llm = chat.get_chat(extended_thinking="Disable")
    chain = prompt | llm 
    result = chain.invoke({
        "input": state,
        "team_members": team_members
    })
    result_dict = json.loads(result.content)
    next = result_dict["next"]
    task = result_dict["task"]
    if next == "FINISHED":
        return
    else:
        tool_info = []
        for tool in tool_list:
            if tool.name == next:
                tool_info.append(tool)
                logger.info(f"tool_info: {tool_info}")
        
        # Agent
        agent, config = chat.create_agent(tool_info)
        messages = [HumanMessage(content=json.dumps(task))]
        response = await agent.ainvoke({"messages": messages}, config)
        output = response["messages"][-1].content
        
        return {
            "messages": [
                HumanMessage(content=json.dumps(task)),
                AIMessage(content=output)
            ]
        }
```

### Reporter의 구현

[reporter.md](./application/reporter.md)을 활용하여 보고서를 생성합니다. 

```python
def Reporter(state: State, config: dict) -> dict:
    prompt_name = "Reporter"
    request_id = config.get("configurable", {}).get("request_id", "")    
    
    key = f"artifacts/{request_id}_steps.md"
    context = chat.get_object(key)

    system_prompt=get_prompt_template(prompt_name)
    llm = chat.get_chat(extended_thinking="Disable")
    human = (
        "다음의 context를 바탕으로 사용자의 질문에 대한 답변을 작성합니다.\n"
        "<question>{question}</question>\n"
        "<context>{context}</context>"
    )
    reporter_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", human)
        ]
    )

    question = state["messages"][0].content
    logger.info(f"question: {question}")

    prompt = reporter_prompt | llm 
    result = prompt.invoke({
        "context": context,
        "question": question
    })
    logger.info(f"result of Reporter: {result}")

    key = f"artifacts/{request_id}_report.md"
    chat.create_object(key, result.content)

    return {
        "report": result.content
    }
```

## MCP의 활용

여기에서는 [MCP-github](https://github.com/kyopark2014/mcp)에서 생성한 MCP server들을 활용합니다. 아래와 같이 메뉴에서 필요한 MCP 서버를 선택한 후에 질문을 입력하면 아래와 같이 tool들에 대한 정보를 MCP server로부터 가져옵니다. 여기에서는 code interpreter와 tavily를 선택하였으므로, repl_coder, repl_drawer, tavily-search, tavily-extract가 사용할 수 있는 tool입니다. 

![image](https://github.com/user-attachments/assets/c1eac880-89be-4a0d-85a7-3f719bdaf032)

이때 "Strawberry의 r의 갯수는?"와 같은 질문에 대해서 아래와 같은 Plan이 실행되었습니다.

<img src="https://github.com/user-attachments/assets/f081f272-67ed-437f-aa70-d819c748a7af" width="550">

Plan에 따라서 code interpreter가 실행되었고 결과는 아래와 같습니다.

<img src="https://github.com/user-attachments/assets/ca1943fe-8c93-4912-aa6d-f378fbb88d13" width="550">

## 활용 예제

아래와 같이 "aws document", "aws diagram", "tavily"을 선택한 후에 "aws에서 생성형 AI chatbot을 RAG와 함께 구현하는 방법?"을 입력하면 아래와 같이 사용할 수 있는 MCP tool들의 정보를 가져와서 적절한 plan을 생성합니다.

![image](https://github.com/user-attachments/assets/76f8b1df-e8ec-4e05-b9a5-3e89c58906ec)

사용할 수 있는 tool은 read_documentation, search_documentation, recommend, generate_diagram, get_diagram_examples, list_icons, tavily-search, tavily-extract 입니다. 이를 이용해 생성한 plan은 아래와 같습니다.

```text
## Title: AWS 생성형 AI Chatbot with RAG 구현 가이드

## Steps:
### 1. search_documentation: AWS RAG 및 Chatbot 관련 문서 검색
- [x] Amazon Bedrock 관련 문서 검색
- [x] RAG 구현 관련 AWS 문서 검색
- [x] Amazon Kendra와 OpenSearch 관련 문서 검색
- [x] AWS Lambda와 API Gateway 관련 문서 검색

### 2. read_documentation: 핵심 문서 상세 분석
- [ ] Bedrock 구현 가이드 분석
  - https://docs.aws.amazon.com/prescriptive-guidance/latest/retrieval-augmented-generation-options/rag-custom-retrievers.html
  - https://docs.aws.amazon.com/bedrock/latest/userguide/evaluation-kb.html
  - https://docs.aws.amazon.com/nova/latest/userguide/rag-systems.html
- [ ] RAG 아키텍처 구현 방법 분석
- [ ] Kendra/OpenSearch 통합 방법 분석
- [ ] Lambda 함수 구현 가이드 분석
  - https://docs.aws.amazon.com/lambda/latest/dg/services-apigateway.html
  - https://docs.aws.amazon.com/lambda/latest/dg/services-apigateway-tutorial.html

### 3. generate_diagram: RAG 기반 Chatbot 아키텍처 다이어그램 생성
- [ ] 전체 시스템 아키텍처 다이어그램 생성
- [ ] 데이터 흐름 표시
- [ ] 주요 AWS 서비스 연동 구조 표시
- [ ] 사용자 요청부터 응답까지의 프로세스 플로우 표시

### 4. tavily-search: 추가 구현 사례 및 모범 사례 조사
- [ ] AWS RAG 구현 사례 검색
- [ ] 성능 최적화 방법 조사
- [ ] 비용 최적화 방안 조사
- [ ] 보안 구성 모범 사례 조사
```

이때 생성된 결과는 아래와 같습니다.

<img src="https://github.com/user-attachments/assets/809cb6cb-aa41-41c3-969f-a9a86cad5609" width="550">

## 실행 결과

"DNA의 strands에 대해 설명해주세요."와 질문을 하고 결과를 확인합니다. 결과는 web page로 확인할 수 있습니다.

먼저, 계획은 아래와 같이 checklist 형태로 주어지며, tavily-search, search_papers, repl_drawer, repl_coder가 목적에 맞게 활용됩니다. 

![image](https://github.com/user-attachments/assets/278884e1-f716-40bc-b8c4-94446a5e347c)

단계별 실행 결과에는 각 tool들의 결과가 아래와 같이 순차적으로 저장됩니다. 

![image](https://github.com/user-attachments/assets/88dd6df3-8f76-43d1-9c32-bc7a476a4e50)

이후, 최종적으로는 아래와 같은 결과 리포트가 생성됩니다. 리포트가 길어서 표시되지 않았지만, 최종 결과는 단계별 실행 결과를 모아서 장문의 리포트를 생성하게 됩니다.

![image](https://github.com/user-attachments/assets/4c6824fe-ef76-4390-b3e4-36b7f27d51bc)
