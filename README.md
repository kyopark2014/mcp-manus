# MCP Manus

여기에서는 MCP로 Manus와 같은 완전 자동화 에이전트(Autonomous AI Agent Redefining)를 구현합니다. 이것은 [Bedrock-Manus: AI Automation Framework Based on Amazon Bedrock](https://github.com/aws-samples/aws-ai-ml-workshop-kr/tree/master/genai/aws-gen-ai-kr/20_applications/08_bedrock_manus)와 [LangManus](https://github.com/Darwin-lfl/langmanus)의 동작 방식과 [Prompt](https://github.com/aws-samples/aws-ai-ml-workshop-kr/tree/master/genai/aws-gen-ai-kr/20_applications/08_bedrock_manus/src/prompts)를 참조하고 있습니다. Bedrock Manus와 LangManus는 Manus의 특징인 완전 자동화 멀티 에이전트의 특징을 가지고 있어서, 복잡한 요청에도 좋은 리포트를 생성할 수 있습니다. MCP는 다양한 데이터 소스를 쉽게 연결할 수 있으므로 Manus 결합시 다양한 애플리케이션을 손쉽게 개발할 수 있습니다.

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
    if result.content != 'to_planner':
        logger.info(f"next: END")
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
def Reporter(state: State) -> dict:
    prompt_name = "Reporter"
    system_prompt=get_prompt_template(prompt_name)
    logger.info(f"system_prompt: {system_prompt}")
    
    llm = chat.get_chat(extended_thinking="Disable")

    human = "{messages}"
    reporter_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", human)
        ]
    )

    prompt = reporter_prompt | llm 
    result = prompt.invoke({
        "messages": state["messages"]
    })

    return {
        "report": result.content,
        "full_plan": state["full_plan"]
    }
```

### MCP의 활용

여기에서는 [MCP-github](https://github.com/kyopark2014/mcp)에서 생성한 MCP server들을 활용합니다. 아래와 같이 메뉴에서 필요한 MCP 서버를 선택한 후에 질문을 입력하면 아래와 같이 tool들에 대한 정보를 MCP server로부터 가져옵니다. 여기에서는 code interpreter와 tavily를 선택하였으므로, repl_coder, repl_drawer, tavily-search, tavily-extract가 사용할 수 있는 tool입니다. 

![image](https://github.com/user-attachments/assets/c1eac880-89be-4a0d-85a7-3f719bdaf032)

이때 "Strawberry의 r의 갯수는?"와 같은 질문에 대해서 아래와 같은 Plan이 실행되었습니다.

<img src="https://github.com/user-attachments/assets/f081f272-67ed-437f-aa70-d819c748a7af" width="550">

Plan에 따라서 code interpreter가 실행되었고 결과는 아래와 같습니다.

<img src="https://github.com/user-attachments/assets/ca1943fe-8c93-4912-aa6d-f378fbb88d13" width="550">
