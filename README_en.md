# MCP Manus

Here a fully automated agent like Manus using MCP is implemented. This is based on the operation methods and prompts from [Bedrock-Manus: AI Automation Framework Based on Amazon Bedrock](https://github.com/aws-samples/aws-ai-ml-workshop-kr/tree/master/genai/aws-gen-ai-kr/20_applications/08_bedrock_manus) and [LangManus](https://github.com/Darwin-lfl/langmanus). Bedrock Manus and LangManus have the characteristics of Manus's fully automated multi-agent system, enabling them to generate good reports even for complex requests. Since MCP can easily connect various data sources, it makes it easy to develop various applications when combined with Manus.

[MCP](https://github.com/modelcontextprotocol) allows easy connection of various data sources to develop AI applications. Here, we use the [LangChain MCP adapter](https://github.com/langchain-ai/langchain-mcp-adapters) to retrieve tool capabilities from multiple MCP servers and use them to perform appropriate tasks. Additionally, we use [LangGraph Builder](https://build.langchain.com/) to easily design the MCP Manus Agent, improving the convenience of additional modifications and understanding. The overall architecture is as follows. MCP Manus can be installed and used locally on a personal PC, and can be deployed as a container on EC2 when needed. At this time, RAG can be utilized using Lambda, and external data such as tavily and wikipedia can be utilized using MCP servers.

<img width="700" alt="image" src="https://github.com/user-attachments/assets/f2bf4f83-279d-4bee-8699-53c3658648c8" />

## Detailed Implementation

### Implementing Workflow using LangBuilder

We define the Manus Workflow using LangBuilder. Visit [LangBuilder](https://build.langchain.com/) and draw the workflow as shown below, then use the code generator button on the right to generate LangGraph code. After downloading the generated stub.py, spec.yml, and implementation.py using Visual Studio or Cursor, implement the necessary nodes. For more details, refer to [Developing Agents with LangGraph Builder](https://github.com/kyopark2014/langgraph-builder).

<img src="./contents/flow_mcp_manus_final.gif" width="500">

When drawn as a graph using LangGraph Studio, it looks like this:

<img src="https://github.com/user-attachments/assets/07beb69d-aaf2-4fc3-bb4b-ddddbec72743" width="500">

### State Implementation

Manus utilizes Planning to implement a fully automated agent. For this purpose, the State uses full_plan to specify the plan that the LLM will execute. Additionally, messages are used to store the results of consecutive operations. The final results and generated report are stored using final_response and report, and displayed in the chat window.

```python
class State(TypedDict):
    full_plan: str
    messages: Annotated[list, add_messages]
    final_response: str
    report: str
```

### Multi-agent Structure

The Agent structure implemented with LangGraph is as follows. For detailed code, refer to [stub.py](./application/stub.py). The Coordinator judges whether the user's request needs planning and performs routing only when planning is necessary. The planner generates an appropriate plan using the system prompt from [planner.md](./application/planner.md). The Operator then executes the tool specified in the plan. Since one agent is created during tool execution, the overall structure operates as a multi-agent system.

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

### Coordinator Implementation

The Coordinator generates a system prompt using [coordinator.md](./application/coordinator.md) and then processes the user's input. At this time, if routing to to_planner is required by the prompt, it sets final_response to to_planner and passes it.

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

### Planner Implementation

The Planner constructs a system prompt using [planner.md](./application/planner.md). At this time, the entire state is passed as input to the planner, and the plan can be updated using full_plan and messages. When the LLM determines that the plan is complete according to the prompt in planner.md, it passes "Completed" in the <status> tag. This can be checked to set the final_response to the last item in messages. If the state is "Proceeding", it returns the generated plan.

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

### Operator Implementation

The Operator executes the appropriate tool according to the generated plan. At this time, the tool operates as an independent agent using tool information obtained from MCP. Here, using the system prompt from [operator.md](./application/operator.md), we obtain the tool and action to perform as next and task. Information about available tools is obtained from [MultiServerMCPClient](https://github.com/langchain-ai/langchain-mcp-adapters).

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

### Reporter Implementation

The report is generated using [reporter.md](./application/reporter.md).

```python
def Reporter(state: State, config: dict) -> dict:
    prompt_name = "Reporter"
    request_id = config.get("configurable", {}).get("request_id", "")    
    
    key = f"artifacts/{request_id}_steps.md"
    context = chat.get_object(key)

    system_prompt=get_prompt_template(prompt_name)
    llm = chat.get_chat(extended_thinking="Disable")
    human = (
        "Based on the following context, write an answer to the user's question.\n"
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

## Using MCP

Here we utilize MCP servers created from [MCP-github](https://github.com/kyopark2014/mcp). After selecting the necessary MCP server from the menu and entering a question, information about the tools is retrieved from the MCP server as shown below. Here, since code interpreter and tavily were selected, repl_coder, repl_drawer, tavily-search, and tavily-extract are the available tools.

![image](https://github.com/user-attachments/assets/eae586f7-5e50-4176-842b-6ae0a1803f63)

At this time, for a question like "How many r's are in Strawberry?", the following Plan was executed:

<img src="https://github.com/user-attachments/assets/f081f272-67ed-437f-aa70-d819c748a7af" width="550">

According to the Plan, the code interpreter was executed and the result was as follows:

<img src="https://github.com/user-attachments/assets/ca1943fe-8c93-4912-aa6d-f378fbb88d13" width="550">

## Usage Examples

After selecting "aws document", "aws diagram", and "tavily" and entering "How to implement a generative AI chatbot with RAG in AWS?", it retrieves information about available MCP tools and generates an appropriate plan as shown below.

![image](https://github.com/user-attachments/assets/76f8b1df-e8ec-4e05-b9a5-3e89c58906ec)

The available tools are read_documentation, search_documentation, recommend, generate_diagram, get_diagram_examples, list_icons, tavily-search, and tavily-extract. The plan generated using these is as follows:

```text
## Title: AWS Generative AI Chatbot with RAG Implementation Guide

## Steps:
### 1. search_documentation: Search AWS RAG and Chatbot related documentation
- [x] Search Amazon Bedrock related documentation
- [x] Search RAG implementation related AWS documentation
- [x] Search Amazon Kendra and OpenSearch related documentation
- [x] Search AWS Lambda and API Gateway related documentation

### 2. read_documentation: Detailed analysis of key documents
- [ ] Analyze Bedrock implementation guide
  - https://docs.aws.amazon.com/prescriptive-guidance/latest/retrieval-augmented-generation-options/rag-custom-retrievers.html
  - https://docs.aws.amazon.com/bedrock/latest/userguide/evaluation-kb.html
  - https://docs.aws.amazon.com/nova/latest/userguide/rag-systems.html
- [ ] Analyze RAG architecture implementation methods
- [ ] Analyze Kendra/OpenSearch integration methods
- [ ] Analyze Lambda function implementation guide
  - https://docs.aws.amazon.com/lambda/latest/dg/services-apigateway.html
  - https://docs.aws.amazon.com/lambda/latest/dg/services-apigateway-tutorial.html

### 3. generate_diagram: Generate RAG-based Chatbot architecture diagram
- [ ] Generate overall system architecture diagram
- [ ] Display data flow
- [ ] Display main AWS service integration structure
- [ ] Display process flow from user request to response

### 4. tavily-search: Research additional implementation cases and best practices
- [ ] Search AWS RAG implementation cases
- [ ] Research performance optimization methods
- [ ] Research cost optimization methods
- [ ] Research security configuration best practices
```

The generated result at this time was as follows:

<img src="https://github.com/user-attachments/assets/809cb6cb-aa41-41c3-969f-a9a86cad5609" width="550">

## Execution Results

Ask the question "Please explain about DNA strands" and check the results. The results can be viewed on the web page.

First, the plan is given in a checklist format, and tavily-search, search_papers, repl_drawer, and repl_coder are utilized appropriately for the purpose.

![image](https://github.com/user-attachments/assets/278884e1-f716-40bc-b8c4-94446a5e347c)

The step-by-step execution results store the results of each tool sequentially as shown below.

![image](https://github.com/user-attachments/assets/88dd6df3-8f76-43d1-9c32-bc7a476a4e50)

Finally, a result report like the following is generated. Although the report is not fully displayed due to its length, the final result generates a long report by collecting the step-by-step execution results.

![image](https://github.com/user-attachments/assets/4c6824fe-ef76-4390-b3e4-36b7f27d51bc) 
