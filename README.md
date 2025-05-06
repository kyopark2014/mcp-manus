# MCP Manus

여기에서는 MCP로 Manus와 같은 완전 자동화 에이전트(Autonomous AI Agent Redefining)를 구현합니다. 이것은 [Bedrock-Manus: AI Automation Framework Based on Amazon Bedrock](https://github.com/aws-samples/aws-ai-ml-workshop-kr/tree/master/genai/aws-gen-ai-kr/20_applications/08_bedrock_manus)와 [LangManus](https://github.com/Darwin-lfl/langmanus)의 동작 방식과 [Prompt](https://github.com/aws-samples/aws-ai-ml-workshop-kr/tree/master/genai/aws-gen-ai-kr/20_applications/08_bedrock_manus/src/prompts)를 참조하고 있습니다. Bedrock Manus와 LangManus는 Manus의 특징인 완전 자동화 멀티 에이전트의 특징을 가지고 있어서, 복잡한 요청에도 좋은 리포트를 생성할 수 있습니다. MCP는 다양한 데이터 소스를 쉽게 연결할 수 있으므로 Manus 결합시 다양한 애플리케이션을 손쉽게 개발할 수 있습니다.

## LangBuilder를 이용해 Workflow 구현

LangBuilder를 이용해 Manus Workflow를 정의합니다. [LangBuilder를](https://build.langchain.com/)에 접속하여 아래와 같이 workflow를 그린 후, 오른쪽의 code generator 버튼을 이용해 LangGraph 코드를 생성합니다. 이후 생성한 stub.py, spec.yml, implementation.py을 visual studio나 cursor로 다운로드 한 후 필요한 노드를 구현합니다. 상세 내용은 [LangGraph Builder로 Agent 개발하기](https://github.com/kyopark2014/langgraph-builder)을 참조합니다.

<img src="./contents/flow_mcp_manus_final.gif" width="500">

이를 LangGraph Studio를 이용해 graph 형태로 그리면 아래와 같습니다.

<img src="https://github.com/user-attachments/assets/07beb69d-aaf2-4fc3-bb4b-ddddbec72743" width="400">

