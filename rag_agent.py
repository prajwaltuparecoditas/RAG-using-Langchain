from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from tools import pdf_ans, web_ans, yt_ans, llm


tools = [pdf_ans, web_ans, yt_ans]
llm_with_tools = llm.bind_tools([pdf_ans])

prompt = hub.pull("hwchase17/openai-tools-agent")
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)
resource = input("Provide resource path(url, pdf, youtube video): ")

while True:
    user_query = input("Enter your query: ")
    response = agent_executor.invoke({"input":f"{user_query} resource path:{resource}"})
    print(response['output'])

