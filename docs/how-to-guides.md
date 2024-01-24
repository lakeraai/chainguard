# How-to Guides
You already have some LangChain code that uses either an LLM or agent? Then look at the code snippets below to see how you can secure it just with a small code change.

Make sure you have installed the **Lakera ChainGuard** package and got your Lakera Guard API key as an environment variable.
```python
from lakera_chainguard import LakeraChainGuard
chain_guard = LakeraChainGuard(endpoint="prompt_injection", raise_error=True)
```

### Guarding LLM
```python
llm = OpenAI()
```
-->
```python
GuardedOpenAI = chain_guard.get_guarded_llm(OpenAI)
llm = GuardedOpenAI()
```

### Guarding ChatLLM
```python
chatllm = ChatOpenAI()
```
-->
```python
GuardedChatOpenAI = chain_guard.get_guarded_chat_llm(ChatOpenAI)
chatllm = GuardedChatOpenAI()
```

### Guarding off-the-shelf agent
```python
llm = OpenAI()
agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)
```
-->
```python
GuardedOpenAI = chain_guard.get_guarded_llm(OpenAI)
llm = GuardedOpenAI()
agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)
```

### Guarding custom agent

```python
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
```
-->
```python
GuardedAgentExecutor = chain_guard.get_guarded_agent_executor()
agent_executor = GuardedAgentExecutor(agent=agent, tools=tools, verbose=True)
```