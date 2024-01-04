import requests
import os
from typing import List, Union, Any, Optional, Type, Dict, Tuple
from langchain.schema import BaseMessage, PromptValue
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.language_models import BaseLLM, BaseChatModel
from langchain_core.outputs import LLMResult, ChatResult
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain.agents import AgentExecutor
from langchain.tools import BaseTool
from langchain_core.agents import AgentAction, AgentFinish, AgentStep

GuardInput = Union[str, List[BaseMessage], PromptValue]
NextStepOutput = List[Union[AgentFinish, AgentAction, AgentStep]]


session = requests.Session()  # Allows persistent connection (create only once)


class LakeraGuard():
    def __init__(self, api_key: str = os.environ.get("LAKERA_GUARD_API_KEY", ""), classification_name:str = "prompt_injection"):
        self.api_key = api_key
        # Check out other endpoint possibilities apart from prompt_injection at https://platform.lakera.ai/docs/api 
        self.classification_name = classification_name

    def call_lakera_guard(self, query: Union[str, list]) -> bool:
        response = session.post(
            f"https://api.lakera.ai/v1/{self.classification_name}",
            json={"input": query},
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        answer = response.json()
        result = answer["results"][0]["categories"][self.classification_name]
        return result
    
    @staticmethod
    def format_to_lakera_guard_input(input: GuardInput) -> Union[str, list]:
        # is input string?
        if isinstance(input, str):
            formatted_input = input
        else:
            if isinstance(input, PromptValue):
                input = input.to_messages()
            if isinstance(input, List):
                formatted_input = [{"role": "system","content": ""}, {"role": "user","content": ""}, {"role": "assistant","content": ""}]
                #For system, human, assistant, we put the last message of each type in the guard input
                for message in input:
                    if not isinstance(message, (HumanMessage, SystemMessage, AIMessage)) or not isinstance(message.content, str):
                        raise TypeError("Input type not supported by Lakera Guard.")
                    if isinstance(message, SystemMessage):
                        formatted_input[0]["content"] = message.content
                    elif isinstance(message, HumanMessage):
                        formatted_input[1]["content"] = message.content = message.content
                    else: #must be AIMessage
                        formatted_input[2]["content"] = message.content = message.content
            else:
                formatted_input = str(input)
        return formatted_input
    
    def lakera_guard(self, input: GuardInput) -> GuardInput:
        formatted_input = LakeraGuard.format_to_lakera_guard_input(input)
        lakera_guard_result = self.call_lakera_guard(formatted_input)
        if lakera_guard_result:
            raise ValueError(f"Lakera Guard detected {self.classification_name}.")
        return input

    def lakera_guard_with_feedback(self, input: GuardInput) -> bool:
        formatted_input = LakeraGuard.format_to_lakera_guard_input(input)
        lakera_guard_result = self.call_lakera_guard(formatted_input)
        return lakera_guard_result
    
    def guard_secured_llm(self, type_of_llm: Type[BaseLLM]):
        lakera_guard_instance = self
        class SecuredLLM(type_of_llm):
            @property
            def _llm_type(self) -> str:
                return "guard_secured_" + super()._llm_type

            def _generate(
                self,
                prompts: List[str],
                **kwargs: Any,
            ) -> LLMResult:
                for prompt in prompts:
                    lakera_guard_instance.lakera_guard(prompt)
                return super()._generate(prompts, **kwargs) 
        return SecuredLLM


    def guard_secured_chat_llm(self, type_of_chat_llm:Type[BaseChatModel]):
        lakera_guard_instance = self
        class SecuredChatLLM(type_of_chat_llm):
            @property
            def _llm_type(self) -> str:
                return "guard_secured_" + super()._llm_type
            def _generate(
                self,
                messages: List[BaseMessage],
                stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any,
            ) -> ChatResult:
                lakera_guard_instance.lakera_guard(messages)
                return super()._generate(messages, stop, run_manager, **kwargs) 
        return SecuredChatLLM
    

    def guard_secured_agent_executor(self):
        lakera_guard_instance = self

        class SecuredAgentExecutor(AgentExecutor):
            def _take_next_step(
                self,
                name_to_tool_map: Dict[str, BaseTool],
                color_mapping: Dict[str, str],
                inputs: Dict[str, str],
                intermediate_steps: List[Tuple[AgentAction, str]],
                *args,
                **kwargs,
            ):
                for val in inputs.values():
                    lakera_guard_instance.lakera_guard(val)
                res = super()._take_next_step(
                    name_to_tool_map,
                    color_mapping,
                    inputs,
                    intermediate_steps,
                    *args,
                    **kwargs,
                )
                for act in intermediate_steps:
                    lakera_guard_instance.lakera_guard(act[1])
                return res
            
        return SecuredAgentExecutor
