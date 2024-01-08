import os
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import requests
from langchain.agents import AgentExecutor
from langchain.schema import BaseMessage, PromptValue
from langchain.tools import BaseTool
from langchain_core.agents import AgentAction, AgentFinish, AgentStep
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, BaseLLM
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult, LLMResult

GuardInput = Union[str, List[BaseMessage], PromptValue]
NextStepOutput = List[Union[AgentFinish, AgentAction, AgentStep]]


session = requests.Session()  # Allows persistent connection (create only once)


class LakeraChainGuard:
    def __init__(
        self,
        api_key: str = os.environ.get("LAKERA_GUARD_API_KEY", ""),
        classifier: str = "prompt_injection",
    ):
        """
        Contains different methods that help with guarding LLMs and agents in LangChain.

        Args:
            api_key: API key for Lakera Guard
            classifier: which AI security risk you want to guard against, see also
                classifiers available here: https://platform.lakera.ai/docs/api
        Returns:

        """
        self.api_key = api_key
        self.classifier = classifier

    def call_lakera_guard(self, query: Union[str, list]) -> bool:
        """
        Makes an API request to the Lakera Guard API endpoint specified in
        self.classifier.

        Args:
            query: User prompt or list of message containing system, user
                and assistant roles.
        Returns:
            The classifier's detection result
        """
        response = session.post(
            f"https://api.lakera.ai/v1/{self.classifier}",
            json={"input": query},
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        answer = response.json()
        result = answer["results"][0]["categories"][self.classifier]
        return result

    @staticmethod
    def format_to_lakera_guard_input(
        input: GuardInput,
    ) -> Union[str, List[dict[str, str]]]:
        """
        Formats the input into LangChain's LLMs or ChatLLMs to be compatible as Lakera
        Guard input.

        Args:
            input: Parameter that follows LangChain's LLM or ChatLLM input format
        Returns:
            Parameter that follows Lakera Guard's input format
        """
        # is input string?
        if isinstance(input, str):
            return input
        else:
            if isinstance(input, PromptValue):
                input = input.to_messages()
            if isinstance(input, List):
                formatted_input = [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": ""},
                    {"role": "assistant", "content": ""},
                ]
                # For system, human, assistant, we put the last message of each
                # type in the guard input
                for message in input:
                    if not isinstance(
                        message, (HumanMessage, SystemMessage, AIMessage)
                    ) or not isinstance(message.content, str):
                        raise TypeError("Input type not supported by Lakera Guard.")
                    if isinstance(message, SystemMessage):
                        formatted_input[0]["content"] = message.content
                    elif isinstance(message, HumanMessage):
                        formatted_input[1]["content"] = message.content
                    else:  # must be AIMessage
                        formatted_input[2]["content"] = message.content
                return formatted_input
            else:
                return str(input)

    def detect(self, input: GuardInput) -> GuardInput:
        """
        Raises error if input contains AI security risk specified in self.classifier.
        Otherwise, lets input through.

        Args:
            input: input to check regarding AI security risk
        Returns:
            input unchanged
        Raises:
            ValueError if input contains AI security risk specified in self.classifier
        """
        formatted_input = LakeraChainGuard.format_to_lakera_guard_input(input)
        lakera_guard_result = self.call_lakera_guard(formatted_input)
        if lakera_guard_result:
            raise ValueError(f"Lakera Guard detected {self.classifier}.")
        return input

    def detect_with_feedback(self, input: GuardInput) -> bool:
        """
        Returns detection result of AI security risk specified in self.classifier
        with regard to the input.

        Args:
            input: input to check regarding AI security risk
        Returns:
            detection result of AI security risk specified in self.classifier
        """
        formatted_input = LakeraChainGuard.format_to_lakera_guard_input(input)
        lakera_guard_result = self.call_lakera_guard(formatted_input)
        return lakera_guard_result

    def get_guarded_llm(self, type_of_llm: Type[BaseLLM]):
        """
        Creates a subclass of type_of_llm where the input to the LLM always gets
        checked w.r.t. AI security risk specified in self.classifier.

        Args:
            type_of_llm: any type of LangChain's LLM specified in
                langchain_community.llms
        Returns:
            Guarded subclass of type_of_llm
        """
        lakera_guard_instance = self

        class GuardedLLM(type_of_llm):
            @property
            def _llm_type(self) -> str:
                return "guarded_" + super()._llm_type

            def _generate(
                self,
                prompts: List[str],
                **kwargs: Any,
            ) -> LLMResult:
                for prompt in prompts:
                    lakera_guard_instance.detect(prompt)

                return super()._generate(prompts, **kwargs)

        return GuardedLLM

    def get_guarded_chat_llm(self, type_of_chat_llm: Type[BaseChatModel]):
        """
        Creates a subclass of type_of_chat_llm in which the input to the ChatLLM always
          gets checked w.r.t. AI security risk specified in self.classifier.

        Args:
            type_of_llm: any type of LangChain's LLM specified in
            langchain_community.chat_models
        Returns:
            Guarded subclass of type_of_llm
        """
        lakera_guard_instance = self

        class GuardedChatLLM(type_of_chat_llm):
            @property
            def _llm_type(self) -> str:
                return "guarded_" + super()._llm_type

            def _generate(
                self,
                messages: List[BaseMessage],
                stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any,
            ) -> ChatResult:
                lakera_guard_instance.detect(messages)
                return super()._generate(messages, stop, run_manager, **kwargs)

        return GuardedChatLLM

    def get_guarded_agent_executor(self):
        """
        Creates a subclass of the AgentExecutor in which the input to the LLM that the
        AgentExecutor is initialized with gets checked w.r.t. AI security risk specified
        in self.classifier.

        Returns:
            Guarded AgentExecutor subclass
        """
        lakera_guard_instance = self

        class GuardedAgentExecutor(AgentExecutor):
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
                    lakera_guard_instance.detect(val)

                res = super()._take_next_step(
                    name_to_tool_map,
                    color_mapping,
                    inputs,
                    intermediate_steps,
                    *args,
                    **kwargs,
                )

                for act in intermediate_steps:
                    lakera_guard_instance.detect(act[1])

                return res

        return GuardedAgentExecutor
