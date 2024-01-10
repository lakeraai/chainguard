from __future__ import annotations
import os
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import warnings
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


class LakeraGuardError(RuntimeError):
    def __init__(self, message: str, lakera_guard_response: dict) -> None:
        """
        Custom error that gets raised if Lakera Guard detects AI security risk.

        Args:
            message: error message
            lakera_guard_response: Lakera Guard's API response in json format
        Returns:
            None
        """
        super().__init__(message)
        self.lakera_guard_response = lakera_guard_response


class LakeraGuardWarning(RuntimeWarning):
    def __init__(self, message: str, lakera_guard_response: dict) -> None:
        """
        Custom warning that gets raised if Lakera Guard detects AI security risk.

        Args:
            message: error message
            lakera_guard_response: Lakera Guard's API response in json format
        Returns:
            None
        """
        super().__init__(message)
        self.lakera_guard_response = lakera_guard_response


session = requests.Session()  # Allows persistent connection (create only once)


class LakeraChainGuard:
    def __init__(
        self,
        api_key: str = os.environ.get("LAKERA_GUARD_API_KEY", ""),
        classifier: str = "prompt_injection",
        raise_error: bool = True,
    ) -> None:
        """
        Contains different methods that help with guarding LLMs and agents in LangChain.

        Args:
            api_key: API key for Lakera Guard
            classifier: which AI security risk you want to guard against, see also
                classifiers available here: https://platform.lakera.ai/docs/api
            raise_error: whether to raise an error or a warning if the classifier
                detects AI security risk
        Returns:
            None
        """
        self.api_key = api_key
        self.classifier = classifier
        self.raise_error = raise_error

    def call_lakera_guard(self, query: Union[str, list[dict[str, str]]]) -> dict:
        """
        Makes an API request to the Lakera Guard API endpoint specified in
        self.classifier.

        Args:
            query: User prompt or list of message containing system, user
                and assistant roles.
        Returns:
            The classifier's API response as dict
        """
        response = session.post(
            f"https://api.lakera.ai/v1/{self.classifier}",
            json={"input": query},
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        answer = response.json()
        # result = answer["results"][0]["categories"][self.classifier]
        return answer

    def format_to_lakera_guard_input(
        self, input: GuardInput
    ) -> Union[str, list[dict[str, str]]]:
        """
        Formats the input into LangChain's LLMs or ChatLLMs to be compatible as Lakera
        Guard input.

        Args:
            input: Object that follows LangChain's LLM or ChatLLM input format
        Returns:
            Object that follows Lakera Guard's input format
        """
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
                if self.classifier != "prompt_injection":
                    return formatted_input[1]["content"]
                return formatted_input
            else:
                return str(input)

    def detect(self, input: GuardInput) -> GuardInput:
        """
        If input contains AI security risk specified in self.classifier, raises either
        LakeraGuardError or LakeraGuardWarning depending on self.raise_error True or
        False. Otherwise, lets input through.

        Args:
            input: input to check regarding AI security risk
        Returns:
            input unchanged
        """
        formatted_input = self.format_to_lakera_guard_input(input)
        lakera_guard_response = self.call_lakera_guard(formatted_input)
        if lakera_guard_response["results"][0]["categories"][self.classifier]:
            if self.raise_error:
                raise LakeraGuardError(
                    f"Lakera Guard detected {self.classifier}.", lakera_guard_response
                )
            else:
                warnings.warn(
                    LakeraGuardWarning(
                        f"Lakera Guard detected {self.classifier}.",
                        lakera_guard_response,
                    )
                )
        return input

    def detect_with_response(self, input: GuardInput) -> dict:
        """
        Returns detection result of AI security risk specified in self.classifier
        with regard to the input.

        Args:
            input: input to check regarding AI security risk
        Returns:
            detection result of AI security risk specified in self.classifier
        """
        formatted_input = self.format_to_lakera_guard_input(input)
        lakera_guard_response = self.call_lakera_guard(formatted_input)
        return lakera_guard_response

    def get_guarded_llm(self, type_of_llm: Type[BaseLLM]) -> Type[BaseLLM]:
        """
        Creates a subclass of type_of_llm where the input to the LLM always gets
        checked w.r.t. AI security risk specified in self.classifier.

        Args:
            type_of_llm: any type of LangChain's LLMs
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

    def get_guarded_chat_llm(
        self, type_of_chat_llm: Type[BaseChatModel]
    ) -> Type[BaseChatModel]:
        """
        Creates a subclass of type_of_chat_llm in which the input to the ChatLLM always
          gets checked w.r.t. AI security risk specified in self.classifier.

        Args:
            type_of_llm: any type of LangChain's ChatLLMs
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

    def get_guarded_agent_executor(self) -> Type[AgentExecutor]:
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
