from __future__ import annotations

import os
import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Literal, TypeVar

import requests
from langchain.agents import AgentExecutor
from langchain.schema import BaseMessage, PromptValue
from langchain.tools import BaseTool
from langchain_core.agents import AgentStep
from langchain.callbacks.manager import (
    CallbackManagerForLLMRun,
    CallbackManagerForChainRun,
)
from langchain.schema.agent import AgentFinish, AgentAction
from langchain_core.language_models import BaseChatModel, BaseLLM
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult, LLMResult

# from langchain.callbacks.manager import CallbackManagerForChainRun

GuardInput = Union[str, List[BaseMessage], PromptValue]
NextStepOutput = List[Union[AgentFinish, AgentAction, AgentStep]]
GuardChatMessages = list[dict[str, str]]
Endpoints = Literal[
    "prompt_injection",
    "moderation",
    "pii",
    "relevant_language",
    "sentiment",
    "unknown_links",
]
BaseLLMT = TypeVar("BaseLLMT", bound=BaseLLM)
BaseChatModelT = TypeVar("BaseChatModelT", bound=BaseChatModel)


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
        api_key: str = "",
        endpoint: Endpoints = "prompt_injection",
        additional_json_properties: dict = dict(),
        raise_error: bool = True,
    ) -> None:
        """
        Contains different methods that help with guarding LLMs and agents in LangChain.

        Args:
            api_key: API key for Lakera Guard
            endpoint: which AI security risk you want to guard against, see also
                classifier endpoints available here: https://platform.lakera.ai/docs/api
            additional_json_properties: add additional key-value pairs to the body of
                the API request apart from 'input', e.g. domain_whitelist for pii
            raise_error: whether to raise an error or a warning if the classifier
                endpoint detects AI security risk
        Returns:
            None
        """
        # In the arguments of the __init__, we cannot set api_key: str =
        # os.environ.get("LAKERA_GUARD_API_KEY") because this would only be
        # evaluated once when the class is imported. This would mean that if the
        # user sets the environment variable (e.g. via load_dotenv()) after importing
        # the class , the class would not use the environment variable.
        self.api_key = api_key or os.environ.get("LAKERA_GUARD_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No Lakera Guard API key provided. Either provide it in the "
                "constructor or set the environment variable LAKERA_GUARD_API_KEY."
            )
        self.endpoint = endpoint
        self.additional_json_properties = additional_json_properties
        self.raise_error = raise_error

    def _call_lakera_guard(self, query: Union[str, GuardChatMessages]) -> dict:
        """
        Makes an API request to the Lakera Guard API endpoint specified in
        self.endpoint.

        Args:
            query: User prompt or list of message containing system, user
                and assistant roles.
        Returns:
            The endpoints's API response as dict
        """
        request_input = {"input": query}

        if "input" in self.additional_json_properties:
            raise ValueError(
                'You cannot specify the "input" argument in additional_json_properties.'
            )

        request_body = self.additional_json_properties | request_input

        response = session.post(
            f"https://api.lakera.ai/v1/{self.endpoint}",
            json=request_body,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        response_body = response.json()

        # Error handling
        if "error" in response_body:
            if response_body["error"] == "Unauthorized":
                raise ValueError(
                    str(response_body) + " Please provide a valid Lakera Guard API key."
                )
            elif response_body["error"] == "Invalid Request":
                raise ValueError(
                    str(response_body)
                    + (
                        f" Provided properties {str(self.additional_json_properties)} "
                        "in 'additional_json_properties' are not valid."
                    )
                )
            else:
                raise ValueError(str(response_body))
        if "results" not in response_body:
            raise ValueError(str(response_body))

        return response_body

    def _convert_to_lakera_guard_input(
        self, prompt: GuardInput
    ) -> Union[str, list[dict[str, str]]]:
        """
        Formats the input into LangChain's LLMs or ChatLLMs to be compatible as Lakera
        Guard input.

        Args:
            prompt: Object that follows LangChain's LLM or ChatLLM input format
        Returns:
            Object that follows Lakera Guard's input format
        """
        if isinstance(prompt, str):
            return prompt
        else:
            if isinstance(prompt, PromptValue):
                prompt = prompt.to_messages()
            if isinstance(prompt, List):
                user_message = ""
                formatted_input = []
                for message in prompt:
                    if not isinstance(
                        message, (HumanMessage, SystemMessage, AIMessage)
                    ) or not isinstance(message.content, str):
                        raise TypeError("Input type not supported by Lakera Guard.")

                    role = "assistant"
                    if isinstance(message, SystemMessage):
                        role = "system"
                    elif isinstance(message, HumanMessage):
                        user_message = message.content
                        role = "user"

                    formatted_input.append({"role": role, "content": message.content})

                if self.endpoint != "prompt_injection":
                    return user_message
                return formatted_input
            else:
                return str(prompt)

    def detect(self, prompt: GuardInput) -> GuardInput:
        """
        If input contains AI security risk specified in self.endpoint, raises either
        LakeraGuardError or LakeraGuardWarning depending on self.raise_error True or
        False. Otherwise, lets input through.

        Args:
            prompt: input to check regarding AI security risk
        Returns:
            prompt unchanged
        """
        formatted_input = self._convert_to_lakera_guard_input(prompt)

        lakera_guard_response = self._call_lakera_guard(formatted_input)

        if lakera_guard_response["results"][0]["flagged"]:
            if self.raise_error:
                raise LakeraGuardError(
                    f"Lakera Guard detected {self.endpoint}.", lakera_guard_response
                )
            else:
                warnings.warn(
                    LakeraGuardWarning(
                        f"Lakera Guard detected {self.endpoint}.",
                        lakera_guard_response,
                    )
                )

        return prompt

    def detect_with_response(self, prompt: GuardInput) -> dict:
        """
        Returns detection result of AI security risk specified in self.endpoint
        with regard to the input.

        Args:
            input: input to check regarding AI security risk
        Returns:
            detection result of AI security risk specified in self.endpoint
        """
        formatted_input = self._convert_to_lakera_guard_input(prompt)

        lakera_guard_response = self._call_lakera_guard(formatted_input)

        return lakera_guard_response

    def get_guarded_llm(self, type_of_llm: Type[BaseLLMT]) -> Type[BaseLLMT]:
        """
        Creates a subclass of type_of_llm where the input to the LLM always gets
        checked w.r.t. AI security risk specified in self.endpoint.

        Args:
            type_of_llm: any type of LangChain's LLMs
        Returns:
            Guarded subclass of type_of_llm
        """
        lakera_guard_instance = self

        class GuardedLLM(type_of_llm):  # type: ignore
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
        self, type_of_chat_llm: Type[BaseChatModelT]
    ) -> Type[BaseChatModelT]:
        """
        Creates a subclass of type_of_chat_llm in which the input to the ChatLLM always
          gets checked w.r.t. AI security risk specified in self.endpoint.

        Args:
            type_of_llm: any type of LangChain's ChatLLMs
        Returns:
            Guarded subclass of type_of_llm
        """
        lakera_guard_instance = self

        class GuardedChatLLM(type_of_chat_llm):  # type: ignore
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
        in self.endpoint.

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
                run_manager: CallbackManagerForChainRun | None = None,
            ) -> Union[AgentFinish, List[Tuple[AgentAction, str]]]:
                for val in inputs.values():
                    lakera_guard_instance.detect(val)

                res = super()._take_next_step(
                    name_to_tool_map,
                    color_mapping,
                    inputs,
                    intermediate_steps,
                    run_manager,
                )

                for act in intermediate_steps:
                    lakera_guard_instance.detect(act[1])

                return res

        return GuardedAgentExecutor
