"""
CompactifAI chat completion transformation
"""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final

import httpx
from pydantic import ConfigDict, TypeAdapter, with_config
from typing_extensions import ReadOnly, Required, TypedDict

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.openai.common_utils import OpenAIError
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse

from ...openai.chat.gpt_transformation import OpenAIGPTConfig

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj
    from litellm.litellm_core_utils.tokenizer import Encoding as Tokenizer

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


@with_config(ConfigDict(extra="allow", strict=True))
class CompactifAIResponseFields(TypedDict, total=False):
    """The chat completion fields of a CompactifAI response body."""

    id: ReadOnly[str]
    choices: ReadOnly[Sequence[Mapping[str, object]]]
    created: ReadOnly[int]
    model: ReadOnly[str]
    system_fingerprint: ReadOnly[str | None]
    usage: ReadOnly[Mapping[str, object]]
    object: ReadOnly[str]


@with_config(ConfigDict(extra="allow", strict=True))
class _CompactifAIToolFunction(TypedDict, total=False):
    arguments: ReadOnly[object]


@with_config(ConfigDict(extra="allow", strict=True))
class _CompactifAIToolCall(TypedDict):
    function: Required[ReadOnly[_CompactifAIToolFunction]]


@with_config(ConfigDict(extra="allow", strict=True))
class _CompactifAIMessage(TypedDict, total=False):
    tool_calls: ReadOnly[Sequence[object] | None]


@with_config(ConfigDict(extra="allow", strict=True))
class _CompactifAIChoice(TypedDict, total=False):
    message: ReadOnly[_CompactifAIMessage | None]


@with_config(ConfigDict(extra="allow", strict=True))
class _CompactifAIJsonModeResponseFields(CompactifAIResponseFields, total=False):
    choices: Required[ReadOnly[Sequence[_CompactifAIChoice]]]


_COMPACTIFAI_RESPONSE_ADAPTER: Final = TypeAdapter(CompactifAIResponseFields)
_COMPACTIFAI_JSON_MODE_RESPONSE_ADAPTER: Final = TypeAdapter(_CompactifAIJsonModeResponseFields)
_COMPACTIFAI_TOOL_CALL_ADAPTER: Final = TypeAdapter(_CompactifAIToolCall)


def _transform_json_mode_choice(choice: _CompactifAIChoice) -> Mapping[str, object]:
    message: Final = choice.get("message")
    if not message:
        return choice

    tool_calls: Final = message.get("tool_calls", [])
    if not tool_calls or len(tool_calls) != 1:
        return choice

    tool_call: Final = _COMPACTIFAI_TOOL_CALL_ADAPTER.validate_python(tool_calls[0])
    # Convert tool calls to content for JSON mode
    return {
        **choice,
        "message": {
            **message,
            "content": tool_call["function"].get("arguments", ""),
            "tool_calls": None,
        },
    }


def _validated_response_fields(
    raw_response: httpx.Response,
    json_mode: bool | None,
) -> CompactifAIResponseFields:
    if not json_mode:
        return _COMPACTIFAI_RESPONSE_ADAPTER.validate_python(raw_response.json())

    response_fields: Final = _COMPACTIFAI_JSON_MODE_RESPONSE_ADAPTER.validate_python(raw_response.json())
    choices: Final = [_transform_json_mode_choice(choice) for choice in response_fields["choices"]]
    return {**response_fields, "choices": choices}


class CompactifAIChatConfig(OpenAIGPTConfig):
    """
    Configuration class for CompactifAI chat completions.
    Since CompactifAI is OpenAI-compatible, we extend OpenAIGPTConfig.
    """

    def _get_openai_compatible_provider_info(
        self,
        api_base: str | None,
        api_key: str | None,
    ) -> tuple[str | None, str | None]:
        """
        Get API base and key for CompactifAI provider.
        """
        api_base = api_base or "https://api.compactif.ai/v1"
        dynamic_api_key: Final = api_key or get_secret_str("COMPACTIFAI_API_KEY") or ""
        return api_base, dynamic_api_key

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: Mapping[str, object],
        messages: Sequence[AllMessageValues],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        encoding: "Tokenizer | None",
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ModelResponse:
        """
        Transform CompactifAI response to LiteLLM format.
        Since CompactifAI is OpenAI-compatible, we can use the standard OpenAI transformation.
        """
        ## LOGGING
        logging_obj.post_call(
            input=messages,
            api_key=api_key,
            original_response=raw_response.text,
            additional_args={"complete_input_dict": request_data},
        )

        ## RESPONSE OBJECT

        # Handle JSON mode if needed
        response_fields: Final = _validated_response_fields(raw_response, json_mode)

        returned_response: Final = ModelResponse(**response_fields)

        # Set model name with provider prefix
        returned_response.model = f"compactifai/{model}"

        return returned_response

    def get_error_class(
        self, error_message: str, status_code: int, headers: dict[str, str] | httpx.Headers
    ) -> BaseLLMException:
        """
        Get the appropriate error class for CompactifAI errors.
        Since CompactifAI is OpenAI-compatible, we use OpenAI error handling.
        """
        return OpenAIError(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )
