"""LLM provider abstractions and concrete model builders."""

from abc import ABC, abstractmethod
from typing import Any
from urllib.parse import urlsplit

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from app.core.config import settings
from app.core.exceptions import ExternalServiceError


def _settings_value(name: str, default: Any = None) -> Any:
    return getattr(settings, name, default)


class LLMProvider(ABC):
    name: str

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Return the default model for this provider."""

    @property
    @abstractmethod
    def available_models(self) -> list[str]:
        """Return the list of available models for this provider."""

    @property
    @abstractmethod
    def api_key(self) -> str | None:
        """Return the API key for this provider, if applicable."""

    @abstractmethod
    def build_model(self, model_name: str | None = None, **kwargs: Any) -> BaseChatModel:
        """Build a model instance. Extra kwargs (e.g. temperature) pass through to the client."""


class OllamaProvider(LLMProvider):
    name = "Ollama"

    @property
    def default_model(self) -> str:  # type: ignore[override]
        return _settings_value("OLLAMA_MODEL", "gpt-oss:20b")

    @property
    def available_models(self) -> list[str]:
        return list(_settings_value("OLLAMA_AVAILABLE_MODELS", [self.default_model]))

    @property
    def api_key(self) -> str | None:
        return _settings_value("OLLAMA_API_KEY")

    def build_model(self, model_name: str | None = None, **kwargs: Any) -> BaseChatModel:
        client_kwargs = {}
        if self.api_key:
            client_kwargs["headers"] = {"Authorization": f"Bearer {self.api_key}"}

        return ChatOllama(
            model=model_name or self.default_model,
            base_url=_settings_value("OLLAMA_BASE_URL"),
            client_kwargs=client_kwargs if client_kwargs else None,
            **kwargs,
        )


class OpenAIProvider(LLMProvider):
    name = "OpenAI"

    @property
    def default_model(self) -> str:
        return _settings_value("OPENAI_MODEL")

    @property
    def available_models(self) -> list[str]:
        return list(_settings_value("OPENAI_AVAILABLE_MODELS", [self.default_model]))

    @property
    def api_key(self) -> str | None:
        return _settings_value("OPENAI_API_KEY")

    def build_model(self, model_name: str | None = None, **kwargs: Any) -> BaseChatModel:
        if not self.api_key or not self.api_key.strip():
            raise ExternalServiceError(
                message="请在 backend/.env 中填写 OPENAI_API_KEY（使用 DeepSeek 时填写 DeepSeek API key），然后重启后端。",
                details={"setting": "OPENAI_API_KEY"},
            )
        base_url = _settings_value("OPENAI_BASE_URL")
        if base_url and urlsplit(base_url).hostname == "api.deepseek.com":
            # Start with the documented non-thinking tool-call protocol. Thinking
            # mode requires preserving reasoning_content across tool turns.
            extra_body = dict(kwargs.get("extra_body") or {})
            extra_body.setdefault("thinking", {"type": "disabled"})
            kwargs["extra_body"] = extra_body
            kwargs.setdefault("use_responses_api", False)
        return ChatOpenAI(
            model=model_name or self.default_model,
            base_url=base_url,
            api_key=SecretStr(self.api_key) if self.api_key else None,
            **kwargs,
        )


class BedrockProvider(LLMProvider):
    """AWS Bedrock（Converse API）。目前用于评估的中立裁判（Claude）。

    凭证不在这里读——boto3 默认链解析（env AWS_ACCESS_KEY_ID/SECRET/SESSION_TOKEN、
    ~/.aws、或 IAM role）。模型 id 用 inference profile，如 us.anthropic.claude-...
    """

    name = "Bedrock"

    @property
    def default_model(self) -> str:
        return _settings_value("BEDROCK_MODEL") or ""

    @property
    def available_models(self) -> list[str]:
        return list(_settings_value("BEDROCK_AVAILABLE_MODELS", []))

    @property
    def api_key(self) -> str | None:
        return None

    def build_model(self, model_name: str | None = None, **kwargs: Any) -> BaseChatModel:
        # 懒加载：没装 langchain-aws 的环境不因模块导入而挂（只有真用 bedrock 才需要）
        from langchain_aws import ChatBedrockConverse

        return ChatBedrockConverse(
            model=model_name or self.default_model,
            region_name=_settings_value("BEDROCK_REGION", "us-east-1"),
            **kwargs,
        )


_PROVIDERS: dict[str, type[LLMProvider]] = {
    "ollama": OllamaProvider,
    "openai": OpenAIProvider,
    "bedrock": BedrockProvider,
}


def get_llm_provider(name: str | None = None) -> LLMProvider:
    """Get an LLM provider. `name` overrides LLM_PROVIDER (used by the eval judge to pick a
    neutral provider); None falls back to settings.LLM_PROVIDER."""
    key = (name or _settings_value("LLM_PROVIDER", "ollama")).lower()
    provider_cls = _PROVIDERS.get(key)
    if provider_cls is None:
        raise ValueError(f"Unsupported LLM provider: {key!r}")
    return provider_cls()
