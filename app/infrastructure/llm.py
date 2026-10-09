"""Chat model client: connects to the local llama3 model served by Ollama."""

from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama


def create_chat_model(
    model: str,
    base_url: str,
    temperature: float,
    num_ctx: int,
    num_gpu: int | None,
    timeout_seconds: float,
) -> BaseChatModel:
    """Build the chat model. Pass num_gpu=0 to force CPU, or None to let Ollama decide."""
    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=temperature,
        num_ctx=num_ctx,
        num_gpu=num_gpu,
        client_kwargs={"timeout": timeout_seconds},
    )