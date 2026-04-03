from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any, Type
import asyncio

import litellm
import numpy as np
from json_repair import repair_json
from pydantic import BaseModel

from fast_graphrag._llm._base import BaseEmbeddingService, BaseLLMService, T_model
from fast_graphrag._models import BaseModelAlias


JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json_text(content: str) -> str:
    fenced = JSON_BLOCK_PATTERN.search(content)
    if fenced:
        return fenced.group(1).strip()
    return content.strip()


def validate_response_model(response_model: Type[T_model], content: str) -> T_model:
    cleaned = extract_json_text(content)
    repaired = repair_json(cleaned)

    if issubclass(response_model, BaseModelAlias):
        parsed = response_model.Model.model_validate_json(repaired)
        return parsed.to_dataclass(parsed)

    return response_model.model_validate_json(repaired)


@dataclass
class LiteLLMService(BaseLLMService):
    temperature: float = field(default=0.0)

    async def send_message(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict[str, str]] | None = None,
        response_model: Type[T_model] | None = None,
        **kwargs: Any,
    ) -> tuple[T_model, list[dict[str, str]]]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history_messages:
            messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})

        response = await litellm.acompletion(
            model=self.model,
            messages=messages,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=kwargs.pop("temperature", self.temperature),
            **kwargs,
        )

        content = response.choices[0].message.content or ""
        if response_model is None:
            parsed_response = content
        else:
            parsed_response = validate_response_model(response_model, content)

        updated_history = messages + [{"role": "assistant", "content": content}]
        return parsed_response, updated_history


@dataclass
class LiteLLMEmbeddingService(BaseEmbeddingService):
    # Gemini BatchEmbedContentsRequest allows at most 100 items per call.
    embedding_batch_size: int = field(default=100)

    async def _encode_batch(self, batch: list[str], model: str) -> list[list[float]]:
        response = await litellm.aembedding(
            model=model,
            input=batch,
            api_key=self.api_key,
            api_base=self.base_url or None,
        )
        vectors = []
        for item in response.data:
            if isinstance(item, dict):
                vectors.append(item["embedding"])
            else:
                vectors.append(item.embedding)
        return vectors

    async def encode(
        self,
        texts: list[str],
        model: str | None = None,
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        resolved_model = model or self.model
        batches = [
            texts[i : i + self.embedding_batch_size]
            for i in range(0, len(texts), self.embedding_batch_size)
        ]
        results = await asyncio.gather(
            *[self._encode_batch(b, resolved_model) for b in batches]
        )
        all_vectors = [v for batch_vectors in results for v in batch_vectors]
        return np.array(all_vectors, dtype=np.float32)
