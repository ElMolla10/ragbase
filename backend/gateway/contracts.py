"""Version-one text gateway contracts and explicit capability failures."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal, Protocol

from backend.policy import Instructions


class GatewayError(ValueError):
    """A normalized provider or request failure."""


class UnsupportedCapabilityError(GatewayError):
    """A requested capability is not validated for this adapter."""


@dataclass(frozen=True)
class GatewayRequest:
    instructions: Instructions
    question: str
    evidence: tuple[str, ...] = ()
    max_output_tokens: int = 512
    images: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()

    def validate(self) -> None:
        if not self.question.strip() or not self.instructions.text.strip():
            raise GatewayError("Question and policy instructions are required")
        if not 1 <= self.max_output_tokens <= 4096:
            raise GatewayError("Output budget must be between 1 and 4096")
        # A byte ceiling for the scaffold, NOT model tokenizer/context validation.
        if (
            sum(len(value.encode("utf-8")) for value in (self.question, *self.evidence))
            > 65536
        ):
            raise GatewayError("Development request exceeds byte budget")
        if self.images or self.tools:
            raise UnsupportedCapabilityError(
                "Images and tools are not enabled in this text scaffold"
            )


@dataclass(frozen=True)
class GatewayEvent:
    kind: Literal["text", "done"]
    text: str = ""
    finish_reason: str | None = None
    # None means unavailable, never zero or an invented estimate.
    usage: dict[str, int] | None = None
    # Request-local routing metadata, never shared mutable adapter state.
    provenance: dict[str, str | None] | None = None


class ModelGateway(Protocol):
    def stream(self, request: GatewayRequest) -> AsyncIterator[GatewayEvent]: ...
