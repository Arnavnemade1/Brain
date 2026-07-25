"""Shared Pydantic base.

Every model serialises to camelCase so the wire format matches the TypeScript
declarations in ``frontend/src/types`` without any client-side mapping layer.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


def to_camel(snake: str) -> str:
    head, *rest = snake.split("_")
    return head + "".join(word.capitalize() for word in rest)


class CamelModel(BaseModel):
    """Base model: camelCase aliases out, either casing accepted in."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        ser_json_inf_nan="constants",
        validate_assignment=False,
    )

    def wire(self) -> dict:
        """Serialise for transport (camelCase keys, JSON-safe values)."""
        return self.model_dump(by_alias=True, mode="json")
