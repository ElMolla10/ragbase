"""Allowlisted calculation plans. Free SQL and invented formulas are rejected."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.analysis.limits import (
    MAX_AGGREGATES,
    MAX_FILTERS,
    MAX_GROUPS,
    MAX_IN_VALUES,
    MAX_PROJECT,
    MAX_SORTS,
    MAX_STEPS,
)
from backend.analysis.profile import (
    CsvDataset,
    DatasetError,
    is_margin_name,
    mentions_margin,
)

_OUTPUT = re.compile(r"^[\w][\w ._\-]{0,79}$", re.UNICODE)
_CONTROL = re.compile(r"[\x00-\x1f]")
_CAUSAL = re.compile(
    r"forecast|predict|causation|causal|why did|توقع|تنبؤ|سبب|لماذا",
    re.IGNORECASE,
)


class FilterSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    column: str = Field(min_length=1, max_length=200)
    operator: Literal[
        "eq",
        "neq",
        "gt",
        "gte",
        "lt",
        "lte",
        "contains",
        "in",
        "is_null",
        "is_not_null",
    ]
    value: str | None = Field(default=None, max_length=200)
    values: list[str] | None = Field(default=None, max_length=MAX_IN_VALUES)

    @field_validator("column", "value")
    @classmethod
    def plain(cls, value):
        return _plain(value)

    @field_validator("values")
    @classmethod
    def plain_values(cls, values):
        if values is None:
            return None
        return [_plain(item, minimum=1) for item in values]


class AggregateSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    function: Literal["sum", "count", "count_distinct", "min", "max", "avg"]
    column: str | None = Field(default=None, max_length=200)
    output: str = Field(min_length=1, max_length=80)

    @field_validator("column")
    @classmethod
    def plain_column(cls, value):
        return _plain(value)

    @field_validator("output")
    @classmethod
    def plain_output(cls, value):
        value = _plain(value, minimum=1)
        if value is None or not _OUTPUT.fullmatch(value):
            raise ValueError("invalid output name")
        return value


class SortSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    column: str = Field(min_length=1, max_length=200)
    direction: Literal["asc", "desc"] = "asc"

    @field_validator("column")
    @classmethod
    def plain(cls, value):
        return _plain(value, minimum=1)


class OperationPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    action: Literal["calculate", "clarify", "not_a_calculation"] = "calculate"
    filters: list[FilterSpec] = Field(default_factory=list, max_length=MAX_FILTERS)
    group_by: list[str] = Field(default_factory=list, max_length=MAX_GROUPS)
    aggregates: list[AggregateSpec] = Field(
        default_factory=list, max_length=MAX_AGGREGATES
    )
    sort: list[SortSpec] = Field(default_factory=list, max_length=MAX_SORTS)
    limit: int | None = Field(default=None, ge=1, le=100)
    project: list[str] = Field(default_factory=list, max_length=MAX_PROJECT)
    metric_label: str | None = Field(default=None, max_length=80)

    @field_validator("group_by", "project")
    @classmethod
    def plain_names(cls, values):
        return [_plain(item, minimum=1) for item in values]

    @field_validator("metric_label")
    @classmethod
    def plain_label(cls, value):
        return _plain(value)

    @model_validator(mode="after")
    def complete(self):
        if self.action != "calculate":
            return self
        for item in self.filters:
            if item.operator == "in" and not item.values:
                raise ValueError("invalid filter")
            if item.operator in {"is_null", "is_not_null"} and (
                item.value is not None or item.values is not None
            ):
                raise ValueError("invalid filter")
            if (
                item.operator not in {"in", "is_null", "is_not_null"}
                and item.value is None
            ):
                raise ValueError("invalid filter")
        outputs = [item.output for item in self.aggregates]
        if len(outputs) != len(set(outputs)):
            raise ValueError("duplicate aggregate name")
        if len({name.casefold() for name in self.group_by}) != len(self.group_by):
            raise ValueError("duplicate group")
        steps = (
            len(self.filters)
            + (1 if self.group_by else 0)
            + len(self.aggregates)
            + len(self.sort)
            + (1 if self.limit else 0)
            + (1 if self.project else 0)
        )
        if steps > MAX_STEPS or steps < 1:
            raise ValueError("step limit")
        if not self.aggregates and not self.project and not self.group_by:
            raise ValueError("empty calculation")
        return self


def _plain(value, *, minimum=0):
    if value is None:
        return None
    if not isinstance(value, str) or _CONTROL.search(value):
        raise ValueError("invalid text")
    if len(value) < minimum:
        raise ValueError("invalid text")
    return value


def interpretation_limited(question: str | None, plan: OperationPlan | None) -> bool:
    parts = [question or ""]
    if plan is not None:
        parts.append(plan.metric_label or "")
        parts.extend(item.output for item in plan.aggregates)
    return bool(_CAUSAL.search("\n".join(parts)))


def margin_ambiguous(
    question: str | None, plan: OperationPlan, dataset: CsvDataset
) -> bool:
    texts = [question or "", plan.metric_label or ""]
    texts.extend(item.output for item in plan.aggregates)
    if not any(mentions_margin(text) for text in texts):
        return False
    used = []
    for item in plan.aggregates:
        if item.column:
            used.append(dataset.column(item.column))
    if used and all(is_margin_name(column.name) for column in used):
        return False
    if not plan.aggregates and plan.project:
        projected = [dataset.column(name) for name in plan.project]
        if projected and all(is_margin_name(column.name) for column in projected):
            return False
    return True


def bind_plan(plan: OperationPlan, dataset: CsvDataset) -> OperationPlan:
    """Resolve names now so a missing column fails before DuckDB starts."""
    if plan.action != "calculate":
        return plan
    for name in [*plan.group_by, *plan.project]:
        dataset.column(name)
    grouped = {dataset.column(name).name for name in plan.group_by}
    for item in plan.aggregates:
        if item.function == "count" and item.column is None:
            continue
        if item.column is None:
            raise DatasetError("invalid_plan")
        column = dataset.column(item.column)
        if item.function in {"sum", "avg", "min", "max"} and column.type != "decimal":
            raise DatasetError("column_not_numeric")
    if plan.aggregates and plan.project:
        projected = {dataset.column(name).name for name in plan.project}
        if not projected.issubset(grouped):
            raise DatasetError("invalid_plan")
    outputs = {item.output for item in plan.aggregates}
    for item in plan.sort:
        if item.column in outputs:
            continue
        column = dataset.column(item.column)
        if plan.aggregates and column.name not in grouped:
            raise DatasetError("invalid_plan")
    for item in plan.filters:
        column = dataset.column(item.column)
        if item.operator in {"gt", "gte", "lt", "lte"} and column.type != "decimal":
            raise DatasetError("column_not_numeric")
        if item.operator == "contains" and column.type == "decimal":
            raise DatasetError("invalid_plan")
    return plan
