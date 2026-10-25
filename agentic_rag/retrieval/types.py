"""The data model of Vertex AI Vector Search, as plain Python.

These mirror ``google.cloud.aiplatform_v1.IndexDatapoint`` and
``FindNeighborsRequest.Query`` field for field, so a backend can be swapped without
touching the code that builds datapoints and queries:

* :class:`Datapoint` - ``datapoint_id``, ``feature_vector`` (dense), ``sparse_embedding``,
  ``restricts`` (token namespaces), ``numeric_restricts``, ``crowding_tag``.
* :class:`Query` - one ``FindNeighborsRequest.Query``: a dense and/or sparse vector, the
  filters, ``neighbor_count``, ``per_crowding_attribute_neighbor_count``,
  ``approximate_neighbor_count`` and the ``rrf`` alpha of a hybrid query.
* :class:`Neighbor` - one ``FindNeighborsResponse.Neighbor``: the id, ``distance`` (dense)
  and ``sparse_distance``.

Semantics (from the Vector Search filtering docs): token restricts are an AND across
namespaces and an OR inside one; a deny token excludes any datapoint that has it; a query
namespace with no allow tokens matches everything; an empty or missing datapoint namespace
is *not* a wildcard. Numeric restricts on a datapoint carry a value, on a query a value and
an operator; every numeric restrict of a query must hold.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from dataclasses import field
from typing import Literal

NumericOp = Literal["LESS", "LESS_EQUAL", "EQUAL", "GREATER_EQUAL", "GREATER", "NOT_EQUAL"]
NUMERIC_OPS: tuple[NumericOp, ...] = (
    "LESS",
    "LESS_EQUAL",
    "EQUAL",
    "GREATER_EQUAL",
    "GREATER",
    "NOT_EQUAL",
)

#: Datapoint ids: what Vector Search accepts and what citations can safely quote.
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._#:/-]{0,199}$")
_NAMESPACE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")


class InvalidDatapointError(ValueError):
    """A datapoint or query that Vector Search would reject."""


def check_id(datapoint_id: str) -> str:
    if not _ID.match(datapoint_id):
        raise InvalidDatapointError(f"invalid datapoint id {datapoint_id!r}")
    return datapoint_id


def _check_namespace(name: str) -> None:
    if not _NAMESPACE.match(name):
        raise InvalidDatapointError(f"invalid namespace {name!r}")


@dataclass(frozen=True)
class SparseVector:
    """``IndexDatapoint.SparseEmbedding``: parallel ``dimensions`` and ``values``."""

    dimensions: tuple[int, ...] = ()
    values: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if len(self.dimensions) != len(self.values):
            raise InvalidDatapointError("sparse dimensions and values differ in length")
        if len(set(self.dimensions)) != len(self.dimensions):
            raise InvalidDatapointError("sparse dimensions must be unique")
        if any(d < 0 for d in self.dimensions):
            raise InvalidDatapointError("sparse dimensions must be non-negative")
        if any(not math.isfinite(v) for v in self.values):
            raise InvalidDatapointError("sparse values must be finite")

    def as_dict(self) -> dict[int, float]:
        return dict(zip(self.dimensions, self.values, strict=True))

    def dot(self, other: SparseVector) -> float:
        mine = self.as_dict()
        return sum(
            mine.get(d, 0.0) * v for d, v in zip(other.dimensions, other.values, strict=True)
        )

    def __bool__(self) -> bool:
        return bool(self.dimensions)


@dataclass(frozen=True)
class Restrict:
    """A token namespace (``IndexDatapoint.Restriction``): ``allow`` and ``deny`` tokens."""

    namespace: str
    allow: tuple[str, ...] = ()
    deny: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _check_namespace(self.namespace)


@dataclass(frozen=True)
class NumericRestrict:
    """``IndexDatapoint.NumericRestriction``. ``op`` is set on queries only, never on data."""

    namespace: str
    value_int: int | None = None
    value_float: float | None = None
    value_double: float | None = None
    op: NumericOp | None = None

    def __post_init__(self) -> None:
        _check_namespace(self.namespace)
        given = [v for v in (self.value_int, self.value_float, self.value_double) if v is not None]
        if len(given) != 1:
            raise InvalidDatapointError(
                f"numeric restrict {self.namespace!r} needs exactly one of value_int, "
                "value_float, value_double"
            )
        if self.op is not None and self.op not in NUMERIC_OPS:
            raise InvalidDatapointError(f"unknown numeric operator {self.op!r}")

    @property
    def value(self) -> float:
        for v in (self.value_int, self.value_float, self.value_double):
            if v is not None:
                return float(v)
        raise AssertionError("unreachable: validated in __post_init__")  # pragma: no cover

    @property
    def kind(self) -> str:
        if self.value_int is not None:
            return "int"
        return "float" if self.value_float is not None else "double"

    def holds_for(self, datapoint_value: float) -> bool:
        """Is ``datapoint_value <op> self.value`` true? (The query is the right-hand side.)"""
        v, q = datapoint_value, self.value
        return {
            "LESS": v < q,
            "LESS_EQUAL": v <= q,
            "EQUAL": v == q,
            "GREATER_EQUAL": v >= q,
            "GREATER": v > q,
            "NOT_EQUAL": v != q,
        }[self.op or "EQUAL"]


@dataclass(frozen=True)
class Datapoint:
    """One ``IndexDatapoint``: what an upsert sends."""

    datapoint_id: str
    feature_vector: tuple[float, ...] = ()
    sparse_embedding: SparseVector = field(default_factory=SparseVector)
    restricts: tuple[Restrict, ...] = ()
    numeric_restricts: tuple[NumericRestrict, ...] = ()
    crowding_tag: str | None = None

    def __post_init__(self) -> None:
        check_id(self.datapoint_id)
        if not self.feature_vector and not self.sparse_embedding:
            raise InvalidDatapointError(f"{self.datapoint_id}: needs a dense or sparse vector")
        if any(not math.isfinite(x) for x in self.feature_vector):
            raise InvalidDatapointError(f"{self.datapoint_id}: dense values must be finite")
        if any(r.deny for r in self.restricts):
            # Supported by Vertex, not used here; the local store refuses rather than guess.
            raise InvalidDatapointError(f"{self.datapoint_id}: datapoint deny lists unsupported")
        if any(n.op is not None for n in self.numeric_restricts):
            raise InvalidDatapointError(
                f"{self.datapoint_id}: numeric restricts on a datapoint must not have an op"
            )
        names = [n.namespace for n in self.numeric_restricts]
        if len(set(names)) != len(names):
            raise InvalidDatapointError(f"{self.datapoint_id}: one value per numeric namespace")

    def tokens(self, namespace: str) -> set[str]:
        return {t for r in self.restricts if r.namespace == namespace for t in r.allow}

    def numeric(self, namespace: str) -> NumericRestrict | None:
        return next((n for n in self.numeric_restricts if n.namespace == namespace), None)


@dataclass(frozen=True)
class Query:
    """One ``FindNeighborsRequest.Query``.

    ``rrf_alpha`` follows ``Query.RRF.alpha``: 1 means dense results only, 0 sparse only,
    anything in between merges both rankings with Reciprocal Rank Fusion. Leave it ``None``
    for a dense-only or sparse-only query (Vertex: an unset alpha uses the dense results).
    """

    feature_vector: tuple[float, ...] = ()
    sparse_embedding: SparseVector = field(default_factory=SparseVector)
    restricts: tuple[Restrict, ...] = ()
    numeric_restricts: tuple[NumericRestrict, ...] = ()
    neighbor_count: int = 10
    per_crowding_attribute_neighbor_count: int | None = None
    approximate_neighbor_count: int | None = None
    rrf_alpha: float | None = None

    def __post_init__(self) -> None:
        if not self.feature_vector and not self.sparse_embedding:
            raise InvalidDatapointError("a query needs a dense or a sparse vector")
        if not 1 <= self.neighbor_count <= 1000:
            raise InvalidDatapointError("neighbor_count must be between 1 and 1000")
        if self.rrf_alpha is not None and not 0.0 <= self.rrf_alpha <= 1.0:
            raise InvalidDatapointError("rrf alpha must be between 0 and 1")
        if any(n.op is None for n in self.numeric_restricts):
            raise InvalidDatapointError("numeric restricts on a query need an op")
        per = self.per_crowding_attribute_neighbor_count
        if per is not None and per < 1:
            raise InvalidDatapointError("per_crowding_attribute_neighbor_count must be >= 1")

    @property
    def mode(self) -> Literal["dense", "sparse", "hybrid"]:
        if self.feature_vector and self.sparse_embedding:
            if self.rrf_alpha is None or self.rrf_alpha >= 1.0:
                return "dense"
            return "sparse" if self.rrf_alpha <= 0.0 else "hybrid"
        return "dense" if self.feature_vector else "sparse"


@dataclass(frozen=True)
class Neighbor:
    """One match. ``distance``/``sparse_distance`` are similarities (higher is closer)."""

    datapoint_id: str
    distance: float | None = None
    sparse_distance: float | None = None
    crowding_tag: str | None = None


def matches(datapoint: Datapoint, query: Query) -> bool:
    """Does ``datapoint`` pass the query's token and numeric restricts (Vertex semantics)?"""
    for restrict in query.restricts:
        tokens = datapoint.tokens(restrict.namespace)
        if restrict.deny and tokens & set(restrict.deny):
            return False
        if restrict.allow and not tokens & set(restrict.allow):
            return False
    for numeric in query.numeric_restricts:
        value = datapoint.numeric(numeric.namespace)
        if value is None or value.kind != numeric.kind or not numeric.holds_for(value.value):
            return False
    return True
