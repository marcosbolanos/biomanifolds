"""Label metadata shared by every label type.

Each label module keeps its own `LABELS` dict and registers its label functions with
`label = label_metadata.decorator(LABELS)`:

    @label(source="computed", kind="continuous")
    def rsa(s): ...
"""

import json
from dataclasses import dataclass
from typing import Callable, Literal, TypedDict

import polars as pl

# kind -> parquet type
KINDS = {
    "continuous": pl.Float32,
    "angle": pl.Float32,
    "count": pl.Int32,
    "binary": pl.Boolean,
    "categorical": pl.String,
}
LEVELS = ("residue", "protein")


@dataclass(frozen=True)
class Label:
    name: str
    fn: Callable
    source: str  # where the data comes from, e.g. "sequence", "swissprot", "afdb", "computed"
    kind: str  # one of KINDS; "angle" is continuous and periodic, in degrees in (-180, 180]
    level: str = "residue"  # "residue": one value per residue; "protein": one value per protein
    mask: str | None = None  # pLDDT mask, AF2 structure labels only

    @property
    def dtype(self):
        return KINDS[self.kind]

    @property
    def description(self):
        return (self.fn.__doc__ or "").strip().split("\n\n")[0].replace("\n", " ")


def decorator(registry: dict):
    """Return a `label(...)` decorator that registers functions in `registry`."""

    def label(source, kind, level="residue", mask=None):
        assert kind in KINDS and level in LEVELS

        def register(fn):
            registry[fn.__name__] = Label(fn.__name__, fn, source, kind, level, mask)
            return fn

        return register

    return label


# Schemas of the JSON `values` summary that `describe` gives for each kind. `label_type`
# names the structure of the dict.

class ContinuousSummary(TypedDict):  # continuous, count
    label_type: Literal["continuous"]
    min: float
    max: float
    mean: float


class PeriodicSummary(TypedDict):  # angle
    label_type: Literal["periodic"]
    min: float
    max: float
    circular_mean: float  # atan2(mean sin, mean cos), degrees


class BinarySummary(TypedDict):  # binary
    label_type: Literal["binary"]
    true_fraction: float


class Category(TypedDict):
    value: str
    fraction: float


class CategoricalSummary(TypedDict):  # categorical
    label_type: Literal["categorical"]
    n_categories: int
    top: list[Category]  # most frequent first, at most TOP_CATEGORIES


TOP_CATEGORIES = 5


def _round(x):
    return float(f"{x:.4g}")


def _summary(lab: Label, stats: dict, frame: pl.LazyFrame):
    n = lab.name
    if lab.kind == "categorical":
        counts = frame.select(pl.col(n).drop_nulls().value_counts(sort=True)).unnest(n).collect(engine="streaming")
        total = counts["count"].sum()
        top = [Category(value=str(v), fraction=_round(c / total)) for v, c in counts.head(TOP_CATEGORIES).iter_rows()]
        return CategoricalSummary(label_type="categorical", n_categories=len(counts), top=top)
    if lab.kind == "binary":
        return BinarySummary(label_type="binary", true_fraction=_round(stats[f"{n}:mean"]))
    if lab.kind == "angle":
        return PeriodicSummary(label_type="periodic", min=_round(stats[f"{n}:min"]), max=_round(stats[f"{n}:max"]),
                            circular_mean=_round(stats[f"{n}:circmean"]))
    return ContinuousSummary(label_type="continuous", min=_round(stats[f"{n}:min"]), max=_round(stats[f"{n}:max"]), mean=_round(stats[f"{n}:mean"]))


def describe(labels: dict, residues: pl.LazyFrame, proteins: pl.LazyFrame | None = None) -> pl.DataFrame:
    """One row per label: its metadata, the fraction of null values, and `values`, a JSON
    summary of the values whose schema depends on the kind (ContinuousSummary, PeriodicSummary,
    BinarySummary or CategoricalSummary)."""
    rows = []
    for level, frame in (("residue", residues), ("protein", proteins)):
        names = [name for name, lab in labels.items() if lab.level == level]
        if not names:
            continue
        numeric = [n for n in names if labels[n].kind != "categorical"]
        stats = frame.select(
            pl.len().alias("rows"),
            *(pl.col(n).null_count().alias(f"{n}:nulls") for n in names),
            *(pl.col(n).cast(pl.Float64).min().alias(f"{n}:min") for n in numeric),
            *(pl.col(n).cast(pl.Float64).max().alias(f"{n}:max") for n in numeric),
            *(pl.col(n).cast(pl.Float64).mean().alias(f"{n}:mean") for n in numeric),
            *(pl.arctan2(pl.col(n).radians().sin().mean(), pl.col(n).radians().cos().mean()).degrees().alias(f"{n}:circmean")
              for n in names if labels[n].kind == "angle"),
        ).collect(engine="streaming").row(0, named=True)
        for n in names:
            lab = labels[n]
            rows.append({
                "label": n, "level": lab.level, "kind": lab.kind, "dtype": str(lab.dtype), "source": lab.source,
                "mask": lab.mask or "", "null_fraction": _round(stats[f"{n}:nulls"] / stats["rows"]),
                "values": json.dumps(_summary(lab, stats, frame)), "description": lab.description,
            })
    return pl.DataFrame(rows)
