"""Run a set of registered labels over many proteins and write them in parquet shards.

Every label type writes to its own directory:
  * part-NNNN.parquet: residue-level labels, one row per (accession, i), SHARD proteins per file;
  * proteins.parquet: protein-level labels, one row per accession (if the type has any);
  * _SUCCESS: written last, so a directory without it is incomplete.
"""

import numpy as np
import polars as pl

SHARD = 10_000  # proteins per residue-level file


def batches(items, size):
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def is_done(out_dir):
    return (out_dir / "_SUCCESS").exists()


def mark_done(out_dir):
    (out_dir / "_SUCCESS").touch()


def residue_frame(labels, accessions, values):
    """One frame from per-protein label arrays. `values[name]` is a list of arrays, one per
    protein, in the order of `accessions`."""
    lengths = [len(v) for v in values[next(iter(values))]]
    columns = {
        "accession": np.repeat(accessions, lengths),
        "i": np.concatenate([np.arange(n) for n in lengths]),
        **{name: np.concatenate(arrays) for name, arrays in values.items()},
    }
    df = pl.DataFrame(columns, nan_to_null=True)
    return df.select(
        pl.col("accession").cast(pl.String), pl.col("i").cast(pl.Int32),
        *(pl.col(name).cast(labels[name].dtype) for name in values),
    )


def run(labels, proteins, out_dir):
    """Compute every label in `labels` for every protein (objects with an `accession`) and
    write the shards to `out_dir`. Label functions take one protein and return an array with
    one value per residue ("residue" level) or one value ("protein" level)."""
    residue_labels = {n: lab for n, lab in labels.items() if lab.level == "residue"}
    protein_labels = {n: lab for n, lab in labels.items() if lab.level == "protein"}

    protein_rows = []
    for k, batch in enumerate(batches(proteins, SHARD)):
        values = {name: [np.asarray(lab.fn(p)) for p in batch] for name, lab in residue_labels.items()}
        residue_frame(labels, [p.accession for p in batch], values).write_parquet(out_dir / f"part-{k:04d}.parquet")
        protein_rows += [{"accession": p.accession, **{n: lab.fn(p) for n, lab in protein_labels.items()}} for p in batch]

    if protein_labels:
        schema = {"accession": pl.String, **{n: lab.dtype for n, lab in protein_labels.items()}}
        pl.DataFrame(protein_rows, schema=schema).write_parquet(out_dir / "proteins.parquet")
    mark_done(out_dir)
