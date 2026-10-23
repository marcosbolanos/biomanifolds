"""ESM-2 embeddings of every residue of a list of protein sequences, with Hugging Face transformers.

The embedding of a residue at layer L is the output of layer L (the residual stream, before the
final layer norm), computed in float32 and saved as float16. Row r of each output .npy is residue
r of the concatenated sequences, in the given order (with `positions`: only the given residues of
each sequence, in the given order). Several layers are computed in one pass.

Also runs as a script, e.g. on a GPU machine with only numpy, polars, torch and transformers:

    python embed_sequences.py proteins.parquet embeddings_L{layer}.npy --layers 18 28 30

where proteins.parquet has a `sequence` column (and optionally a `positions` column: list of the
residues to keep) and {layer} in the output name is replaced by each layer.
"""

import numpy as np

MODEL = "facebook/esm2_t30_150M_UR50D"


def embed_sequences(sequences, out_paths, max_tokens=16384, device=None, positions=None):
    """`out_paths`: {layer: path of the .npy for that layer}. `positions`: for each sequence, the 0-based
    residues to keep (default: all)."""
    import torch
    from transformers import AutoTokenizer, EsmModel

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = EsmModel.from_pretrained(MODEL, add_pooling_layer=False, attn_implementation="sdpa").eval().to(device)
    model.encoder.layer = model.encoder.layer[:max(out_paths)]  # run layers 1..max layer only
    model.encoder.emb_layer_norm_after = None  # keep the residual stream: no final layer norm
    captured = {}
    for layer in out_paths:  # read-only hooks: store the output of every requested layer
        model.encoder.layer[layer - 1].register_forward_hook(
            lambda _module, _input, output, layer=layer: captured.__setitem__(layer, output[0] if isinstance(output, tuple) else output))

    lengths = np.array([len(s) for s in sequences])
    kept = [np.arange(n) for n in lengths] if positions is None else [np.asarray(p, dtype=np.int64) for p in positions]
    starts = np.r_[0, np.cumsum([len(k) for k in kept])]  # rows of sequence j: starts[j] .. starts[j + 1]
    tmp_paths = {layer: path.with_name(path.stem + ".tmp.npy") for layer, path in out_paths.items()}
    outs = {layer: np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(int(starts[-1]), model.config.hidden_size))
            for layer, tmp in tmp_paths.items()}

    # batches of similar lengths (little padding) with at most max_tokens tokens, <cls> and <eos> included
    order, batch = np.argsort(lengths), []
    batches = []
    for j in order:
        if batch and (len(batch) + 1) * (lengths[j] + 2) > max_tokens:
            batches.append(batch)
            batch = []
        batch.append(j)
    batches.append(batch)

    with torch.no_grad():
        for k, batch in enumerate(batches):
            tokens = tokenizer([sequences[j] for j in batch], return_tensors="pt", padding=True).to(device)
            model(**tokens)
            for layer, out in outs.items():
                x = captured[layer].half().cpu().numpy()  # (batch, tokens, hidden)
                for b, j in enumerate(batch):
                    out[starts[j]:starts[j + 1]] = x[b, kept[j] + 1]  # token 0 is <cls>
            if k % 100 == 0:
                print(f"batch {k}/{len(batches)}", flush=True)
    for layer, out in outs.items():
        out.flush()
        del out
    outs.clear()
    for layer, tmp in tmp_paths.items():
        tmp.rename(out_paths[layer])


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    import polars as pl

    parser = argparse.ArgumentParser()
    parser.add_argument("proteins", type=Path)
    parser.add_argument("out", help="output path with {layer}, e.g. embeddings_L{layer}.npy")
    parser.add_argument("--layers", type=int, nargs="+", default=[18])
    parser.add_argument("--max-tokens", type=int, default=16384)
    args = parser.parse_args()
    proteins = pl.read_parquet(args.proteins)
    embed_sequences(proteins["sequence"].to_list(), {layer: Path(args.out.format(layer=layer)) for layer in args.layers}, args.max_tokens,
                    positions=proteins["positions"].to_list() if "positions" in proteins.columns else None)
