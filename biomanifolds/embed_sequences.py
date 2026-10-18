"""ESM-2 embeddings of every residue of a list of protein sequences, with Hugging Face transformers.

The embedding of a residue is the output of layer `layer` (the residual stream, before the final
layer norm), computed in float32 and saved as float16. Row r of the output .npy is residue r of
the concatenated sequences, in the given order.

Also runs as a script, e.g. on a GPU machine with only numpy, polars, torch and transformers:

    python embed_sequences.py proteins.parquet embeddings.npy --layer 18

where proteins.parquet has a `sequence` column.
"""

import numpy as np

MODEL = "facebook/esm2_t30_150M_UR50D"


def embed_sequences(sequences, out_path, layer=18, max_tokens=16384, device=None):
    import torch
    from transformers import AutoTokenizer, EsmModel

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = EsmModel.from_pretrained(MODEL, add_pooling_layer=False, attn_implementation="sdpa").eval().to(device)
    model.encoder.layer = model.encoder.layer[:layer]  # run layers 1..layer only
    model.encoder.emb_layer_norm_after = None  # keep the residual stream: no final layer norm

    lengths = np.array([len(s) for s in sequences])
    starts = np.r_[0, np.cumsum(lengths)]  # rows of sequence j: starts[j] .. starts[j + 1]
    tmp_path = out_path.with_name(out_path.stem + ".tmp.npy")
    out = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=np.float16, shape=(int(starts[-1]), model.config.hidden_size))

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
            x = model(**tokens).last_hidden_state.half().cpu().numpy()  # (batch, tokens, hidden)
            for b, j in enumerate(batch):
                out[starts[j]:starts[j + 1]] = x[b, 1:lengths[j] + 1]  # token 0 is <cls>
            if k % 100 == 0:
                print(f"batch {k}/{len(batches)}", flush=True)
    out.flush()
    del out
    tmp_path.rename(out_path)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    import polars as pl

    parser = argparse.ArgumentParser()
    parser.add_argument("proteins", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--max-tokens", type=int, default=16384)
    args = parser.parse_args()
    embed_sequences(pl.read_parquet(args.proteins)["sequence"].to_list(), args.out, args.layer, args.max_tokens)
