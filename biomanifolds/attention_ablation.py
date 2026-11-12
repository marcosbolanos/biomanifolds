"""Mean ablation of attention heads of ESM-2 150M and the linear decodability of the helix face angle (notebook 10_pt2).

A. block_ablation(L): all the heads of block L mean-ablated; held-out ridge R^2_k of (cos k theta, sin k theta), k = 1..K, at every
   point from L.1 (after the attention of block L) to z (the vector read by the unembedding matrix). All the train proteins of
   notebook 9 (the same residues and folds).
B. head_ablation(L): no ablation, each head of block L, all its heads; R^2_k at L.1, L.2 and (L+1).2 (z for the last block).
   The 5000 proteins of notebook 10. The model runs once to block L per batch; blocks L and L+1 run again for each variant.
M. moment_ablation(L): as A (all the heads of block L mean-ablated, the proteins of A), with the targets cos theta, sin theta and the
   real and imaginary parts of the helical hydrophobic moment M_alpha (notebook 11_v2), at every point from L.1 to z; L = 0: no
   ablation, every point from emb, and also z without the dense layer and the GELU of the LM head. For each target: the held-out R^2
   of its own ridge decoder; for each pair (decoder, target): the squared correlation of the projection on the decoder with the
   target, on the held-out fold; the decoder and the pattern (encoder direction) of each target, fitted on all the folds.
   moment_ablation(L, "mlp"): the same with the output of the MLP of block L mean-ablated, at every point from L.2 to z.
   The logits of the 20 amino acids (AA order) at every helix residue are saved too (float16, in the order of the batches; the run
   without ablation also saves the protein index and the position of each row, and the logits without the dense layer and the GELU).

Mean ablation: the output of a head (its dimensions before the output matrix W_O of the attention) is replaced by its mean over
all the tokens (not the padding) of all the proteins; for an MLP, its output (before it is added to the residual stream). Each block
is saved in its own file (A_blockLL.npz, B_blockLL.npz, M_blockLL.npz, M_mlpLL.npz).
"""

import numpy as np
import polars as pl
import torch
from tqdm.auto import tqdm

from biomanifolds.angle_study import samples
from biomanifolds.angle_study.bins import row_bins
from biomanifolds.angle_study.constants import AA, EISENBERG
from biomanifolds.angle_study.models import Esm2

MODEL = "facebook/esm2_t30_150M_UR50D"
MAX_TOKENS = 16384  # tokens per forward pass, special tokens included
N_FOLDS, K = 5, 4  # folds of proteins; frequencies k = 1..K
ALPHA = 1e-4  # fixed ridge penalty, times the mean variance of the features
N_SUBSET = 5000  # proteins of experiment B (the sample of notebook 10)
WINDOW = 5  # the hydrophobic moment uses residues i - 5 .. i + 5
MOMENT_TARGETS = ["cos theta", "sin theta", "Re M alpha", "Im M alpha"]  # targets of experiment M
HYDROPHOBICITY = np.zeros(256)
for _a, _v in EISENBERG.items():
    HYDROPHOBICITY[ord(_a)] = _v  # other letters (X, U, ...): 0


def first(o):
    return o[0] if isinstance(o, tuple) else o


def stages(n_layers):
    """The points of the model and their names: emb, (L, 1) after the attention of block L, (L, 2) after its MLP, z."""
    points = [0] + [(layer, part) for layer in range(1, n_layers + 1) for part in (1, 2)] + ["z"]
    return points, ["emb"] + [f"{layer}.{part}" for layer, part in points[1:-1]] + ["z"]


def alpha_moment(sequence, positions):
    """Helical hydrophobic moment M(i) = sum_{|D| <= WINDOW} H(a_{i+D}) e^{i 100deg D} at the positions (Eisenberg scale; outside: 0)."""
    h = HYDROPHOBICITY[np.frombuffer(sequence.encode(), dtype=np.uint8)]
    padded = np.r_[np.zeros(WINDOW), h, np.zeros(WINDOW)]
    offsets = np.arange(-WINDOW, WINDOW + 1)
    return padded[np.asarray(positions)[:, None] + WINDOW + offsets[None, :]] @ np.exp(1j * np.deg2rad(100) * offsets)


def decoder_scores(xtx, xty, yty, sum_y, n):
    """For each target: the held-out R^2 of its own ridge decoder, pooled over the folds (targets,). For each pair (decoder d, target t):
    the squared correlation of the projection w_d^T h with target t on the held-out fold, the mean over the folds (decoders, targets)."""
    features = np.arange(1, xtx.shape[-1])
    total_xtx, total_xty, total_n = xtx.sum(0), xty.sum(0), n.sum()
    sse, sst, corr2 = np.zeros((N_FOLDS, xty.shape[-1])), np.zeros((N_FOLDS, xty.shape[-1])), np.zeros((xty.shape[-1], xty.shape[-1]))
    for f in range(N_FOLDS):
        a_xtx, a_xty, n_train = total_xtx - xtx[f], total_xty - xty[f], total_n - n[f]  # the other folds
        mean_x, mean_y = a_xtx[0, features] / n_train, a_xty[0] / n_train
        cxx = a_xtx[np.ix_(features, features)] / n_train - np.outer(mean_x, mean_x)
        cxy = a_xty[features] / n_train - np.outer(mean_x, mean_y)
        beta = np.linalg.solve(cxx + ALPHA * np.trace(cxx) / len(features) * np.eye(len(features)), cxy)
        b = np.vstack([mean_y - mean_x @ beta, beta])
        S, t, yy = xtx[f], xty[f], yty[f]
        sst[f] = yy - sum_y[f] ** 2 / n[f]
        sse[f] = yy - 2 * (b * t).sum(0) + (b * (S @ b)).sum(0)  # b^T S b for every target
        test_mean_x = S[0, features] / n[f]  # covariances of the held-out fold
        test_cxx = S[np.ix_(features, features)] / n[f] - np.outer(test_mean_x, test_mean_x)
        test_cxy = t[features] / n[f] - np.outer(test_mean_x, sum_y[f] / n[f])
        var_p = (beta * (test_cxx @ beta)).sum(0)
        corr2 += (beta.T @ test_cxy) ** 2 / (var_p[:, None] * (sst[f] / n[f])[None, :]) / N_FOLDS
    return 1 - sse.sum(0) / sst.sum(0), corr2


def full_vectors(xtx, xty, yty, sum_y, n):
    """The ridge decoder and the pattern Cov(h, y) / Var(y) of each target, fitted on all the folds: two (hidden, targets) arrays."""
    features, total_n = np.arange(1, xtx.shape[-1]), n.sum()
    total_xtx, total_xty = xtx.sum(0), xty.sum(0)
    mean_x, mean_y = total_xtx[0, features] / total_n, sum_y.sum(0) / total_n
    cxx = total_xtx[np.ix_(features, features)] / total_n - np.outer(mean_x, mean_x)
    cxy = total_xty[features] / total_n - np.outer(mean_x, mean_y)
    var_y = yty.sum(0) / total_n - mean_y**2
    decoder = np.linalg.solve(cxx + ALPHA * np.trace(cxx) / len(features) * np.eye(len(features)), cxy)
    return decoder, cxy / var_y


def r2_from_sums(xtx, xty, yty, sum_y, n, features, cos_cols, sin_cols):
    """Held-out R^2_k (k = 1..K) of (cos k theta, sin k theta), cos and sin together: pooled over the folds (K,) and per fold (folds, K);
    and the held-out R^2 of each target alone, pooled over the folds (cos columns, then sin columns: 2K,)."""
    idx, columns = np.r_[0, features], np.r_[cos_cols, sin_cols]
    total_xtx, total_xty, total_n = xtx.sum(0), xty.sum(0), n.sum()
    sse, sst = np.zeros((N_FOLDS, len(columns))), np.zeros((N_FOLDS, len(columns)))
    for f in range(N_FOLDS):
        a_xtx, a_xty, n_train = total_xtx - xtx[f], total_xty - xty[f], total_n - n[f]  # the other folds
        mean_x, mean_y = a_xtx[0, features] / n_train, a_xty[0, columns] / n_train
        cxx = a_xtx[np.ix_(features, features)] / n_train - np.outer(mean_x, mean_x)
        cxy = a_xty[np.ix_(features, columns)] / n_train - np.outer(mean_x, mean_y)
        beta = np.linalg.solve(cxx + ALPHA * np.trace(cxx) / len(features) * np.eye(len(features)), cxy)
        b = np.vstack([mean_y - mean_x @ beta, beta])
        S, t, yy = xtx[f][np.ix_(idx, idx)], xty[f][np.ix_(idx, columns)], yty[f][columns]
        sst[f] = yy - sum_y[f][columns] ** 2 / n[f]
        sse[f] = yy - 2 * (b * t).sum(0) + (b * (S @ b)).sum(0)  # b^T S b for every target
    half = len(cos_cols)
    pair = lambda e: e[:, :half] + e[:, half:]
    return 1 - pair(sse).sum(0) / pair(sst).sum(0), 1 - pair(sse) / pair(sst), 1 - sse.sum(0) / sst.sum(0)


class Sums:
    """Sums of the ridge fits for n_sets sets of features (one set: one point of one variant), per fold.
    Products in float32 on the GPU after the subtraction of a fixed reference mean (the fits have an intercept), sums in float64."""

    def __init__(self, n_sets, hidden, device, n_targets=2 * K):
        self.xtx = torch.zeros(n_sets, N_FOLDS, 1 + hidden, 1 + hidden, dtype=torch.float64, device=device)
        self.xty = torch.zeros(n_sets, N_FOLDS, 1 + hidden, n_targets, dtype=torch.float64, device=device)
        self.yty, self.sum_y, self.n = np.zeros((N_FOLDS, n_targets)), np.zeros((N_FOLDS, n_targets)), np.zeros(N_FOLDS)
        self.reference = {}

    def add_targets(self, y, masks):
        for f, m in enumerate(masks):
            self.yty[f] += (y[m].double() ** 2).sum(0).cpu().numpy()
            self.sum_y[f] += y[m].double().sum(0).cpu().numpy()
            self.n[f] += m.sum().item()

    def add(self, i, h, y, masks):
        h = h.float()
        if i not in self.reference:
            self.reference[i] = h.mean(0)
        x = torch.cat([torch.ones(len(h), 1, device=h.device), h - self.reference[i]], 1)
        for f, m in enumerate(masks):
            self.xtx[i, f] += (x[m].T @ x[m]).double()
            self.xty[i, f] += (x[m].T @ y[m]).double()

    def r2(self, i):
        hidden = self.xtx.shape[-1] - 1
        return r2_from_sums(self.xtx[i].cpu().numpy(), self.xty[i].cpu().numpy(), self.yty, self.sum_y, self.n,
                            np.arange(1, 1 + hidden), np.arange(K), np.arange(K, 2 * K))

    def decoders(self, i):
        xtx, xty = self.xtx[i].cpu().numpy(), self.xty[i].cpu().numpy()
        return decoder_scores(xtx, xty, self.yty, self.sum_y, self.n) + full_vectors(xtx, xty, self.yty, self.sum_y, self.n)


class Stop(Exception):
    pass


class Study:
    """The model, the proteins (all of notebook 9, and the 5000 of notebook 10), the head means, and the two experiments."""

    def __init__(self, out_dir, max_proteins=None):
        self.out_dir = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        self.model = Esm2(MODEL)
        self.n_layers, self.n_heads = self.model.n_layers, self.model.model.config.num_attention_heads
        self.head_dim = self.model.hidden_size // self.n_heads
        self.blocks = self.model.model.esm.encoder.layer
        self.stages, self.stage_names = stages(self.n_layers)

        # the train proteins of notebook 2 with helix residues (notebook 9), and the sample of 5000 of them (notebook 10)
        proteins = samples.train_proteins(0.10)
        rows = proteins.select("accession", i="positions").explode("i")
        residues = rows.with_columns(bin=row_bins(rows)["helix_face"]).filter(pl.col("bin") >= 0)
        proteins = proteins.join(residues.group_by("accession").agg(helix_i=pl.col("i"), helix_bin=pl.col("bin")), on="accession").sort("accession")
        subset = proteins.sample(N_SUBSET, seed=0).sort("accession")
        if max_proteins:
            proteins, subset = proteins.head(max_proteins), subset.head(max_proteins)
        self.data = {"all": self.prepare(proteins), "subset": self.prepare(subset)}
        self.head_means = torch.tensor(self.compute_head_means(), device=self.model.device)  # (blocks, hidden)
        self._mlp_means = None

    @staticmethod
    def prepare(df):
        """Sequences, helix residues and folds of the proteins, and batches of similar lengths with at most MAX_TOKENS tokens."""
        data = {"seqs": df["sequence"].to_list(), "h_i": df["helix_i"].to_list(), "h_b": df["helix_bin"].to_list(),
                "fold": (df["accession"].hash(seed=0) % N_FOLDS).to_numpy()}
        data["h_m"] = [alpha_moment(s, i) for s, i in zip(data["seqs"], data["h_i"])]  # M_alpha of the helix residues
        lengths = np.array([len(s) for s in data["seqs"]])
        data["batches"], batch = [], []
        for j in np.argsort(lengths):
            if batch and (len(batch) + 1) * (lengths[j] + 2) > MAX_TOKENS:
                data["batches"].append(batch)
                batch = []
            batch.append(j)
        data["batches"].append(batch)
        return data

    def helix_batch(self, d, batch):
        """Positions of the helix residues of a batch, their targets y (cos k theta, then sin k theta) and the fold masks."""
        device = self.model.device
        batch_index = torch.tensor(np.concatenate([np.full(len(d["h_i"][j]), b) for b, j in enumerate(batch)]), device=device)
        token_index = torch.tensor(np.concatenate([np.asarray(d["h_i"][j]) + self.model.offset for j in batch]), device=device)
        bins = np.concatenate([np.asarray(d["h_b"][j]) for j in batch])
        fold = np.concatenate([np.full(len(d["h_i"][j]), d["fold"][j]) for j in batch])
        theta = torch.tensor(np.deg2rad(bins - 179.5), dtype=torch.float32, device=device)
        ks = torch.arange(1, K + 1, device=device, dtype=torch.float32)
        y = torch.cat([torch.cos(theta[:, None] * ks), torch.sin(theta[:, None] * ks)], 1)
        return batch_index, token_index, y, [torch.tensor(fold == f, device=device) for f in range(N_FOLDS)]

    def moment_targets(self, d, batch):
        """As helix_batch, with the targets of experiment M: cos theta, sin theta, Re M_alpha, Im M_alpha."""
        batch_index, token_index, y, masks = self.helix_batch(d, batch)
        m = torch.tensor(np.concatenate([d["h_m"][j] for j in batch]), device=y.device)
        return batch_index, token_index, torch.stack([y[:, 0], y[:, K], m.real.float(), m.imag.float()], 1), masks

    def capture(self, points, captured):
        """Forward hooks that put the residual stream at the points (not z) in captured."""
        hooks = []
        for s in points:
            if s == 0:
                hooks.append(self.model.model.esm.embeddings.register_forward_hook(lambda m, i, o: captured.__setitem__(0, first(o))))
            elif s != "z":
                block = self.blocks[s[0] - 1]
                module = block.attention if s[1] == 1 else block  # the attention module adds its input (the residual stream) to its output
                hooks.append(module.register_forward_hook(lambda m, i, o, s=s: captured.__setitem__(s, first(o))))
        return hooks

    def to_z(self, h):
        """z from the output of the last block: the final layer norm, the dense layer, the exact GELU and the layer norm of the head."""
        head = self.model.model.lm_head
        x = head.dense(self.model.model.esm.encoder.emb_layer_norm_after(h))
        return head.layer_norm(x * 0.5 * (1 + torch.erf(x / np.sqrt(2))))

    def compute_head_means(self):
        """Mean of the input of W_O (the head outputs, concatenated) in every block, over all the tokens that are not padding."""
        path = self.out_dir / "head_means.npy"
        if path.exists():
            return np.load(path)
        d = self.data["all"]
        total = torch.zeros(self.n_layers, self.model.hidden_size, dtype=torch.float64, device=self.model.device)
        count, mask = 0, {}

        def accumulate(args, L):  # returns None: a pre-hook that returns a value would replace the input of W_O
            total[L] += args[0][mask["m"]].double().sum(0)

        hooks = [self.blocks[L].attention.output.dense.register_forward_pre_hook(lambda m, args, L=L: accumulate(args, L))
                 for L in range(self.n_layers)]
        try:
            for batch in tqdm(d["batches"], desc="head means", mininterval=60):
                input_ids, attention_mask = self.model.token_ids([d["seqs"][j] for j in batch])
                mask["m"] = attention_mask.bool()
                count += int(attention_mask.sum())
                with torch.no_grad():
                    self.model.model.esm(input_ids=input_ids, attention_mask=attention_mask)
        finally:
            for hook in hooks:
                hook.remove()
        means = (total / count).float().cpu().numpy()
        np.save(path, means)
        return means

    @property
    def mlp_means(self):
        """Mean of the MLP output (before it is added to the residual stream) in every block, over all the tokens that are not padding."""
        if self._mlp_means is None:
            path = self.out_dir / "mlp_means.npy"
            if not path.exists():
                d = self.data["all"]
                total = torch.zeros(self.n_layers, self.model.hidden_size, dtype=torch.float64, device=self.model.device)
                count, mask = 0, {}

                def accumulate(output, L):  # returns None: a forward hook that returns a value would replace the output
                    total[L] += output[mask["m"]].double().sum(0)

                hooks = [self.blocks[L].output.dense.register_forward_hook(lambda m, i, o, L=L: accumulate(o, L)) for L in range(self.n_layers)]
                try:
                    for batch in tqdm(d["batches"], desc="MLP means", mininterval=60):
                        input_ids, attention_mask = self.model.token_ids([d["seqs"][j] for j in batch])
                        mask["m"] = attention_mask.bool()
                        count += int(attention_mask.sum())
                        with torch.no_grad():
                            self.model.model.esm(input_ids=input_ids, attention_mask=attention_mask)
                finally:
                    for hook in hooks:
                        hook.remove()
                np.save(path, (total / count).float().cpu().numpy())
            self._mlp_means = torch.tensor(np.load(path), device=self.model.device)
        return self._mlp_means

    def ablate_mlp(self, L):
        """Forward hook on the output dense layer of the MLP of block L: its output becomes its mean (the residual is added after it)."""
        means = self.mlp_means[L - 1]
        return self.blocks[L - 1].output.dense.register_forward_hook(lambda m, i, o: means.expand_as(o).clone())

    def ablate(self, L, heads):
        """Pre-hook on W_O of block L: the heads (indices) get their mean output."""
        dims = torch.cat([torch.arange(h * self.head_dim, (h + 1) * self.head_dim) for h in heads]).to(self.model.device)

        def hook(module, args):
            x = args[0].clone()
            x[..., dims] = self.head_means[L - 1, dims]
            return (x,)

        return self.blocks[L - 1].attention.output.dense.register_forward_pre_hook(hook)

    def block_ablation(self, L):
        """A: R^2_k at every point from L.1 to z, with all the heads of block L mean-ablated (all the proteins).
        r2_cols: the R^2 of each target alone (cos k theta for k = 1..K, then sin k theta)."""
        path = self.out_dir / f"A_block{L:02d}.npz"
        if path.exists():
            result = dict(np.load(path))
            if "r2_cols" in result:  # older files have only the pooled R^2_k: compute them again
                return result
        d, start = self.data["all"], self.stages.index((L, 1))
        points = self.stages[start:]
        sums = Sums(len(points), self.model.hidden_size, self.model.device)
        captured = {}
        hooks = [self.ablate(L, range(self.n_heads))] + self.capture(points, captured)
        try:
            for batch in tqdm(d["batches"], desc=f"A: block {L}", mininterval=120):
                input_ids, attention_mask = self.model.token_ids([d["seqs"][j] for j in batch])
                with torch.no_grad():
                    self.model.model.esm(input_ids=input_ids, attention_mask=attention_mask)
                    captured["z"] = self.to_z(captured[(self.n_layers, 2)])
                batch_index, token_index, y, masks = self.helix_batch(d, batch)
                sums.add_targets(y, masks)
                for i, s in enumerate(points):
                    sums.add(i, captured[s][batch_index, token_index], y, masks)
        finally:
            for hook in hooks:
                hook.remove()
        fits = [sums.r2(i) for i in range(len(points))]
        result = {"stages": np.array(self.stage_names[start:]), "r2": np.array([f[0] for f in fits]), "r2_fold": np.array([f[1] for f in fits]),
                  "r2_cols": np.array([f[2] for f in fits])}
        np.savez(path, **result)
        del sums
        torch.cuda.empty_cache()
        return result

    def moment_ablation(self, L, part="attention"):
        """M: the decoder scores of MOMENT_TARGETS at every point after the ablation to z, with all the heads of block L (part
        "attention", from L.1) or its MLP ("mlp", from L.2) mean-ablated (all the proteins); L = 0: no ablation, every point from emb,
        and the extra set "z_no_dense" (z without the dense layer and the GELU of the LM head). r2: (points, targets); corr2: (points,
        decoders, targets); decoders, patterns: (points, hidden, targets)."""
        assert part in ("attention", "mlp") and (L > 0 or part == "attention")
        path = self.out_dir / (f"M_block{L:02d}.npz" if part == "attention" else f"M_mlp{L:02d}.npz")
        if path.exists():
            return dict(np.load(path))
        logits_path = path.with_name(path.stem + "_logits.npz")
        d, start = self.data["all"], 0 if L == 0 else self.stages.index((L, 1 if part == "attention" else 2))
        points = self.stages[start:]
        extra = ["z_no_dense"] if L == 0 else []
        sums = Sums(len(points) + len(extra), self.model.hidden_size, self.model.device, len(MOMENT_TARGETS))
        captured = {}
        ablation = [] if L == 0 else [self.ablate(L, range(self.n_heads)) if part == "attention" else self.ablate_mlp(L)]
        hooks = ablation + self.capture(points, captured)
        head, aa_ids = self.model.model.lm_head, torch.tensor(self.model.ids(AA), device=self.model.device)
        logits = {name: [] for name in ["z"] + extra}
        try:
            for batch in tqdm(d["batches"], desc=f"M: block {L}", mininterval=120):
                input_ids, attention_mask = self.model.token_ids([d["seqs"][j] for j in batch])
                with torch.no_grad():
                    self.model.model.esm(input_ids=input_ids, attention_mask=attention_mask)
                    captured["z"] = self.to_z(captured[(self.n_layers, 2)])
                    if extra:
                        captured["z_no_dense"] = head.layer_norm(self.model.model.esm.encoder.emb_layer_norm_after(captured[(self.n_layers, 2)]))
                batch_index, token_index, y, masks = self.moment_targets(d, batch)
                with torch.no_grad():
                    for name in logits:  # the decoder of the LM head (tied to the embeddings) and its bias, for the 20 amino acids
                        z = captured[name][batch_index, token_index]
                        logits[name].append((head.decoder(z) + head.bias)[:, aa_ids].half().cpu().numpy())
                sums.add_targets(y, masks)
                for i, s in enumerate(points + extra):
                    sums.add(i, captured[s][batch_index, token_index], y, masks)
        finally:
            for hook in hooks:
                hook.remove()
        fits = [sums.decoders(i) for i in range(len(points) + len(extra))]
        keys = ["r2", "corr2", "decoders", "patterns"]
        result = {"stages": np.array(self.stage_names[start:]), "targets": np.array(MOMENT_TARGETS),
                  **{key: np.array([f[k] for f in fits[:len(points)]]) for k, key in enumerate(keys)}}
        for name, f in zip(extra, fits[len(points):]):
            result.update({f"{name}_{key}": f[k] for k, key in enumerate(keys)})
        saved = {name: np.concatenate(rows) for name, rows in logits.items()}
        if L == 0:
            order = [j for batch in d["batches"] for j in batch]
            saved["protein"] = np.concatenate([np.full(len(d["h_i"][j]), j) for j in order])  # index in Study.data["all"]
            saved["position"] = np.concatenate([np.asarray(d["h_i"][j]) for j in order])
        np.savez(logits_path, amino_acids=np.array(list(AA)), **saved)
        np.savez(path, **result)
        del sums
        torch.cuda.empty_cache()
        return result

    def head_ablation(self, L):
        """B: R^2_k at L.1, L.2 and (L+1).2 (z for the last block) for no ablation, each head of block L and all its heads (5000 proteins)."""
        path = self.out_dir / f"B_block{L:02d}.npz"
        if path.exists():
            return dict(np.load(path))
        d = self.data["subset"]
        variants = [[]] + [[h] for h in range(self.n_heads)] + [list(range(self.n_heads))]
        points = [f"{L}.1", f"{L}.2", f"{L + 1}.2" if L < self.n_layers else "z"]
        sums = Sums(len(variants) * len(points), self.model.hidden_size, self.model.device)
        block, following = self.blocks[L - 1], self.blocks[L] if L < self.n_layers else None
        checked = False
        for batch in tqdm(d["batches"], desc=f"B: block {L}", mininterval=120):
            input_ids, attention_mask = self.model.token_ids([d["seqs"][j] for j in batch])
            grabbed = {}

            def grab(module, args, kwargs):  # the input of block L, its attention mask and position embeddings; then stop
                grabbed["args"], grabbed["kwargs"] = args, kwargs
                raise Stop

            pre = block.register_forward_pre_hook(grab, with_kwargs=True)
            try:
                with torch.no_grad():
                    self.model.model.esm(input_ids=input_ids, attention_mask=attention_mask)
            except Stop:
                pass
            finally:
                pre.remove()
            kwargs = dict(grabbed["kwargs"])
            hidden = grabbed["args"][0] if grabbed["args"] else kwargs.pop("hidden_states")
            batch_index, token_index, y, masks = self.helix_batch(d, batch)
            sums.add_targets(y, masks)
            for v, heads in enumerate(variants):
                after_attention = {}
                hooks = [block.attention.register_forward_hook(lambda m, i, o: after_attention.__setitem__("h", first(o)))]
                if heads:
                    hooks.append(self.ablate(L, heads))
                try:
                    with torch.no_grad():
                        out = first(block(hidden, **kwargs))
                        nxt = first(following(out, **kwargs)) if following is not None else self.to_z(out)
                finally:
                    for hook in hooks:
                        hook.remove()
                if not checked and not heads:  # the block run alone gives the same output as the whole model
                    reference = {}
                    ref_hook = block.register_forward_hook(lambda m, i, o: reference.__setitem__("h", first(o)))
                    try:
                        with torch.no_grad():
                            self.model.model.esm(input_ids=input_ids, attention_mask=attention_mask)
                    finally:
                        ref_hook.remove()
                    difference = (reference["h"] - out).abs().max().item()
                    assert difference < 1e-3 * reference["h"].abs().max().item(), difference
                    checked = True
                for p, h in enumerate([after_attention["h"], out, nxt]):
                    sums.add(v * len(points) + p, h[batch_index, token_index], y, masks)
        fits = [[sums.r2(v * len(points) + p) for p in range(len(points))] for v in range(len(variants))]
        result = {"points": np.array(points), "variants": np.array(["none"] + [f"head {h}" for h in range(self.n_heads)] + ["all heads"]),
                  "r2": np.array([[f[0] for f in row] for row in fits]), "r2_fold": np.array([[f[1] for f in row] for row in fits])}
        np.savez(path, **result)
        del sums
        torch.cuda.empty_cache()
        return result
