"""Model configs and adapters: the operations of the study for one model family.

A model config (configs/models/*.toml) has:
  name             short name, also the name of the output directory
  hf_id            Hugging Face model id
  adapter          model family, a key of ADAPTERS
  layers           layers of the bins and the PCA (notebook 2)
  ablation_layers  layers of the ablation (notebook 3); default: layers
  steering_layer   layer of the steering (notebook 5)
  max_tokens       tokens per forward pass of whole proteins (bins); default 16384
  edit_max_tokens  tokens per forward pass of the edited copies (ablation); default 131072
  matmul_precision float32 matrix products: "highest" (full fp32) or "high" (TF32 on NVIDIA GPUs, ~27% faster for 150M); default "highest"

Layer L is the residual stream after transformer block L (1-based), before the final layer norm.
"""

import tomllib

import torch


def load_config(path):
    with open(path, "rb") as f:
        config = tomllib.load(f)
    config.setdefault("ablation_layers", config["layers"])
    config.setdefault("max_tokens", 16384)
    config.setdefault("edit_max_tokens", 2**17)
    config.setdefault("matmul_precision", "highest")
    return config


class Esm2:
    """ESM-2 masked language models of Hugging Face transformers (facebook/esm2_*)."""

    def __init__(self, hf_id, device=None):
        from transformers import AutoTokenizer, EsmForMaskedLM

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)
        self.model = EsmForMaskedLM.from_pretrained(hf_id, attn_implementation="sdpa").eval().to(self.device)
        self.n_layers = len(self.model.esm.encoder.layer)
        self.hidden_size = self.model.config.hidden_size
        self.mask_id = self.tokenizer.mask_token_id
        self.vocabulary = self.tokenizer.convert_ids_to_tokens(list(range(len(self.tokenizer))))
        self.offset = 1  # residue i is token i + 1 (token 0 is <cls>)

    def token_ids(self, sequences):
        """Token ids and attention mask of a batch of sequences, padded to the longest."""
        tokens = self.tokenizer(sequences, return_tensors="pt", padding=True).to(self.device)
        return tokens["input_ids"], tokens["attention_mask"]

    def ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(list(tokens))

    @torch.no_grad()
    def hidden_states(self, input_ids, attention_mask, layers):
        """{layer: residual stream (batch, tokens, hidden)} of a padded batch, with read-only hooks."""
        captured, hooks = {}, []
        for layer in layers:
            hooks.append(self.model.esm.encoder.layer[layer - 1].register_forward_hook(
                lambda _module, _input, output, layer=layer: captured.__setitem__(layer, output[0] if isinstance(output, tuple) else output)))
        try:
            self.model.esm(input_ids=input_ids, attention_mask=attention_mask)
        finally:
            for hook in hooks:
                hook.remove()
        return captured

    @torch.no_grad()
    def embed(self, input_ids):
        """Residual stream before block 1, for unpadded batches (every sequence of the batch has the same length)."""
        return self.model.esm.embeddings(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))

    @torch.no_grad()
    def run(self, h, start, end):
        """Blocks start+1 .. end on the residual stream h (batch, tokens, hidden), unpadded."""
        rotary = self.model.esm.rotary_embeddings(h, torch.arange(h.shape[1], device=h.device)[None])  # rotary position embeddings
        for block in self.model.esm.encoder.layer[start:end]:
            h = block(h, position_embeddings=rotary)
        return h

    @torch.no_grad()
    def head(self, h):
        """Logits over the vocabulary from final residual streams (rows, hidden): final layer norm and the amino-acid head."""
        return self.model.lm_head(self.model.esm.encoder.emb_layer_norm_after(h))


ADAPTERS = {"esm2": Esm2}


def load_model(config, device=None):
    torch.set_float32_matmul_precision(config["matmul_precision"])
    return ADAPTERS[config["adapter"]](config["hf_id"], device)
