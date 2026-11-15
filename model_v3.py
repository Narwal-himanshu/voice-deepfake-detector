"""V3 model: XLS-R 300M front-end with learnable layer fusion + AASIST-style back-end.

Why v3: the v1 model used ONLY the final XLS-R layer (xlsr_aasist.py line 122).
But deepfake artefacts live at different depths: lower layers keep acoustic /
channel cues, higher layers are more semantic. The strongest cross-domain
anti-spoofing systems (e.g. XLSR-SLS, ACM MM 2024) learn a softmax-weighted sum
of ALL 24 transformer layers, which generalises far better to unseen TTS/VC
systems. This file implements that, keeping the v1 back-end for comparability.

Checkpoints are saved as a dict:
    {"state_dict": ..., "config": {...}, "arch": "xlsr_aasist_v3", "epoch": ...}
so the exact architecture can be rebuilt at load time.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SR = 16000
MAX_LEN = 4 * SR


def fix_length(x, n=MAX_LEN):
    if len(x) >= n:
        return x[:n]
    if len(x) == 0:
        return np.zeros(n, dtype=np.float32)
    return np.tile(x, int(np.ceil(n / len(x))))[:n]


# ---------------------------------------------------------------- back-end
class ResBlock(nn.Module):
    def __init__(self, cin, cout, pool_t=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.short = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        self.pool = nn.MaxPool2d((1, pool_t)) if pool_t > 1 else nn.Identity()

    def forward(self, x):
        y = self.c1(F.selu(self.bn1(x)))
        y = self.c2(F.selu(self.bn2(y)))
        return self.pool(y + self.short(x))


class GraphAttn(nn.Module):
    """Attention over a fully connected graph. nodes (B, N, din) -> (B, N, dout)."""

    def __init__(self, din, dout):
        super().__init__()
        self.q, self.k, self.v = nn.Linear(din, dout), nn.Linear(din, dout), nn.Linear(din, dout)
        self.res = nn.Linear(din, dout)
        self.bn = nn.BatchNorm1d(dout)
        self.scale = dout ** -0.5

    def forward(self, h):
        att = torch.softmax(self.q(h) @ self.k(h).transpose(1, 2) * self.scale, dim=-1)
        out = att @ self.v(h) + self.res(h)
        return F.selu(self.bn(out.transpose(1, 2)).transpose(1, 2))


class Branch(nn.Module):
    """Graph attention on one axis (spectral or temporal) + a learnable master node."""

    def __init__(self, d):
        super().__init__()
        self.g1, self.g2 = GraphAttn(d, d), GraphAttn(d, d)
        self.master = nn.Parameter(torch.randn(1, 1, d) * 0.1)

    def forward(self, nodes):
        h = self.g1(nodes)
        hm = self.g2(torch.cat([self.master.expand(h.size(0), -1, -1), h], dim=1))
        return hm[:, 0], hm[:, 1:]


def tiny_ssl_config():
    """Small Wav2Vec2 config used only by selftest_v3 / --debug-tiny."""
    from transformers import Wav2Vec2Config
    return Wav2Vec2Config(
        vocab_size=32, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=64, hidden_act="gelu", hidden_dropout=0.0,
        activation_dropout=0.0, attention_dropout=0.0, feat_proj_dropout=0.0,
        final_dropout=0.0, layerdrop=0.0, apply_spec_augment=False,
        mask_time_prob=0.0, mask_feature_prob=0.0,
        conv_dim=(16, 16), conv_stride=(5, 2), conv_kernel=(10, 3),
        num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2,
        do_stable_layer_norm=False, feat_extract_norm="group",
        feat_extract_activation="gelu",
    )


def build_tiny_ssl():
    from transformers import Wav2Vec2Model
    return Wav2Vec2Model(tiny_ssl_config())


def parse_layers(layers):
    """'all' -> all hidden states; '0,4,8,24' -> those indices."""
    if layers is None or layers == "all":
        return None
    if isinstance(layers, (list, tuple)):
        return [int(i) for i in layers]
    return [int(i) for i in str(layers).split(",") if str(i).strip() != ""]


class XLSR_AASIST_V3(nn.Module):
    def __init__(self, ssl_name="facebook/wav2vec2-xls-r-300m", ssl=None, d=64,
                 layers="all", dropout=0.5, tiny=False):
        super().__init__()
        if ssl is None:
            if tiny:
                ssl = build_tiny_ssl()
            else:
                from transformers import Wav2Vec2Model
                ssl = Wav2Vec2Model.from_pretrained(ssl_name, layerdrop=0.0,
                                                    apply_spec_augment=False)
        self.ssl = ssl
        self.ssl.freeze_feature_encoder()  # keep the small CNN feature encoder frozen

        n_states = self.ssl.config.num_hidden_layers + 1  # +1 for the embedding output
        idx = parse_layers(layers)
        if idx is None:
            idx = list(range(n_states))
        idx = [i for i in idx if 0 <= i < n_states] or list(range(n_states))

        # learnable softmax weights over the selected layers (this is the core v3 change)
        self.register_buffer("layer_indices", torch.tensor(idx, dtype=torch.long))
        self.layer_logits = nn.Parameter(torch.zeros(len(idx)))

        hidden = self.ssl.config.hidden_size
        self.proj = nn.Linear(hidden, 128)
        self.bn0 = nn.BatchNorm2d(1)
        self.enc = nn.Sequential(ResBlock(1, 32, 2), ResBlock(32, 32, 2), ResBlock(32, d), ResBlock(d, d))
        self.spec, self.temp = Branch(d), Branch(d)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(6 * d, 1)

        self.config = {
            "arch": "xlsr_aasist_v3",
            "ssl_name": ssl_name,
            "d": d,
            "layers": idx,
            "dropout": dropout,
            "tiny": bool(tiny),
        }

    def layer_weights(self):
        return torch.softmax(self.layer_logits, dim=0)

    def fused_ssl(self, wav):
        out = self.ssl(wav, output_hidden_states=True)
        hs = out.hidden_states
        w = self.layer_weights().to(hs[0].dtype)
        feat = hs[self.layer_indices[0]] * w[0]
        for j in range(1, len(self.layer_indices)):
            feat = feat + hs[self.layer_indices[j]] * w[j]
        return feat

    def forward(self, wav):  # wav (B, 64000) -> logits (B,)
        wav = wav.float()
        wav = (wav - wav.mean(1, keepdim=True)) / torch.sqrt(wav.var(1, keepdim=True) + 1e-7)
        h = self.fused_ssl(wav)                        # (B, T, hidden)
        x = self.proj(h).transpose(1, 2).unsqueeze(1)  # (B, 1, 128, T)
        x = F.selu(self.bn0(F.max_pool2d(x, (3, 3))))
        x = self.enc(x)                                # (B, d, freq, time)
        s = x.abs().amax(dim=3).transpose(1, 2)        # spectral nodes
        t = x.abs().amax(dim=2).transpose(1, 2)        # temporal nodes
        ms, ns = self.spec(s)
        mt, nt = self.temp(t)
        z = torch.cat([ms, ns.amax(1), ns.mean(1), mt, nt.amax(1), nt.mean(1)], dim=1)
        return self.out(self.drop(z)).squeeze(1)


# ---------------------------------------------------------------- io helpers
def build_ssl_from_config(ssl_name="facebook/wav2vec2-xls-r-300m"):
    """Build an untrained XLS-R skeleton from a tiny config file (no 1.2 GB download)."""
    from transformers import Wav2Vec2Config, Wav2Vec2Model
    cfg = Wav2Vec2Config.from_pretrained(ssl_name, layerdrop=0.0, apply_spec_augment=False)
    return Wav2Vec2Model(cfg)


def save_checkpoint(path, model, extra=None):
    payload = {
        "state_dict": model.state_dict(),
        "config": getattr(model, "config", {}),
        "arch": getattr(model, "config", {}).get("arch", "unknown"),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(ckpt_path):
    try:
        return torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:  # older torch
        return torch.load(ckpt_path, map_location="cpu")


def load_model_v3(ckpt_path, device="cpu", ssl_name=None):
    """Rebuild a v3 model from a checkpoint. Works with plain state_dicts too."""
    obj = load_checkpoint(ckpt_path)
    if isinstance(obj, dict) and "state_dict" in obj:
        state, cfg = obj["state_dict"], obj.get("config", {})
    else:
        state, cfg = obj, {}
    cfg = dict(cfg or {})
    name = ssl_name or cfg.get("ssl_name", "facebook/wav2vec2-xls-r-300m")
    if cfg.get("tiny"):
        model = XLSR_AASIST_V3(ssl=build_tiny_ssl(), ssl_name=name,
                               d=cfg.get("d", 64), layers=cfg.get("layers", "all"),
                               dropout=cfg.get("dropout", 0.5), tiny=True)
    else:
        model = XLSR_AASIST_V3(ssl=build_ssl_from_config(name), ssl_name=name,
                               d=cfg.get("d", 64), layers=cfg.get("layers", "all"),
                               dropout=cfg.get("dropout", 0.5))
    model.load_state_dict(state)
    return model.to(device)


@torch.no_grad()
def predict_logits(model, loader, device, amp=True):
    model.eval()
    labels, scores = [], []
    for batch in loader:
        wav, y = batch[0], batch[1]
        with torch.autocast("cuda", dtype=torch.float16,
                            enabled=bool(amp) and device.type == "cuda"):
            logits = model(wav.to(device))
        scores += logits.float().cpu().tolist()
        labels += y.tolist()
    return labels, scores


@torch.no_grad()
def predict(model, loader, device, amp=True, temperature=1.0):
    labels, logits = predict_logits(model, loader, device, amp)
    probs = torch.sigmoid(torch.tensor(logits) / float(temperature)).tolist()
    return labels, probs
