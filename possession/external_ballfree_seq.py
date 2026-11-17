"""Train the ball-free sequence model (cnn3) from external matches only (Metrica sample, DFL).

python -m possession.external_ballfree_seq --metrica-root /path/to/metrica-sample-data \
    --dfl /path/DFL_02_01_matchinformation_...xml:/path/DFL_04_03_positions_raw_observed_...xml ... \
    --out models/ballfree_seq_cnn3.npz

Labels are provider possession references on other games; nothing from the challenge match is used.
Training is deterministic (fixed seed, deterministic torch kernels); needs torch, inference
(possession.ballfree_seq) does not. Matches are used in argument order: Metrica Sample_Game_1..3,
then the DFL pairs as given (the shipped weights used J03WMX, J03WN1, J03WOH, J03WOY).
With those four DFL pairs, --iters 500 --seed 0 on a CUDA GPU (NVIDIA L40S, torch 2.4.1, default
TF32 convolution settings) reproduces models/ballfree_seq_cnn3.npz bit for bit; CPU training or
other GPUs give slightly different floating-point weights.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from possession import AWAY, HOME
from possession.ballfree_seq import period_views

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

METRICA_GAMES = ("Sample_Game_1", "Sample_Game_2", "Sample_Game_3")
STEP = 5             # 25 fps -> 5 fps
CHUNK = 200          # training chunk length in 5 fps steps (40 s)
BATCH = 64
DILATIONS = (1, 2, 2)  # kernel 7: receptive field +-15 steps = +-3 s
HIDDEN = 48
KERNEL = 7
DROPOUT = 0.1


def metrica_matches(root, protocol):
    from possession.external_metrica_validation import _load_events, _load_game
    from possession.external_possession_events import possession_chain_references
    for game in METRICA_GAMES:
        data = _load_game(Path(root), game)
        events = _load_events(data["event_path"], {
            "PASS", "CARRY", "RECOVERY", "BALL LOST", "BALL OUT", "SHOT", "SET PIECE", "FAULT RECEIVED",
        })
        truth, _ = possession_chain_references(events, data["frames"], data["periods"], data["fps"],
                                               protocol["references"])
        edges = np.r_[0, np.flatnonzero(np.diff(data["periods"])) + 1, len(truth)]
        yield game, (105.0, 68.0), [(data["home"][a:b], data["away"][a:b], truth[a:b])
                                    for a, b in zip(edges[:-1], edges[1:])]


def dfl_matches(pairs):
    from possession.external_dfl_validation import load_dfl_match
    for metadata, positions in pairs:
        pitch, periods = load_dfl_match(metadata, positions)
        yield Path(positions).stem, pitch, [(p["home"], p["away"], p["truth"]) for p in periods]


def match_views(pitch, periods):
    H, A, truth, bounds, start = [], [], [], [], 0
    for home, away, t in periods:
        h, a = period_views(home, away, pitch, 25)
        H.append(h); A.append(a); truth.append(np.asarray(t, dtype=np.int8))
        bounds.append((start, start + len(t))); start += len(t)
    return {"H": np.concatenate(H), "A": np.concatenate(A), "truth": np.concatenate(truth), "bounds": bounds}


def build_model(channels, mean, std):
    import torch
    import torch.nn as nn

    class ViewNet(nn.Module):
        def __init__(self):
            super().__init__()
            layers = [nn.Conv1d(channels, HIDDEN, 1), nn.GELU()]
            for d in DILATIONS:
                layers += [nn.Conv1d(HIDDEN, HIDDEN, KERNEL, dilation=d, padding=d * (KERNEL // 2)), nn.GELU(),
                           nn.Dropout(DROPOUT)]
            self.net = nn.Sequential(*layers, nn.Conv1d(HIDDEN, 1, 1))

        def forward(self, x):  # (B, T, C) -> (B, T)
            return self.net(x.transpose(1, 2))[:, 0]

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
            self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))
            self.g = ViewNet()

        def forward(self, home, away):  # antisymmetric home logit
            both = (torch.cat([home, away]) - self.mean) / self.std
            s = self.g(both)
            return s[:len(home)] - s[len(home):]

    return Model()


def train(matches, iters=500, seed=0, device="cpu"):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed); np.random.seed(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    allx = np.concatenate([np.concatenate([d["H"], d["A"]]) for d in matches])
    mean, std = allx.mean(0), allx.std(0) + 1e-3
    # 5 fps phase-shifted sequences per period; loss only on live frames
    seqs = []
    for d in matches:
        for a, b in d["bounds"]:
            for o in range(STEP):
                t = d["truth"][a + o:b:STEP]
                seqs.append((d["H"][a + o:b:STEP], d["A"][a + o:b:STEP], (t == HOME).astype(np.float32),
                             np.isin(t, (HOME, AWAY)).astype(np.float32)))
    lengths = np.array([len(s[0]) for s in seqs], float)
    rng = np.random.default_rng(seed)
    model = build_model(allx.shape[1], mean, std).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=iters, pct_start=0.1)
    tensors = [tuple(torch.as_tensor(x, device=device) for x in s) for s in seqs]
    for _ in range(iters):
        idx = rng.choice(len(seqs), BATCH, p=lengths / lengths.sum())
        starts = [rng.integers(0, max(1, len(seqs[i][0]) - CHUNK)) for i in idx]
        H, A, y, m = (torch.stack([tensors[i][k][s:s + CHUNK] for i, s in zip(idx, starts)]) for k in range(4))
        loss = (nn.functional.binary_cross_entropy_with_logits(model(H, A), y, reduction="none") * m).sum() / m.sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return model


def export(state_dict, path):
    """Write the numpy weights used by possession.ballfree_seq."""
    get = lambda key: state_dict[key].detach().cpu().numpy().astype(np.float32)
    convs = [1 + 3 * i + 1 for i in range(len(DILATIONS))]  # nn.Sequential indices of dilated convs
    out = {"mean": get("mean"), "std": get("std"),
           "input_weight": get("g.net.0.weight")[:, :, 0], "input_bias": get("g.net.0.bias"),
           "output_weight": get("g.net.%d.weight" % (convs[-1] + 3))[:, :, 0],
           "output_bias": get("g.net.%d.bias" % (convs[-1] + 3)),
           "dilations": np.array(DILATIONS, dtype=np.int64)}
    for i, index in enumerate(convs):
        out["conv%d_weight" % i] = get("g.net.%d.weight" % index)
        out["conv%d_bias" % i] = get("g.net.%d.bias" % index)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metrica-root", required=True)
    parser.add_argument("--protocol", default="configs/metrica_core_development.json")
    parser.add_argument("--dfl", action="append", default=[], help="METADATA.xml:POSITIONS.xml")
    parser.add_argument("--out", default="models/ballfree_seq_cnn3.npz")
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="e.g. cpu or cuda:0 (default: cuda if available)")
    args = parser.parse_args(argv)
    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    protocol = json.loads(Path(args.protocol).read_text())
    pairs = [item.split(":", 1) for item in args.dfl]
    matches, names = [], []
    for name, pitch, periods in list(metrica_matches(args.metrica_root, protocol)) + list(dfl_matches(pairs)):
        matches.append(match_views(pitch, periods)); names.append(name)
    model = train(matches, args.iters, args.seed, device)
    export(model.state_dict(), args.out)
    print(json.dumps({"weights": args.out, "matches": names, "iters": args.iters, "seed": args.seed, "device": device}))


if __name__ == "__main__":
    main()
