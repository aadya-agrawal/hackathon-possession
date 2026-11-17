# Reproducing and inspecting the pipeline

The [technical document](Westside.md) describes the method, decisions and limitations. This
guide covers how to install the environment, regenerate the submission, regenerate the committed
models and video outputs, and rerun the validation. All commands run from the repository root.
Only regenerating the video outputs or the models needs a GPU.

## 1. Environment and tests

```bash
git clone https://github.com/aadya-agrawal/hackathon-possession.git
cd hackathon-possession
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-pinned.txt
python -m unittest discover -s tests -p test_possession.py
python -m possession.cli --help
```

`requirements-pinned.txt` pins the CPU environment (Python 3.9, NumPy 1.26.4, pandas 2.3.3) that
reproduces the submission byte for byte; NumPy 2.x changes scalar promotion and is not supported.
`requirements-tools.txt` adds the pinned GPU stack used by `tools/`. `requirements.txt` and
`requirements-starter.txt` are the organisers' original evaluation and starter manifests. The
tests need no data, weights or network access.

## 2. Match data

Obtain the licensed bundle through the official challenge access process ([DATA.md](../DATA.md))
and point `MTAH_DATA_DIR` at it. The data must not be committed or redistributed; `data/` contains
only its README. The pipeline reads the tactical tracking, metadata, homography and Smart Tagging,
and the broadcast tracking, metadata and homography. It does not read either video.

## 3. Regenerating the submission

```bash
export MTAH_DATA_DIR=/absolute/path/to/your/licensed/bundle
python -m possession.cli \
  --config configs/submission_pre_kickoff.json \
  --output Westside.csv
python evaluation/validate_submission.py Westside.csv
cmp Westside.csv /path/to/uploaded/Westside.csv          # no output: byte-identical
```

The run takes about 20 seconds on a many-core CPU. `configs/submission_pre_kickoff.json` is also
the CLI default. The submitted CSV is not committed; compare with the file uploaded to the
competition, or with the SHA-256 below.

| File | SHA-256 |
| --- | --- |
| `configs/submission_pre_kickoff.json` | `59caff7fcae80ab93ed28ecbf73e8af00b177e3425a1b5a44692fc5f1cf108e4` |
| Submitted `Westside.csv` (not committed) | `f91eccb244277037d81b2910b44f52bd7b09041c00e7fb599195c68554c92734` |
| `models/ballfree_seq_ball_cnn3.npz` (ball-aware model) | `9ba386ccdd3ac40937fc568550f3cc0bec45719530e8c965c2591603e0119c17` |
| `models/ballfree_seq_cnn3_aug.npz` (players-only model) | `bfd114a3504f1aede968b8d22b3de1c14f69198357d6df37b7853482b400dc22` |
| `models/ballfree_seq_cnn3.npz` (earlier players-only model) | `1ad27e3f352cdf20a4d657940cbcfcd2639bf35377fa38c523885c6d67d43a28` |
| `evidence/team_at_ball_gaps.npz` | `2f70315a9c990319b58f52a18da35d1b87d24b60cb23cdf033caf31475e95223` |
| `evidence/track_team_override.npz` | `97179f978e4eb31f837e8438d951e22f537a4147e9f29607b432f93f9996eca0` |

**Previous configurations.** Every adopted change kept its predecessor. Each
`configs/submission_*.json` records, in its `working_submission_selection` section, the previous
configuration and its hash, the parameters that changed and the evidence for the change; the
matching `*_trial.json` is the configuration as it was evaluated. Any of them can be run with
`--config`. The immediately previous configuration is `configs/submission_ball_aware.json`
(CSV `a0ce1766f4b1bebdc7a0146d3f6f8dd0a0417464fcac2dcb2166b73ff55b4863`).

**Time units.** The challenge tracking runs at 25 fps, and all frame-based parameters in the
selected configuration are in 25-fps frames.

## 4. Video outputs (committed)

The selected configuration reads two files computed automatically from the tactical video: the
team of the player at the ball during tactical ball gaps (`evidence/team_at_ball_gaps.npz`) and a
per-frame correction of track teams from kit colour (`evidence/track_team_override.npz`). They
contain only our detector and classifier outputs, not tracking positions or video content, and
they are committed so the submission can be regenerated without the video or a GPU.

To regenerate them, install `requirements-tools.txt`. The scripts use public detector weights
(SoccerNet-v3D `yolo-sn-ball-opt.pt` and Ultralytics `yolo26l.pt` / `yolo26x.pt`), which
`tools/fetch_video_weights.py` downloads and checks against their SHA-256. The few
match-specific inputs of the extractor (masking-test windows excluded from kit-colour training,
static detector-clutter locations and each half's playing direction) are derived automatically
from the data bundle by `tools/derive_match_constants.py` and cached in the work folder. Outputs
are bit-identical across NVIDIA L40S GPUs; CPU inference differs slightly, so the committed files
are the reference.

```bash
export MTAH_DATA_DIR=/absolute/path/to/your/licensed/bundle
python tools/fetch_video_weights.py                      # cache/yolo-sn-ball-opt.pt, cache/yolo26l.pt, cache/yolo26x.pt
python tools/extract_team_at_ball.py run --devices 0 --work /tmp/tab_work \
  --out /tmp/team_at_ball_gaps.npz                       # about 45 min on 6 GPUs
python tools/audit_track_teams.py run --devices 0 --tab-work /tmp/tab_work \
  --work /tmp/audit --out /tmp/track_team_override.npz
sha256sum /tmp/team_at_ball_gaps.npz /tmp/track_team_override.npz
```

Each script's docstring lists its inputs, constants and options.

## 5. Models

The ball-free models are trained on external matches only (Metrica sample games 1–3 and DFL
J03WMX, J03WN1, J03WOH, J03WOY). From the provided match, training uses only per-frame counts of
tracked players and the mask of frames where the ball is missing, to make the external data look
like the challenge tracking. Training is deterministic on a given GPU model.

```bash
python tools/train_ballfree_augmented.py --metrica-root /path/to/metrica-sample-data \
  --dfl META_J03WMX.xml:POS_J03WMX.xml --dfl ... --device cuda:0   # models/ballfree_seq_cnn3_aug.npz
python tools/train_ballfree_ball.py --metrica-root /path/to/metrica-sample-data \
  --dfl META_J03WMX.xml:POS_J03WMX.xml --dfl ... --device cuda:0   # models/ballfree_seq_ball_cnn3.npz
python -m possession.external_ballfree_seq --help                    # models/ballfree_seq_cnn3.npz
```

Inference uses NumPy only; PyTorch is needed only for training.

## 6. Validation

### Selection tooling (`validation/`)

Every adopted change was evaluated with the scripts in `validation/` (usage in each module
docstring; paths come from command-line options or `METRICA_ROOT`, `DFL_ROOT`, `MTAH_DATA_DIR`,
`FOLD_DIR` and `VALIDATION_CACHE`):

| Script | Role |
| --- | --- |
| `fast_eval.py` | Metrica regression checks (8 degradation scenarios × 60 checks per sample game) and DFL spell F1, comparing a candidate configuration with a baseline |
| `build_challenge_inputs.py` | Local cache of challenge tracking masks used by the simulations and the masking test (never commit its output) |
| `proxy_eval.py`, `proxy_eval_families.py` | DFL matches degraded to resemble the challenge tracking: replayed missing players and missing ball, optional ball noise, and a noise model fitted to challenge touch statistics |
| `fit_artifact.py`, `challenge_stats.py`, `analyse_families.py`, `artifact_protocol.json` | Fitting and fixed protocol of that noise model |
| `train_folds.py` | Models trained without each evaluation match, for unbiased evaluation (GPU) |
| `swap_windows.py`, `swap_run.py`, `score_swap.py`, `merge_team_evidence.py` | Masking test: hide the tracked ball in windows where it was visible and compare the output with the same configuration's output on the full data |
| `oracle_breakdown.py` | Breakdown of the remaining error into in-play/dead and team decisions on external data |

Reference values of the masking test for the previous configuration: development windows 0.9292
(whole match) / 0.8105 (inside windows), test windows 0.9474 / 0.8582 spell F1.

The Metrica sample data must be pinned to the revision used throughout:

```bash
git clone https://github.com/metrica-sports/sample-data.git /path/to/metrica-sample-data
git -C /path/to/metrica-sample-data switch --detach e706dd506b360d69d9d123d5b8026e7294b13996
```

`configs/external_transfer_validation_v2.json` defines the 25-fps external validation protocol;
the original `configs/external_transfer_validation.json` is its base.

### Robustness checks

[`tests/run_validation_framework.py`](../tests/run_validation_framework.py) runs the automatic
robustness suite on the provided match (determinism, stability under small perturbations of the
inputs, and structural checks); the selected configuration passes 128 of 134 checks. These are
stability diagnostics, not accuracy measurements.

```bash
python tests/run_validation_framework.py --help
```

The public five-minute ground truth is used only for the format and alignment checks in
[EVALUATION.md](../EVALUATION.md), never for tuning.

## 7. External geometry calibration

The ball-kinematics limits in the configuration (maximum interpolation gap, ball speed and
acceleration) were selected on SoccerNet Game State Reconstruction train/valid annotations. Obtain
SoccerNet GSR through its official access process, outside this repository, and extract `train/`
and `valid/` only:

```bash
python -m possession.external_soccernet \
  --root /path/to/SoccerNetGS --output artifacts/soccernet_gamestate_manifest.json
python -m possession.external_soccernet_metrics \
  --manifest artifacts/soccernet_gamestate_manifest.json \
  --output artifacts/soccernet_geometry_metrics.json
python -m possession.external_parameter_selection \
  --metrics artifacts/soccernet_geometry_metrics.json \
  --base-config configs/default.json --target-fps 50 \
  --output-config configs/calibration_local.json \
  --output-report artifacts/external_parameter_validation.json
```

This reproduces the original selection procedure, which was expressed at 50 fps; do not overwrite
the selected configuration with its output.

## 8. Third-party assets and licences

Nothing below is vendored: weights are downloaded by `tools/fetch_video_weights.py` and datasets
are obtained from their publishers. The match bundle is licensed for this competition only and is
never committed; `evidence/` holds only our own detector and classifier outputs.

| Asset | Used for | Licence / terms |
| --- | --- | --- |
| Ultralytics library and `yolo26l.pt` / `yolo26x.pt` detectors | `tools/` video outputs | AGPL-3.0 |
| SoccerNet-v3D `yolo-sn-ball-opt.pt` ball detector ([repo](https://github.com/mguti97/SoccerNet-v3D)) | `tools/` video outputs | GPL-2.0 |
| DFL Bundesliga tracking and event dataset (Bassek et al.) | training the committed models, validation | CC BY 4.0 |
| Metrica Sports sample data ([repo](https://github.com/metrica-sports/sample-data)) | training and validation | public sample data; no licence file published |
| SoccerNet Game State Reconstruction | ball-kinematics calibration | SoccerNet terms |
| SoccerTrack v2 ([dataset](https://huggingface.co/datasets/atomscott/soccertrack-v2)) | early contact-timing diagnostics only | CC BY 4.0 |

No human labels on the provided match are used anywhere. Two SoccerTrack v2 touch events were
checked by hand during early exploration; they are not used by any model, parameter or the
submission.

The repository's `LICENSE` is the organisers' MIT licence for their original files only; our own
code and documentation are released under AGPL-3.0 (`LICENSE-AGPL-3.0.txt`), which is compatible
with the AGPL-3.0 Ultralytics dependency.
