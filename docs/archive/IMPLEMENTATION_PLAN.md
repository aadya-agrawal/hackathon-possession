# Implementation Plan

> Historical working log, not the current team status or a compliance certificate.
> Start with the [technical document](../Westside.md) and
> [REPRODUCIBILITY.md](../REPRODUCIBILITY.md).
> Paths, dataset roles, and decisions below describe their original experiment dates and may
> be superseded. In particular, Game 3 is no longer an unseen test.

## Current Status

The initial tracking-only vertical slice is implemented:

- raw ATD and team metadata parsing;
- ball outlier rejection and short-gap interpolation;
- compact team-control feature extraction;
- Smart Tagging team, set-piece, and restart priors;
- deterministic `H`/`A`/`D` temporal decoding;
- strict submission writing, validation, and aggregate diagnostics;
- JSON configuration and one-command execution.

Run the synthetic behavioral suite with:

```bash
python -m unittest discover -s tests -v
```

The suite covers trajectory repair, malformed frame sequences, team geometry, fuzzy Smart Tagging
team mapping, brief touches, sustained possession changes, dead-ball restarts, missing-evidence
continuity, configuration validation, and submission invariants.

Run the complete label-free promotion framework with:

```bash
python tests/run_validation_framework.py
```

The runner executes the unit suite, deterministic clean inference, proxy checks, coordinate-noise,
missing-observation and downsampling stress tests, end-to-end boundary stability, restart
stability, and parameter sensitivity. Acceptance thresholds are versioned in
`tests/robustness_policy.json`; aggregate results are written to the ignored
`artifacts/validation_framework.json`. It exits nonzero if any check fails. The promoted
`submissions/possession_mvp.csv` remains the comparison reference and must only be replaced after
the candidate passes the framework.

Intentional experiment runs should also be appended to the version-controlled family ledger:

```bash
python tests/run_validation_framework.py \
  --record-family restart-localization \
  --record-variant displacement-confidence-v3 \
  --decision promoted \
  --notes "Confidence-gated displacement onset with fixed-offset fallback"
```

`experiments/results.csv` compares variants within each experiment family and records the
framework version, decision, pass/fail counts, failed checks, configuration and label hashes,
structural metrics, headline proxies, perturbation stability, and notes. Failed and rejected
experiments should be recorded as well as promoted ones. Recording is explicit so routine reruns
do not create duplicate ledger rows.

Run the full pipeline after the match assets are available with:

```bash
python -m possession.cli
```

The first 199,319-frame data-backed iteration is complete. The current tracking-only system
produces a valid submission with 298 possessions, a mean possession duration of 21.5 seconds, and
three possessions shorter than one second. It runs in approximately 20 seconds on the development
machine.

The default live-team transition confirmation is two seconds. This follows the challenge's
controlled-possession guidance and prevents a brief touch, deflection, or duel from becoming a new
possession. Strong restart evidence can still establish control without waiting for a live-team
handover.

Dead-ball evidence now combines automatic restart windows with ball absence and aggregate player
motion. A stoppage is decoded as one contiguous interval leading into a restart rather than as
independent frame decisions. Live possession changes use two confirmation paths: two seconds for
ordinary evidence, or 0.8 seconds when an observed ball is close to the candidate team with a
clear distance or local-density advantage.

On the public correctness window, the current system reports 12 possessions against 17 reference
possessions and reaches 0.4138 spellF1 and 0.6380 macroF1. This small window is not a model-selection
set; the result is recorded to confirm that alignment, team orientation, and possession semantics
work. Video-based ball recovery and richer relative-motion features remain future extensions; the
first semi-Markov family has been evaluated and retained only as an optional ablation.

### Next model improvements

Development after the first complete iteration must not use the public possession window for model
selection. Each change should be accepted through synthetic tests, football semantics, unlabeled
full-match diagnostics, robustness checks, and other-match validation if accessible.

1. **Ball-gap inference**
   - Compare constant-velocity, acceleration-aware, and damped motion models by masking fully
     observed tactical ball tracks.
   - Measure prediction error by horizon and motion regime without possession labels.
   - Use endpoint-aware interpolation when both sides of a gap are trustworthy.
   - Use forward prediction only for a bounded horizon, with decaying confidence and physical
     speed, acceleration, pitch-bound, and direction-change guards.
   - Do not force a position through long gaps when the motion model is uncertain.

   The first implementation is complete. A label-free masked-trajectory benchmark showed that a
   seven-frame quadratic fit outperformed last-step constant velocity through approximately 0.8
   seconds, while both became unreliable around one second. The production model therefore predicts
   at most ten frames (0.4 seconds) from either observed edge, clips fitted acceleration to 20
   metres per second squared, rejects movement above 45 metres per second, enforces pitch bounds,
   and decays confidence from 0.45 to 0.10. Incompatible forward and backward hypotheses leave the
   center missing rather than forcing a trajectory. This recovers 2,400 frames without materially
   changing full-match possession structure.

2. **Relative player-ball motion**
   - Estimate whether nearby players approach the ball, separate after a touch, or subsequently
     move coherently with it.
   - Detect ball velocity and direction changes near a player as reception or interception cues.
   - Distinguish controlled possession from proximity during deflections and aerial challenges.

3. **Per-event restart localization**
   - Replace the shared restart offset with a motion-derived restart frame inside each Smart
     Tagging window.
   - Use ball movement, reappearance, and coordinated player acceleration to locate the restart.

   The confidence-gated implementation is complete. It searches two seconds around the fixed
   estimate and uses observed-ball support plus ten-frame net displacement to detect either a
   low-to-high motion transition or ball reappearance. A restart must show at least a 1 m/s speed
   increase; events without strong evidence retain the fixed 88-frame offset. On the full match,
   22 of 59 restarts are localized. Median ball speed in the ten frames after the chosen restart
   rises from 1.78 m/s for fixed offsets to 7.85 m/s. Under 0.10 m coordinate noise, localization
   status agrees for 98.3% of events and 96.6% remain within five frames; deterministic 10- and
   25-frame missing-observation tests retain 98.3% status agreement. The resulting submission
   changes 0.29% of frames, reduces sub-second possessions from five to three, and preserves the
   fixed-offset method as a per-event fallback.

4. **Semi-Markov decoding**
   - Decode complete possession and dead-ball segments using emission evidence, transition costs,
     and duration priors.
   - Keep the deterministic state machine as a reproducible fallback and ablation.

   A deterministic expanded-state Viterbi implementation is available behind `decoder_mode`.
   The structure-matched variant used 20-frame live and 25-frame dead minimum durations,
   H/A transition cost 20, live/dead cost 10, and dead-emission threshold 0.22. It produced 315
   possessions, zero sub-second spells, 18.7% dead coverage, and passed every perturbation and
   parameter-sensitivity check. It was nevertheless rejected: only 86.7% of decoded team
   boundaries retained geometric control for most of the next second, versus 93.9% for the state
   machine. Raising transition cost to 30 improved that proxy to 93.5% but reduced H/A switches
   from 189 to 93, indicating severe under-segmentation. Framework version 2 now gates decoded
   boundary persistence, team-switch and possession ratios, and dead-share drift. The global
   decoder remains available for future emission-model research; the state machine stays promoted.

   A follow-up adaptive-cost variant resolved that trade-off by retaining cost 30 for unsupported
   H/A transitions while reducing it to 10 when the incoming team has at least eight strong-control
   frames in the next 20. It produces 318 possessions and 180 H/A switches, with 97.2% majority
   and 94.4% strong decoded-boundary persistence. It passed all 60 framework-v2 checks, including
   coordinate noise, missing observations, 50% observation retention, and decoder duration/cost
   sensitivity. This `adaptive-strong-control-v1` variant supersedes the constant-cost candidate
   and is the promoted decoder; the state machine remains the deterministic fallback and reference.

5. **Proposal-constrained HSMM**
   - Use adaptive semi-Markov boundaries as proposals and search a ±25-frame neighborhood.
   - Score complete H/A/D segments with broad log-duration likelihoods, transition compatibility,
     accumulated emissions, and local boundary-evidence gains.
   - Keep maximum segment duration at 7,500 frames, above the longest segment produced by the
     promoted decoder.

   Duration weights 0.5, 1, 2, and 4 were screened without public labels. Duration scoring alone
   slightly reduced both decoded-boundary persistence measures. Adding a boundary potential restored
   318 possessions and 181 H/A switches. The balanced candidate passed all 64 framework-v2 checks
   and was stable under ±20% duration mode, duration weight, and boundary weight changes. It was
   not promoted: majority boundary persistence fell from 97.2% to 96.1%, while strong persistence
   improved only from 94.4% to 94.5%; coordinate-noise and 50% observation-retention agreement also
   remained lower than the adaptive semi-Markov reference. The implementation remains available
   through `decoder_mode = "hsmm"` as a validated ablation.

   Revisit the HSMM family when one or more independent matches are available. Future improvements
   should include duration priors estimated out of sample, restart-type-specific dead durations,
   missingness-calibrated emissions, boundary proposals independent of the current decoder, richer
   player-ball handover features, and uncertainty-aware segment scores. With labeled multi-match
   data, replace hand-tuned scores with a small TCN encoder and semi-Markov CRF; reserve transformer
   encoders for substantially larger multimodal datasets.

6. **Selective tactical-video recovery**
   - Run ball detection only in high-value missing intervals near candidate transitions.
   - Use pitch projection, inverse homography, detector confidence, and physical trajectory checks.
   - Keep broadcast-video synchronization out of scope until tactical-only recovery is stable.

   The tactical H.264 recording has been verified at 1920×1080, 25 fps, and exactly 199,319
   frames. Its homography contains one valid row per frame. A conservative diagnostic ranks 44
   missing-ball intervals, covering 1,943 frames, within 75 frames of decoded H/A boundaries; all
   have two tracking anchors and valid low-error homographies. Three intervals directly cross an
   H/A boundary.

   Four tiled detectors were compared on the same 48 observed-ball frames. At confidence 0.03,
   generic YOLO11l reached 72.9% recall within 40 pixels with 2.33 candidates per frame; YOLO26l
   reached 81.3% with 4.77 candidates; YOLO26x reached 77.1% with 4.35 candidates; and the public
   single-class SoccerNet YOLO11 model reached 81.3% with only 1.40 candidates. SoccerNet is the
   selected task-specific detector. At confidence 0.01 it covered 84.4%, 92.2%, and 79.2% of the
   three boundary-crossing intervals with 1.43–1.52 candidates per frame. YOLO26 remains a useful
   generic comparison, not the primary detector.

   SAM3 was then tested without manual input on one 77-frame boundary interval. Text-only
   `"soccer ball"` prompting produced actual masks on 19 frames, disagreed with the nearest
   SoccerNet candidate by 434 pixels at the median, and ended 340 pixels from the right tracking
   anchor. A second compliant test used the highest-confidence SoccerNet candidate as an automatic
   box seed. It produced masks on 28 frames, but none of the 20 frames with both signals agreed
   within 40 pixels; median disagreement was 219 pixels. Both SAM3 variants are rejected. Manual
   click, box, or frame selection remains prohibited.

   A deterministic second-order temporal path selector now combines two tracking anchors,
   detection confidence, missed-frame penalties, speed, and acceleration. On five automatically
   selected 20-frame masks from the five longest observed-ball runs, the conservative five-frame
   gap and 120 m/s² acceleration gates accepted three windows and selected 49 points. Median error
   was 0.16 m, p90 error was 0.65 m, and every selected point was within 1 m of the hidden ATD
   position. Confidence thresholds from 0.01 through 0.10 selected identical coordinates on shared
   frames. A ten-frame gate accepted all five windows but admitted four errors above 2 m, so it was
   rejected in favor of abstention.

   The selector is validated but not yet fused. The next gate is running it over complete
   two-anchor missing runs, not only boundary-centered diagnostic slices, then measuring endpoint
   consistency and possession-framework effects. Until that passes, video diagnostics do not
   modify possession labels.

   Complete-run validation was subsequently performed for all three missing runs that cross an
   H/A boundary. The runs contain 277, 386, and 1,717 frames. At the 0.03 primary threshold,
   detector coverage was 37.2%, 88.3%, and 87.8%; all three paths abstained. A second pass at the
   predeclared 0.01 rescue threshold increased coverage to 64.3%, 97.7%, and 93.7%, with maximum
   detector gaps of 13, 3, and 15 frames, but the strict physical selector still abstained on all
   three. Disabling practical acceleration limits and raising speed limits to 60 m/s did not
   connect any run. Paths appeared only under implausibly permissive gates, so the global
   two-anchor formulation is rejected rather than forced into trajectory fusion.

   This failure motivates a different Kalman experiment than simple post-path smoothing. The next
   family should use a bounded beam of constant-velocity or constant-acceleration Kalman
   hypotheses to associate noisy candidates through local occlusions, using innovation gating,
   confidence-aware measurement noise, missing-update penalties, and explicit track termination.
   An offline RTS pass may smooth an accepted hypothesis afterward. The family must first pass the
   same masked observed-ball proxy; it must not bridge an entire run merely to satisfy distant
   anchors.

   The Kalman-beam plus RTS family is now implemented as the `kalman_rts` video path mode. It uses
   a four-dimensional position/velocity state, constant-velocity prediction with acceleration
   process noise, confidence-scaled measurement covariance, Mahalanobis innovation gating, bounded
   missing updates, deterministic beam pruning, and an RTS backward pass. Reports retain the
   selected detector index, observation confidence, and smoothed position uncertainty separately,
   preventing unsupported RTS estimates from being mistaken for observations.

   Six process/measurement-noise combinations were screened on the five deterministic masked
   20-frame windows. The conservative 15 m/s² process-noise and 0.5 m base measurement-noise
   candidate accepted three windows and selected 60 observations. Across all 60 smoothed frames,
   median error was 0.14 m, p90 error was 0.28 m, and every frame was within 1 m of the hidden ATD
   position. Increasing process noise to 60 m/s² accepted an additional window but raised p90 error
   to 4.82 m and was rejected. The conservative model still abstained on all three complete missing
   runs, confirming that long runs require terminated and restarted local tracklets rather than one
   global motion hypothesis. No Kalman or RTS output is fused into possession inference.

   Automatic local restart strategies were also evaluated. Unanchored detector-to-detector
   tracklets covered 95% of the five masked windows, but p90 error reached 15.2 m and only 63.2% of
   covered frames were within 1 m. Their internal uncertainty was falsely optimistic, so
   confidence thresholds alone cannot make unanchored restarts safe. This variant is rejected.

   A safer directional variant starts from the real tracking anchor after each missing run and
   associates detector candidates backward toward the nearby possession boundary. It abstained on
   two of the three boundary runs. The remaining run produced only a 14-frame tracklet supported
   by five observations, with maximum position uncertainty of 1.31 m. This is also insufficient
   for fusion. Both implementations remain available as ablations, but video recovery is paused
   until a new independent cue can resolve detector identity; no physical gate should be weakened
   to manufacture coverage.

   Smarter automatic video-model seeding was then tested using the strongest compliant cue:
   homography-projected, truly observed ATD ball positions after each missing run. On the second
   boundary, box-only SAM3 localized a 12-pixel seed to within 0.94 pixels but produced no masks on
   earlier frames. A 24-pixel box selected a large wrong object. Combining the 12-pixel ATD box
   with the generic `"soccer ball"` text produced only four masks and still failed to reach the
   boundary 12 frames earlier. ATD-seeded SAM3 is therefore rejected.

   Homography-initialized pyramidal Lucas–Kanade flow was also evaluated backward for 12 frames.
   Direct ATD pixel seeds looked stable but failed a 20-window observed-ball proxy: under a strict
   0.5-pixel forward/backward gate, 9/20 tracks passed while p90 pitch error was 2.59 m and the
   worst error was 13.57 m. Refining the ATD projection to the nearest SoccerNet candidate improved
   robustness. The stable 20-pixel seed-radius and 0.5-pixel flow gate accepted 4/20 tracks with
   0.50 m median error, 1.02 m p90 error, and all frames within 2 m.

   That strict combined policy rejected the first two target boundaries because their nearest
   detector seeds were 37.5 and 29.4 pixels from the ATD projection. It accepted the third at
   19.6 pixels and propagated to the boundary with 0.13-pixel maximum forward/backward error.
   However, the resulting boundary position was 22.4 m from the nearest SoccerNet boundary
   candidate, despite a 0.315-confidence candidate being present. The independent cues therefore
   disagree and the path is rejected for fusion. Video-derived possession remains disabled.

   Player-relative candidate rescoring is now implemented. It loads player positions only for
   requested frames, projects each detector candidate to metric pitch coordinates, records nearest
   player and team distances, and combines detector confidence with a soft logarithmic distance
   cost. On the 100-frame observed-ball proxy, confidence-only SoccerNet ranking achieved 60%
   top-candidate accuracy within 2 m. Player-relative weights up to 0.5 did not improve that rate;
   the best conservative weight reduced p90 error from 20.5 m to 14.4 m, but the candidate-set
   oracle itself reached only 61% within 2 m. Candidate recall, not ranking, is the main limitation.

   Adding low-threshold YOLO26 rescue candidates also failed to improve the pitch-space oracle
   beyond 61%. Generic candidates were rarely selected after a conservative source penalty, and
   weaker penalties reduced top-candidate accuracy. The SoccerNet-plus-YOLO26 player ensemble is
   therefore rejected.

   Player geometry does provide useful arbitration for the one strict optical-flow path at boundary
   88,340. Its flow endpoint is 2.04 m from an away player and only about 3.5 m from the observed
   right-side ATD anchor across 12 frames. In contrast, the 0.315-confidence SoccerNet boundary
   candidate is 20.4 m from every player and substantially less consistent with the anchor. This
   does not justify promotion, but it retains that 12-frame flow path as the sole candidate for a
   controlled, non-promoted fusion ablation. All other video candidates remain rejected.

   The controlled fusion ablation is now reproducible through an optional video-observation JSON
   hook shared by the CLI and robustness framework. Video observations receive source code 4,
   replace only configured missing/predicted sources, retain explicit confidence, enforce unique
   in-range frames and maximum-speed continuity, and recompute trajectory velocity. The default
   configuration keeps the hook disabled.

   Fusing the 12 strict flow observations at frames 88,340–88,351 replaced two missing and ten
   predicted trajectory frames. The candidate passed all 59 applicable framework checks, including
   coordinate noise, missing windows, observation retention, decoder sensitivity, restart
   sensitivity, and determinism. However, the resulting possession CSV was byte-identical to the
   champion and had the same label SHA. The ablation is therefore retained only as a validated
   no-op diagnostic; it adds no submission value and is not promoted.

   Video targeting now uses a counterfactual decoder-influence report rather than missingness near
   a boundary alone. The report shortlists missing-ball transitions, injects realistic 0.3 control
   support without forcing strong control, and independently tests incoming-team evidence before
   the boundary and outgoing-team evidence after it over 5, 10, 20, and 40 frames. Each boundary
   records the minimum support window that moves it, maximum shift, missing frames on each side,
   and an actionable priority.

   This analysis explains the frame-88,340 no-op before video inference is attempted. Incoming A
   support before the boundary does not move it at 5 or 10 frames and removes the matching
   transition under longer interventions; outgoing H support after it can delay the switch by at
   most eight frames. The recovered path instead supplied A evidence after the switch, which has
   no counterfactual decision value.

   The highest-value new targets are:
   - frame 48,645 (A→H): 70 missing post-boundary frames; outgoing A support delays the switch by
     exactly 5, 10, 20, or 40 frames;
   - frame 131,121 (H→A): 71 missing pre-boundary frames; incoming A support advances the switch by
     exactly 5, 10, 20, or 40 frames;
   - frame 147,517 (A→H): 71 missing post-boundary frames; outgoing A support delays the switch by
     5, 10, or 20 frames.

   Expensive video recovery should now run only where a recoverable path overlaps the actionable
   side and supports the counterfactual team. This avoids spending GPU time on evidence that cannot
   affect possession and prevents post hoc weakening of decoder thresholds.

   Applying this policy to the top targets produced one promotable path. At frame 48,645, SoccerNet
   detected an anchor-connected 45-frame track with 100% frame support, 0.61 median confidence,
   sub-0.67 m maximum Kalman uncertainty, plausible 0.56 m/s median motion, and a 0.45 m median
   distance to the expected away player. The pre-boundary target at 131,121 abstained, while the
   147,517 path was rejected for low confidence and 2 m uncertainty.

   The actionable 40 post-boundary frames at 48,645–48,684 are now fused as source code 4 through
   the default configuration. As predicted by the counterfactual report, the A→H transition moves
   40 frames later, changing exactly those frames from H to A without changing possession count,
   team-switch count, dead-ball frames, restart localization, or restart ownership.

   The promoted candidate passed all 59 framework checks against a freshly regenerated tracking-only
   reference. Decoded-boundary majority persistence improved from 97.21% to 97.75%, strong
   persistence from 94.41% to 94.94%, 50%-retention frame agreement from 96.47% to 96.49%, and
   50%-retention boundary recall from 61.11% to 62.22%. This is the first video component promoted
   into the champion stack; all other video paths remain diagnostic or rejected.

   The counterfactual influence map was regenerated after promotion. The corrected transition now
   appears at frame 48,685 and remains sensitive to additional outgoing-A evidence. It is excluded
   from repeated targeting because it is the same already-modified transition; repeatedly extending
   a path merely because the resulting boundary remains sensitive would create self-reinforcing
   drift. After that exclusion, frames 131,121 and 147,517 remain the highest-priority independent
   targets.

   Fixed 5- and 10-frame actionable hypotheses were tested for both remaining targets with complete
   detector support, 0.30 median raw detector confidence, 1 m maximum RTS uncertainty, 4 m maximum
   expected-team distance, and a 1 m team-distance margin required for promotion. Backward recovery
   at frame 131,121 abstained for both windows: only 9.1% and 37.5% of the anchor-connected windows
   contained away-geometry candidates, so no tracklet was formed. Forward recovery at frame 147,517
   had complete detector and away-player support, but median detector confidence was only 0.10 and
   0.12, while maximum uncertainty was 1.35 m and 1.03 m. Both variants were rejected without
   fusion. No combined variant exists, and the promoted submission remains byte-identical.

   The next independent target at frame 48,596 was then tested with a fixed five-frame incoming-A
   hypothesis and the nearest reliable right anchor at frame 48,600. SoccerNet produced candidates
   on six of nine anchor-connected frames, but none passed the away-player distance and team-margin
   gates. The selector therefore abstained without forming a tracklet.

   A detector-only feasibility screen covered the remaining lower-priority targets. Frames 138,906
   and 30,239 had zero expected-team geometry coverage over their five actionable frames. Frame
   61,541 had complete home-team geometry support and 0.53 median best-candidate confidence, so it
   advanced to anchored Kalman/RTS recovery. Its five actionable frames passed every evidence gate:
   complete detector support, 0.53 median raw confidence, 0.69 m maximum uncertainty, 1.15 m maximum
   distance to a home player, and a team-distance margin above 57 m.

   Fusing frames 61,536–61,540 moved the A→H boundary five frames earlier exactly as predicted,
   without changing possession count, switch count, dead frames, restart localization, or restart
   ownership. The candidate passed all 59 non-unit robustness checks plus unit tests. It was still
   rejected for promotion under the predeclared multi-proxy rule: decoded-boundary persistence and
   retention boundary recall were unchanged, while 50-percent retention frame agreement decreased
   from 96.4890% to 96.4865%. The small coordinate-noise agreement gains were insufficient to offset
   the lack of independent proxy improvement. The 40-frame video champion remains unchanged.

### Starter asset audit

The official starter Python files and all three notebooks have been reviewed. The current pipeline
already incorporates their core alignment and modeling guidance: ATD frame IDs are one-based,
submission frames are zero-based, pitch distances require metric scaling, player slots are
temporary tracklets, ball coverage is sparse, Smart Tagging intervals are noisy phase windows,
and public-window smoothing selection does not generalize to the full match.

The audit also identified additional experiment opportunities:

- preserve Smart Tagging restart `Type`, `Half`, and other context rather than retaining only team;
- infer ownership for the 32 of 59 set pieces tagged `N/A`, including all 26 throw-ins and all six
  kickoffs, using confidence-gated post-restart geometry and adjacent phase agreement;
- use restart type for confidence requirements and diagnostics, while retaining unknown ownership
  when evidence conflicts;
- use tactical homography later for video overlays, transition diagnostics, and selective ball
  recovery;
- consider the separately timed broadcast tracking only after tactical-only recovery is stable;
- use interval-level decoding, as recommended by the starter baseline, through the planned
  semi-Markov experiment.

The next experiment family is `restart-ownership`. It must calibrate only against organizer
Smart Tagging teams as weak proxy labels, apply inference only to `N/A` events, pass the automated
label-free framework, and record promoted and rejected variants in `experiments/results.csv`.

The first `restart-ownership` variant is promoted. `strict-geometry-v1` preserves restart type,
uses type-aware eligibility and prior duration, and requires at least ten reliable votes in the
first second after a localized restart. A vote requires a player within 2 m plus either a 1 m
team-distance advantage or a local-density advantage; at least 70% must agree, and a conflicting
adjacent phase rejects the inference. This resolves seven of 32 unknown events: four throw-ins,
two post-goal kickoffs, and one starting kickoff. When replayed against known organizer teams, it
covers six of 27 events and agrees on all six. All seven inferred owners survive coordinate-noise,
missing-observation, and 50%-retention tests with the same team. The variant passes all 49
automated checks, changes 15 frames, and leaves possession and short-spell counts unchanged.

Two broader variants were evaluated before closing the family. `balanced-geometry-v2` relaxed the
minimum to seven votes, 2.5 m distance, and 0.75 m advantage; proxy coverage rose to 25.9%, but
accuracy fell to 85.7%. `broad-geometry-v3` used five votes, 3 m, and 0.5 m; coverage rose to
40.7%, but accuracy fell to 72.7%. Both passed every robustness check and failed only the
predeclared 90% proxy-accuracy requirement. They are rejected and recorded in the experiment
ledger. `strict-geometry-v1` remains promoted, establishing that the family is precision-limited
rather than perturbation-limited.

## 1. Objective

Build a reproducible system that assigns one label to every tactical-video frame:

- `H`: home team has controlled possession while the ball is live
- `A`: away team has controlled possession while the ball is live
- `D`: the ball is dead
- `X`: the system abstains

The final output must contain exactly 199,319 ordered frame rows and is ranked primarily by
`spellF1`. The design must therefore prioritize coherent possession segments and accurate
transitions over isolated frame accuracy.

The target is to improve on the published `baseline_smooth50` score of 0.2433 spellF1 without
overfitting to the 5-minute public ground-truth window.

## 2. Strategy

Use the provided tactical tracking, Smart Tagging, and homography as the primary data sources.

The system will have four stages:

1. Load and align the provided assets.
2. Repair noisy ball observations and derive frame-level geometric features.
3. Convert those features into evidence for home control, away control, and dead ball.
4. Decode the evidence into temporally coherent possession spells.

The first complete version should be deterministic and rule-based. A learned model is an optional
extension only after the rule-based pipeline is reliable and reproducible.

## 3. Constraints

### Competition constraints

- Do not hand-label or hand-correct any frame from the provided match.
- Do not use human-produced event, possession, tracking, or annotation data about the provided
  match.
- Use the public 5-minute labels only to verify alignment and implementation correctness, not to
  select thresholds or model variants.
- Do not commit or redistribute the match data, video clips, frame-level tracking exports, or
  derived data that could reconstruct the source assets.
- The public repository must regenerate the exact submitted CSV.
- All required models and external resources must be publicly accessible to the organizers.

### Engineering constraints

- A missing ball observation must not automatically imply `D`.
- A nearest-player change must not automatically imply a possession change.
- Track slots must be treated as temporary team-affiliated tracklets, not persistent player
  identities.
- Broadcast video is optional and is not aligned with the tactical video.
- The tactical frame index is the authoritative timeline for every intermediate result and output.

## 4. Proposed Repository Structure

Add implementation files only when the corresponding milestone begins:

```text
possession/
  __init__.py
  assets.py             # Asset discovery and validation
  tracking.py           # ATD parsing into frame-aligned arrays
  tagging.py            # Smart Tagging parsing and frame alignment
  trajectory.py         # Ball cleanup, interpolation, velocity, confidence
  features.py           # Team-control and dead-ball features
  evidence.py           # H/A/D frame-level evidence scores
  decoder.py            # Temporal state machine or semi-Markov decoder
  submission.py         # CSV generation and strict validation
  diagnostics.py        # Aggregate diagnostics without exporting restricted data
  cli.py                 # Reproducible command-line entry point
configs/
  default.json          # Documented, frozen parameters
```

Tests should cover the core transformations using synthetic arrays and tiny generated fixtures.
They must not include competition data.

Generated caches and submissions should live under ignored directories such as:

```text
artifacts/
cache/
submissions/
```

Before adding these paths, verify that they are excluded by `.gitignore`.

## 5. Data Pipeline

### 5.1 Asset loading

Use `games.py` and the existing loaders wherever possible.

At startup, validate:

- tactical ATD metadata and raw tracking exist;
- tactical Smart Tagging exists;
- tactical homography exists;
- the tactical video exists only if video recovery is enabled;
- the frame count is exactly 199,319;
- player track slots have a known team assignment;
- every loaded interval and array can be represented on the tactical frame timeline.

Fail explicitly when a required asset is unavailable or malformed. Do not silently substitute an
empty array or produce a success-shaped submission.

### 5.2 Tracking representation

Convert the raw ATD stream into frame-aligned arrays:

```text
ball_xy          [T, 2]
ball_speed       [T]
player_xy        [T, P, 2]
player_speed     [T, P]
player_team      [P]
player_valid     [T, P]
```

Coordinates should remain normalized while loading and be converted to metres before calculating
distances or speeds. Use the pitch dimensions exposed by the match metadata rather than hard-coded
field dimensions.

Preserve missing values as `NaN` and maintain explicit validity masks.

### 5.3 Smart Tagging representation

Map each Smart Tagging interval to tactical frames with `round(seconds * 25)`.

Derive frame-level context including:

- interval code;
- tagged team, including explicit `N/A`;
- set-piece type;
- half;
- attacking and defensive transition indicators;
- confidence based on whether multiple compatible tags overlap.

Tags provide priors, not final labels. A tagged team must not overwrite strong contradictory
tracking evidence without transition confirmation.

## 7. Ball Trajectory Repair

Ball observations are missing on a large portion of the match, so trajectory repair must be
confidence-aware.

### 7.1 Raw observation filtering

Reject a raw ball point when one or more of these conditions hold:

- position is outside a small tolerance around the pitch;
- instantaneous speed is physically implausible;
- it creates an isolated one-frame jump followed by an immediate return;
- it is inconsistent with both neighboring observations;
- the value is non-finite.

Use metric displacement and elapsed time, not pixels per frame.

### 7.2 Gap classification

Classify each missing interval:

- short: up to approximately 0.5 seconds;
- medium: approximately 0.5 to 2 seconds;
- long: over approximately 2 seconds.

These are initial football-informed ranges, not values to tune on the public labels.

### 7.3 Gap handling

For short internal gaps:

- interpolate only when both endpoints are valid;
- require plausible displacement and endpoint velocity;
- assign lower confidence than observed points.

For medium gaps:

- extrapolate only briefly from a stable pre-gap trajectory;
- prefer possession-state continuity over an uncertain position;
- use nearby player motion to estimate which team remains in control;
- keep positional confidence low.

For long gaps:

- do not invent a full ball trajectory;
- rely on the temporal decoder, player geometry, and Smart Tagging;
- use `X` only when abstention is safer and its forward-fill behavior is understood.

### 7.4 Derived trajectory features

Compute:

- speed and acceleration;
- direction of travel;
- recent direction change;
- distance from pitch boundary;
- observed/interpolated/missing status;
- observation confidence;
- gap age and distance to the next observation;
- stationary-ball evidence.

## 8. Team-Control Features

For each frame and team, compute:

- distance from the ball to the nearest player;
- distances to the nearest two and three players;
- number of players within 1, 2, and 4 metres;
- nearest-player speed;
- nearest-player approach velocity toward the ball;
- local numerical advantage around the ball;
- persistence of proximity over recent frames;
- whether the ball and nearest player move coherently;
- whether another team has only a brief, isolated proximity advantage.

Core comparative features:

```text
distance_advantage = nearest_away_distance - nearest_home_distance
density_advantage = nearby_home_count - nearby_away_count
approach_advantage = home_approach_velocity - away_approach_velocity
```

Positive and negative signs must be documented and tested to avoid accidentally reversing teams.

When the ball position is unavailable, derive weaker proxy features:

- team centroid movement;
- local player convergence;
- which team's nearby tracklets move coherently with the last ball direction;
- Smart Tagging team prior;
- last high-confidence possessor.

## 9. Dead-Ball Evidence

Create a dead-ball evidence score from:

- set-piece and restart tags;
- kickoff and halftime boundaries;
- ball outside pitch bounds;
- long stationary-ball periods;
- prolonged low player speed around a restart location;
- disappearance of the ball near a boundary followed by a tagged restart;
- intervals before kickoff, during halftime, and after the final whistle.

Do not use ball missingness alone as dead-ball evidence.

The decoder should represent dead-ball stretches explicitly. `H -> D -> H` is two possessions,
even if the same team restarts.

## 10. Frame-Level Evidence

Produce three scores for each frame:

```text
home_evidence[T]
away_evidence[T]
dead_evidence[T]
```

Evidence should combine interpretable components rather than immediately applying hard labels.
For example:

```text
home_evidence =
    proximity_weight * home_proximity
  + density_weight * home_density
  + motion_weight * home_motion_coherence
  + tag_weight * home_tag_prior
  + continuity_weight * recent_home_control
```

The relative-motion benchmark supports using a 2 m/s ball-player speed
threshold for high-confidence handovers: one-second geometric persistence
increased from 84.3% for proximity-only events to 96.1%. A continuous
motion-coherence bonus was also tested, but disabled because candidate-event
persistence was marginally better with zero weight (74.4% versus 74.2%).

Robustness testing qualifies this result. Thresholds from 1.5 to 3 m/s retained
at least 95.3% persistence and agreed with the 2 m/s decoder on at least 99.6%
of frames. Deterministic missing-ball masking retained 98.97% to 99.91% frame
agreement. However, adding 0.10 m ball-coordinate noise reduced motion-gated
boundary recall to 87.2%, compared with 96.2% for proximity-only control; at
0.25 m the recalls were 76.9% and 87.6%. A track-consistent five-frame
estimate, an eleven-frame estimate, and thresholds of 3 and 4 m/s did not
close this robustness gap without weakening other behavior. Relative-motion
gating is therefore retained as an available experiment but disabled in the
promoted configuration. The acceleration-aware proximity model remains the
stable model until relative motion can pass both persistence and noise tests.

Each component should be normalized to a comparable scale and be available in diagnostics.

Avoid converting low-confidence evidence into `D`. Low confidence and dead ball are separate
concepts.

## 11. Temporal Decoder

### 11.1 Initial decoder

Implement a deterministic finite-state machine with:

```text
HOME_CONTROL
AWAY_CONTROL
DEAD_BALL
```

Maintain:

- current state;
- candidate next state;
- consecutive candidate evidence;
- confidence in the current state;
- time since the last reliable ball observation;
- time since the last confirmed transition.

### 11.2 Live possession changes

Confirm `HOME_CONTROL -> AWAY_CONTROL`, or the reverse, only when:

- the candidate team has sustained geometric advantage;
- the advantage persists for a confirmation window;
- evidence is consistent with control rather than a single touch;
- no strong dead-ball evidence occurs between the states.

Stronger evidence can shorten the confirmation window. Examples include:

- the opponent remains closest while moving coherently with the ball;
- the ball changes velocity near an opponent and then travels with that team;
- multiple opponent players surround the ball after an interception;
- a compatible attacking-transition tag begins.

During a loose ball, deflection, or aerial duel, retain the last confirmed live possessor.

### 11.3 Live-to-dead transitions

Enter `DEAD_BALL` when:

- a reliable set-piece or stoppage interval begins;
- the ball leaves the pitch with corroborating context;
- several independent dead-ball signals persist.

Avoid declaring `D` on short tracking losses during active play.

### 11.4 Dead-to-live transitions

Exit `DEAD_BALL` when:

- a restart interval ends and the ball begins moving;
- a tagged restart provides a team prior;
- a team demonstrates sustained control immediately after the restart.

If the restart team is `N/A`, delay team assignment until geometric evidence is stable.

### 11.5 Optional second decoder

After the state machine works, evaluate a semi-Markov or Viterbi decoder with:

- frame-level emission scores;
- transition penalties;
- state-duration priors;
- separate penalties for `H <-> A`, live-to-dead, and dead-to-live transitions.

This should be an alternative decoder behind configuration, not a replacement until it is proven
more robust on non-public validation data.

## 12. Optional Tactical-Video Ball Recovery

Add video-based recovery only after the tracking-only system is complete.

Restrict inference to tactical frames where:

- ATD ball tracking is missing or rejected;
- tactical homography is valid;
- the surrounding frames indicate active play;
- the missing interval is likely to contain a meaningful transition.

Pipeline:

1. Detect candidate ball pixels.
2. Filter candidates using the projected pitch region.
3. Map accepted pixels to normalized pitch coordinates with inverse homography.
4. Reject candidates with excessive reprojection error or implausible world velocity.
5. Fuse accepted detections with ATD observations using confidence.

The first four diagnostic surfaces are implemented through `--video-report` and
`--video-detection-report`. Detector execution is local, uses a pinned public model in the MMM2
environment, disables Ultralytics synchronization, and processes only ranked tactical intervals.
No detector output is currently fused into the promoted trajectory.

Do not process the unsynchronized broadcast video in the first version.

Success criterion: recovered observations must improve transition evidence without substantially
increasing short false possessions.

## 13. Validation Plan

### 13.1 Structural validation

Every generated submission must satisfy:

- exact header `frame,label`;
- exactly 199,319 data rows;
- frames `0` through `199318` in order;
- labels only from `H`, `A`, `D`, and `X`;
- deterministic output for a fixed config;
- successful execution of the official validator.

### 13.2 Synthetic tests

Create synthetic tests for:

- constant home and away possession;
- one clean possession transition;
- a brief opponent touch that must not transition;
- a loose-ball interval;
- a dead-ball restart by the same team;
- a dead-ball restart by the other team;
- missing ball inside a possession;
- missing ball across a real transition;
- tracking identity fragmentation;
- outlier ball jumps;
- halftime reset;
- reversed team assignments.

### 13.3 Behavioral diagnostics

For every full-match run, report:

- total predicted home and away possessions;
- possession transitions per minute;
- possession-duration percentiles;
- count of possessions shorter than 0.5, 1, and 2 seconds;
- total dead-ball duration;
- ball observation and interpolation coverage;
- transitions during missing-ball intervals;
- transitions supported by strong, medium, or weak evidence;
- disagreement rates between geometry and Smart Tagging;
- use and duration of `X`.

These statistics should be printed or saved as aggregate JSON. Do not publish frame-level derived
competition data.

### 13.4 Public ground truth

Use the public window only for:

- confirming frame alignment;
- detecting home/away reversal;
- confirming `D` semantics;
- verifying that the scorer and submission format work;
- inspecting clear implementation failures.

Do not choose smoothing windows, thresholds, feature weights, or decoder variants based on which
one scores best on the 17 public possessions.

### 13.5 Robustness checks

The automated runner applies:

- 0.10 m and 0.25 m deterministic ball-coordinate noise;
- periodic 10-frame and 25-frame missing-observation windows;
- 50 percent observation retention;
- plus or minus 10 percent decoder-duration perturbations;
- plus or minus 20 percent restart-search perturbations;
- plus or minus 25 percent restart-motion-threshold perturbations.

Each perturbation checks frame agreement, possession-boundary recall, and feature-specific
stability against the clean candidate. It also checks deterministic output, short-spell
regressions, strong-control persistence, and restart speed gain. Large structural changes or a
failed policy threshold block promotion. The public window is not part of this runner.

Framework v3 additionally prevents strong global averages from hiding weak match segments. It
reports boundary persistence and same-direction boundary recall separately for the first and
second halves, H→A and A→H transitions, restart-proximal transitions, and open play. Cohorts with
at least 20 transitions are hard gates under every observation perturbation and decoder/restart
sensitivity variant; smaller cohorts remain diagnostic. The half split comes from the supplied
tracking alignment, and restart proximity uses a fixed 75-frame radius.

The promoted champion passes all 135 framework-v3 checks. First/second-half majority persistence
is 96.94%/98.75%, and H→A/A→H majority persistence is 97.85%/97.65%. The weakest eligible
observation cohort is second-half same-direction recall under 50-percent observation retention at
59.26%, above the predeclared 40% threshold. The weakest decoder sensitivity cohort is A→H recall
at 81.61% under a 20-percent transition-cost increase, above the 80% threshold. A clean run from
the default configuration reproduces the promoted CSV byte-for-byte.

### 13.7 SoccerNet external ground truth

SoccerNet Game State Reconstruction v1.3 is the preferred source for external ball, player, team,
and pitch annotations. Although the task is described primarily in terms of athlete game state,
its native `Labels-GameState.json` schema also represents the ball as an object whose
`attributes.role` is `ball`. Those human ground-truth image boxes can replace planned
manual ball clicks. A missing ball object is treated as unknown or unlabeled, never as evidence
for dead ball.

The current SoccerNet package downloads migrated gated datasets from Hugging Face. Install or
update `SoccerNet` and `huggingface_hub`, authenticate once with `hf auth login`, and use the
standard SoccerNet download API. Dataset files stay outside this repository.

The importer maps the official `train`, `valid`, and `test` directories to development,
validation, and untouched test respectively. It rejects labels older than v1.3, ignores the
challenge split, records the source label path and role selector, and inventories ball, player,
team, pitch-coordinate, camera, and pitch annotations. Generate the manifest after the gated data
has been downloaded:

```bash
python -m possession.external_soccernet \
  --root /path/to/SoccerNetGS \
  --output artifacts/soccernet_gamestate_manifest.json
```

The current SoccerNet-v3D checkpoint reports `data=snv3.yaml` in its training metadata but does
not identify the exact training clips. Detector accuracy measurement is therefore disabled in the
SoccerNet configuration until clip-level overlap is resolved. The labels remain usable for
geometry and temporal-policy development, while detector evaluation must use only a proven
disjoint split.

The separate SoccerNet Tracking MOT files contain ground-truth boxes for players, goalkeepers,
referees, balls, and other people, but the public MOT rows do not encode object class. They are not
automatically accepted as ball ground truth unless a separate authoritative role mapping is
available. GSR labels are preferable because the ball role is explicit.

The downloaded v1.3 development and validation corpus contains 115 clips, 86,250 frames, and
81,831 ball boxes. Frames with multiple simultaneous ball-role boxes are retained as ambiguous
provenance but excluded from metrics; this leaves 39,537 usable development boxes and 39,978
usable validation boxes. The test archive remains compressed and is not included in the manifest.
Label-only calibration statistics cover ball image diameter, missing-label runs, pitch speed,
pitch acceleration, and nearest-player pitch distance. They do not run the detector and therefore
cannot leak detector training overlap into threshold selection. Pitch-derived metrics accept only
bottom-middle projections within the regulation 105×68 m field bounds; out-of-pitch projections
are counted and rejected rather than inflating motion or distance thresholds.

External parameter selection is deterministic and development-only: speed uses development p99
plus 10 percent, acceleration uses development p90 plus 10 percent, missing-gap duration uses
development p75, and Kalman process acceleration uses development p75. Values are rounded upward
before validation and must cover the corresponding frozen validation statistics. The generated
compliance configuration explicitly sets `video_ball_observations_path` to null, so none of the
40 match-specific promoted frames can enter inference. The current champion remains an
experimental result and is not compliance-certified; only a clean output regenerated from the
tracking-only compliance configuration is a certification candidate.

The command-line default now points to `configs/compliant_tracking.json` and writes
`submissions/possession_compliant.csv`. A load-time invariant rejects any tracking-only compliance
configuration that re-enables `video_ball_observations_path`. The prior `configs/default.json`
and `submissions/possession_mvp.csv` remain preserved only as experimental comparison evidence.

The tracking-only compliance candidate passes all 135 framework-v3 checks and regenerates
byte-for-byte through the default CLI. It contains 199,319 frames, 319 possessions, one possession
shorter than one second, and zero video-derived frames. Its SHA-256 is
`4390fd4bd6509eb95fc35f24be69384feb6f365dc6dfb3cff087194d9e41dd72`. These checks
establish reproducibility and the absence of match-specific video fusion, not a complete
compliance certificate. Only the motion/gap overrides have external-selection provenance; the
other decoder, control, and restart settings are inherited from the earlier experimental
configuration. Their provenance still needs review or external-only reselection before claiming
that every tunable setting satisfies the clarified rules. Final eligibility remains subject to
the organizers' interpretation and the competition data license.

### 13.8 External transfer validation

The post-selection transfer protocol is frozen before inspecting Metrica outcomes. Metrica Sample
Games 1–2 are validation-only; Sample Game 3 remains untouched. Deterministic masks test 0.2,
0.75, and 2.0-second ball gaps, while synchronized pass, recovery, set-piece, carry, and shot
events test whether the nearest-player team matches the event team. Failed checks reject transfer
support but never trigger parameter adjustment.

The corrected Metrica validation still passes all 14 checks. An earlier report applied 50 fps
frame counts directly to 25 fps data, doubling the configured time windows. The runner now scales
frame durations and cumulative-emission transition costs, isolates periods, and distributes
non-overlapping masks across the entire match rather than only its first quarter. Both games
recover 100 percent of 0.75-second masks; p90 error is 1.08 m and 1.36 m. For two-second gaps,
the pipeline abstains on 81.3 percent and 80.9 percent of frames. These results supersede the
earlier timing-invalid report. Nearest-team agreement remains approximately 96.5 percent, but it
does not measure production decoder accuracy. The Metrica sample repository is pinned to commit
`e706dd506b360d69d9d123d5b8026e7294b13996`; its README permits community use with attribution,
but it contains no standalone license file, so organizer confirmation remains advisable for a
strict licensing record.

### 13.9 Paired external candidate benchmark

The comparison evaluates six configurations on the same Metrica Games 1-2, source timestamps,
period-isolated masks, and 2,472 scoring-only event anchors. Events never enter inference.
Production evidence and decoder functions are reused; the array adapter is tested against the
tracking-file adapter. Smart Tagging and restart priors are absent, so this is not an evaluation of
the full tagging-assisted challenge pipeline. State-machine variants are architecture ablations
of the saved configurations, not reconstructed historical checkpoints. Legacy match-specific
video fusion is disabled for every candidate.

| Candidate | Clean event accuracy | Masked event accuracy | Live spells | Spells under 1 s |
| --- | ---: | ---: | ---: | ---: |
| Raw nearest team | 96.48% | 95.15% | 4,201 | 2,471 |
| Nearest team, +/-1 s smoothing | 88.15% | 87.82% | 1,252 | 253 |
| Legacy tracking, state machine | 80.95% | 80.58% | 520 | 14 |
| Legacy tracking, semi-Markov | 90.74% | 90.57% | 682 | 52 |
| External tracking, state machine | 80.78% | 80.95% | 519 | 14 |
| External tracking, semi-Markov | 90.74% | 90.57% | 684 | 54 |

These are sparse live-event correctness measures and fragmentation diagnostics, not frame macro
F1, dead-ball accuracy, or possession spell F1. The raw baseline wins at event instants while
fragmenting heavily. External semi-Markov improves anchor accuracy by 9.95 percentage points over
the external state-machine ablation, but matches legacy semi-Markov exactly on clean anchors.
The masked subset contains only 38 anchors; small decoder gains on that subset are inconclusive.

For 0.75-second masks, legacy repair recovers 51.3%/50.8% of frames, while external repair recovers
100%/100%. The fraction of all masked frames reconstructed within 2 m rises from 50.8%/50.1% to
94.8%/94.7%, a 44.1/44.7-point gain. This coverage-aware measure prevents a conservative candidate
from appearing superior simply by abstaining on difficult frames. External p90 error among
recovered frames is higher than legacy's, as expected from the expanded coverage.

Paired 95% bootstrap intervals use fixed-seed one-minute event blocks or whole masked windows
within each game; they do not establish generalization across matches. The 0.75-second within-2 m
gain has intervals of [40.8, 46.8] and [41.5, 47.8] percentage points. No candidate is promoted and
the submission/configuration hashes remain unchanged.

Regenerate the paired report with:

```bash
python -m possession.external_metrica_validation \
  --root /path/to/metrica-sample-data \
  --compare-candidates \
  --output artifacts/comparison_metrica_candidates.json
```

Next, obtain external possession-spell labels on development footage to calibrate recovery
transitions, dead-ball handling, and decoder costs against actual boundaries. Keep Metrica Game 3
and SoccerNet test untouched for the final frozen evaluation. Games 1-2 have now been inspected
repeatedly and are diagnostic comparison data, not a pristine final test.

### 13.10 Event-derived recovery references and candidate revisits

External datasets are not inherently unable to provide possession supervision. The missing piece
is a definition-aware reference adapter, not necessarily more manual ball annotations.
[Metrica's event definitions](https://github.com/metrica-sports/sample-data/blob/master/documentation/events-definitions.pdf)
describe same-team possession transfers, losses, recoveries, and out-of-play/restart events.
They can supply event-derived possession-chain and dead-ball references, with ambiguous intervals
kept unknown and coverage reported. Such references must be distinguished from the challenge's
possession ground truth.

The recovery adapter pairs a recovery with exactly one opposing-team ball loss whose endpoint
is within 0.2 seconds in the same period. The loss endpoint and recovery timestamp bracket the
boundary uncertainty. Conflicting, ambiguous, duplicated, and unpaired cases are accounted for
rather than converted to invented labels. Matches between predicted and reference boundaries
are one-to-one and same-direction. Subsequent known events censor post-recovery support; short
reference intervals are reported separately. Events remain scoring-only inputs.

The two Metrica games provide 333 paired references out of 526 recoveries. There are 249 eligible
0.75-second recovery-centred masks, covering 251 reference anchors. The six frozen configurations
are evaluated on identical inputs without retuning:

| Candidate | Correct team at paired recovery | Boundary recall within 0.5 s | Correct team at masked recovery |
| --- | ---: | ---: | ---: |
| Raw nearest team | 91.29% | 62.76% | 0.00% |
| Nearest team, +/-1 s smoothing | 79.28% | 32.13% | 68.13% |
| Legacy tracking, state machine | 49.85% | 33.63% | 20.32% |
| Legacy tracking, semi-Markov | 75.98% | 42.64% | 41.43% |
| External tracking, state machine | 49.85% | 33.63% | 32.67% |
| External tracking, semi-Markov | 75.98% | 42.64% | 47.41% |

External repair improves masked recovery accuracy by 6.62 and 5.00 percentage points over legacy
semi-Markov in Games 1 and 2. Paired within-match bootstrap intervals are [2.65, 11.04] and
[0.00, 10.35] points; the second game is inconclusive at this precision. Clean recovery accuracy
is unchanged. On paired clean references, all semi-Markov errors select the previous team rather
than dead ball. Only 42.64% of recovery boundaries match within 0.5 seconds, even though event
accuracy is 75.98%. Among boundaries matched within a two-second search window, median signed
offset is -0.24 seconds in both games. Raw nearest-team boundaries are also often early, so this
does not by itself prove decoder anticipation or timestamp misalignment. Blindly reducing global
transition delays would be unjustified.

Revisit ideas in this order:

1. Local transition evidence and adaptive transition penalties, calibrated externally against
   both missed and early switches; distinguish instantaneous player control from continuing team
   possession during a pass.
2. Soft player-relative motion/co-moving evidence, rather than reinstating the rejected universal
   2 m/s hard gate that can suppress first touches and fast interceptions.
3. Boundary refinement and the existing HSMM approach once event-derived or independent spell
   references are available. Old rejection by challenge-match persistence proxies is not an
   external accuracy verdict; stronger generic duration smoothing alone is not supported.
4. Explicit dead-ball evaluation using event-derived out-of-play/restart references, then a
   separate tagging-assisted experiment. Provider annotations used as feature inputs must not
   also be presented as independent evaluation truth, and their quality differs from challenge
   automatic Smart Tagging.
5. Video/Kalman uncertainty and fusion only after decoder weaknesses and checkpoint overlap are
   addressed. SAM3, optical-flow-only recovery, and generic detector ensembles stay lower priority.
   Old match-specific artifacts and challenge-selected gates cannot be reused as compliant results.

The [official PFF 2022 World Cup release](https://www.blog.fc.pff.com/blog/pff-fc-release-2022-world-cup-data)
is a promising larger external source: it provides tracking, events, metadata, and specification
PDFs for all 64 games. Its official access form, usage terms, possession semantics, and timing
must be checked before import. Event identifiers must not be assumed to equal team possession
spells. Preserve match-level split assignments across PFF and StatsBomb to avoid
using the same match for both development and evaluation. StatsBomb possession identifiers are
useful event-level supervision but do not independently provide frame-level ball-alive truth.
SoccerNet GSR still primarily supports geometry/video validation, not ready-made
possession-spell ground truth.

Run the recovery-focused analysis with:

```bash
python -m possession.external_metrica_validation \
  --root /path/to/metrica-sample-data \
  --recovery-analysis \
  --output artifacts/analysis_metrica_recoveries.json
```

No candidate is promoted and the challenge submission remains unchanged. A subsequent tuning
phase must explicitly designate external development data instead of retuning on these diagnostic
validation outcomes.

### 13.11 Authorized provider-reference calibration

The user approved a new calibration split: Metrica Sample Game 1 is development, Game 2 is
diagnostic validation, and Game 3 remains unevaluated final test. The earlier frozen transfer and
recovery reports are retained as historical audits; Game 2 is not described as a pristine test.

The possession-chain adapter derives H/A/D/unknown intervals from provider events. Same-team
pass/control chains can bridge observed event gaps, but losses, shots, stoppages, conflicting
claims, and period boundaries censor unsupported continuation. Confirmed out-of-play/restart
pairs support dead intervals. Loss/recovery timestamp brackets and a 0.12-second boundary guard
remain unknown for frame scoring. A live spell enters the reference-conditioned IoU score only
when its start/end are supported and its interior is unambiguous. Predictions retain their full
extent; they are not clipped to improve overlap.

Game 1 has 90.56% known-frame coverage and 216 complete provider-reference live spells. Game 2
has 88.96% coverage and 155 such spells. These are provider-event-derived references, not the
challenge's possession ground truth.

The first development-only grid tested twelve local causal handover variants. Contact requires
current evidence, geometric separation, and sufficient past support; reliable observed or
interpolated ball positions may contribute, but unobserved, extrapolated, low-confidence, and
video-fused points may not. None safely beat the baseline under the development objective and
fragmentation guards, so the saved first candidate remains the baseline. The optional mechanism
is available for later experiments but is not enabled in the selected configuration.

That experiment exposed the larger no-tagging weakness: only about 0.3% of confirmed dead frames
were detected. A separate development-only family added an opt-in tracking stoppage signal.
It never infers dead ball from missing observations alone: it requires sustained missing ball,
sustained low median player speed, and a sufficiently stationary player fraction. Existing tag
evidence is retained via a maximum rather than double-counted by addition.

The second search evaluated six stoppage hypotheses, then twelve handover variants around the
development-selected stoppage seed. It froze the simple stoppage-only candidate before diagnostic
validation: at the 50 fps configuration timeline, missing ball must persist for 50 frames,
collective motion is checked over 50 frames, at least 80% of that window must meet the slow-motion
condition, median player speed must be at most 1.5 m/s, and at least half the players must be
stationary. The dead-evidence weight is 0.9. Frame counts rescale to 25 fps for Metrica.

| Diagnostic Game 2 metric | Baseline | Frozen external candidate |
| --- | ---: | ---: |
| Known-frame macro F1 | 0.464 | 0.784 |
| Dead-ball recall on known references | 0.34% | 59.26% |
| Reference-conditioned spell F1 at IoU 0.5 | 0.396 | 0.508 |
| Correct team at paired recovery | 71.63% | 73.05% |
| Recovery-boundary recall within 0.5 s | 44.68% | 45.39% |
| Reference-conditioned spell F1 with recovery masks | 0.327 | 0.449 |
| Correct team at masked recovery subset | 45.00% | 47.00% |
| Live spells shorter than one second | 22 | 24 |

All thirteen diagnostic validation guards pass. The gain is primarily dead-ball separation, not
a solved handover problem. The causal handover variants remain rejected by this development
search rather than being promoted on isolated recovery gains.

Generate the candidate and evidence with:

```bash
python -m possession.external_handover_calibration \
  --root /path/to/metrica-sample-data \
  --protocol configs/metrica_stoppage_handover_calibration.json \
  --output-config configs/metrica_stoppage_handover_candidate.json \
  --output-report artifacts/calibration_metrica_stoppage_handover.json \
  --output-references artifacts/metrica_stoppage_possession_references.json
```

The configuration/report are experimental and not automatically promoted. Both new mechanisms
default off, existing configurations remain valid, and the default challenge submission is
unchanged. Only the new overrides have external calibration provenance; inherited control,
decoder, and tagging settings still require their separate audit or external reselection before
full compliance is claimed. Game 3 and SoccerNet test have not been evaluated.

### 13.12 Frozen transfer check and promotion verdict

The candidate and baseline configuration hashes were locked before the check. The final protocol
evaluated diagnostic Game 2 and previously unevaluated Game 3 under no tags, coarse restart tags,
and degraded restart tags, each with clean input and predetermined 2-second/5-second known-live
ball masks. No model parameter or acceptance threshold was changed after results were inspected.

The external Game 3 reader resolves frame-specific EPTS channel layouts and substitutions by
player identity rather than assuming all 35 roster entries appear in every 22-player row. Native
JSON events, including carries, are mapped to the same provider-reference schema. Per-type
restart windows and vote counts are time-normalized when moving 50 fps configurations to 25 fps.

Tag simulation uses SET PIECE records only. It does not use possession-reference intervals or
loss/recovery labels to construct features. This is still a simulation from provider annotations,
not an independently measured automatic tagging system. Primary frame scores exclude three
seconds around provider restart clocks, and primary spell scores exclude spells with shared
restart endpoints. The unfiltered shared-endpoint provider score is retained as a secondary
proxy, not presented as independent truth.

| Clean coarse-tag condition | Baseline | Candidate |
| --- | ---: | ---: |
| Game 2 primary macro F1 | 0.573 | 0.798 |
| Game 2 primary spell F1 at IoU 0.5 | 0.437 | 0.462 |
| Game 3 primary macro F1 | 0.659 | 0.858 |
| Game 3 primary spell F1 at IoU 0.5 | 0.467 | 0.492 |

The primary spell populations contain 100 Game 2 spells and 210 Game 3 spells. The roughly
0.025 gain beyond coarse tags is encouraging but much smaller than the earlier no-tagging
reference-conditioned gain; populations and feature inputs differ, so the scores must not be
interpreted as a direct estimate of challenge spell F1.

The live-missingness safety checks fail on Game 2. With coarse tags and 100 known-live masks,
false dead-ball frame fractions rise from 0.52% to 3.28% for two-second gaps and from 0.21% to
4.30% for five-second gaps. The fraction of windows containing false dead ball rises from 2% to
11% and from 2% to 15%, respectively. These exceed the fixed incremental safety limits of one
percentage point of masked frames and five points of affected windows. Game 3 passes its frozen
stress gates, illustrating that clean transfer and safety can vary by match.

Overall, the external check completes but passes only 76 of 94 gates. The failed gates are
Game 2 live-missingness safety checks, not code or test failures. Game 3 is now consumed by this
fixed final evaluation and must not be reused as a tuning set. A future revised model requires
new external holdout evidence rather than selecting against this final result.

The automatic challenge check uses only tracking and automatic tagging, with no provided video
review or possession-ground-truth scoring. It produces a separate experimental CSV and passes
133 of 134 label-only robustness gates; the independently run full unit suite passes 98 tests.
The failed gate is dead-share difference: 10.47 percentage points versus the predeclared five
point maximum. The candidate emits 58,926 D frames versus the current default's 38,056, a change
of 20,870 frames. This is a structural-risk flag, not proof of accuracy without ground truth.

Promotion is blocked. Retain the current default submission; do not retune the candidate on
challenge outputs or relax the gates. The next development hypothesis is additional independent
corroboration for dead ball during long tracking gaps, not a blanket low-motion/missing-ball rule.
Inherited parameter provenance remains a separate compliance concern.

Reproduce the frozen external assessment with:

```bash
python -m possession.external_transfer_check \
  --root /path/to/metrica-sample-data \
  --output artifacts/frozen_stoppage_external_transfer.json
```

The reports are `artifacts/frozen_stoppage_external_transfer.json` and
`artifacts/validation_stoppage_challenge_transfer.json`. The failed experiment's
`submissions/possession_stoppage_transfer_check.csv` is diagnostic only, not the selected
submission.

### 13.13 Guarded development with long-live-gap hard negatives

The next development protocol fixes the earlier augmentation blind spot: the original synthetic
gaps lasted 0.75 seconds, while the rejected stoppage detector required at least one second of
missing ball. Game 1 selection now includes clean input, recovery-centred gaps, and 100 known-live
windows each at two and five seconds, under no tags and degraded restart tags. Games 2-3 and
challenge data are excluded from this selection. Fresh external holdout validation is pending.

Seven predeclared ablations compare the current baseline, the rejected uncorroborated stoppage
rule as a nonselectable diagnostic, corroborated stoppage, two gap-handover policies, and their
combinations. Every scenario must pass both IoU 0.5/0.75 spell-score guards, recovery and boundary
guards, short-spell guards, and live-gap false-D/window-fragmentation guards. Macro F1 is no longer
part of the selection objective; spell overlap receives the largest weight.

Corroborated stoppage requires the existing low-motion/missing-run evidence plus a recent observed
pitch exit or a pre-restart window. Exit evidence requires at least two consecutive observed
outside-pitch points; predicted points cannot create it. Observed in-pitch reappearance or a
subsequent restart invalidates old exit evidence. Trusted same-team gap endpoints veto the added
stoppage signal when neither an exit nor a restart inside the gap supports dead ball. This never
removes the baseline's existing tag/out-of-pitch evidence.

Gap-handover policies require two confidently opposing observed endpoint teams around a gap of
at most two seconds. Only reliable observed/interpolated contact can create additional support;
predicted or low-confidence points cannot. The augment ablation adds support to the existing
discount policy. The contact-guarded ablation restricts the cheap transition discount to current
contact and sufficient past support inside that bounded context, including 0.2 seconds after
reappearance. Outside those contexts, the baseline discount policy is unchanged.

The selected development-only candidate is `gap_contact_guarded`; corroborated stoppage is not
enabled in it. Its development objective is 0.44266 versus 0.43908 for baseline, and all 60 guards
pass. The augment candidate nearly ties it at 0.44264; this microscopic difference is not evidence
of superiority on unseen matches. The uncorroborated diagnostic fails twelve development guards,
so the expanded negative tests now reject it before any validation or challenge run.

| Game 1 development measure | Baseline | Selected gap candidate |
| --- | ---: | ---: |
| Clean/no-tag reference spell F1, IoU 0.5 | 0.4540 | 0.4611 |
| Clean/degraded-tag reference spell F1, IoU 0.5 | 0.4726 | 0.4784 |
| Clean/no-tag recovery-boundary recall within 0.5 s | 41.15% | 41.67% |
| Recovery-masked/no-tag boundary recall within 0.5 s | 25.52% | 26.56% |
| Two-second live-mask false-D frame fraction | 1.30% | 1.30% |
| Five-second live-mask false-D frame fraction | 1.432% | 1.432% |

There are trade-offs: nominal recovery-instant accuracy falls by one event out of 192 (0.52
percentage points), within the predeclared one-point guard. Recovery-masked/no-tag IoU 0.75 F1
also falls slightly, while most spell/timing scores improve. All scenarios have one fewer
subsecond live spell. These small development gains are not validated challenge improvements.

Run the development-only search with:

```bash
python -m possession.external_guarded_calibration \
  --root /path/to/metrica-sample-data \
  --output-config configs/metrica_guarded_candidate.json \
  --output-report artifacts/development_guarded_possession.json
```

The saved candidate and report carry explicit development-only provenance. There is no automatic
promotion, and the default challenge CSV still regenerates byte-for-byte. All 107 unit tests pass.
New signal flags default off, and historical experimental configurations retain their old
behavior. Fresh external holdout evidence would strengthen confidence, but is not itself a
challenge-rule requirement. Inspected external games may support an explicitly diagnostic release
decision. The inherited-parameter provenance audit remains necessary before a full compliance claim.

### 13.14 Bounded gap-candidate release decision

The user authorized a single promote-or-retain pass for the frozen gap candidate. No parameters
were changed, no further candidate search was performed, and no fresh/unseen-test claim was made.
The candidate configuration hash is
`de48bec40e7a80cb92b9106fce873a286a9222acedca169e88bb116d55306707`.

The release protocol compares against the current baseline on reused diagnostic Games 2-3 with
no/coarse/degraded tags, clean data, recovery-centred masks, and two-/five-second known-live masks.
All per-scenario non-regression and missingness checks pass. However, tagged clean spell F1,
averaged over IoU 0.5 and 0.75, has zero improvement in both games: 0.33603 vs 0.33603 in Game 2
and 0.35472 vs 0.35472 in Game 3. Thus 158 of 159 external gates pass; the sole failure is the
predeclared minimum mean spell gain of 0.002. Frame labels do change externally, but the matched
spell scores do not improve.

The automatic challenge rehearsal uses tracking/tagging only, with no video or possession
ground-truth evaluation. All 134 robustness gates pass; the separately run unit suite passes 111
tests. Crucially, the candidate CSV and its repeated generation are byte-identical to the current
submission, all with SHA-256
`4390fd4bd6509eb95fc35f24be69384feb6f365dc6dfb3cff087194d9e41dd72`. They contain
199,319 frames and 319 possessions. Switching configurations would not change the submitted answer
or its evaluation score for this match.

The provenance audit independently verifies the new gap overrides against the Game 1 experiment,
input hashes, and protocol hash, and verifies the three external kinematic overrides. Shared active
blocks remain unresolved: the existing history documents competition-game proxy selection for
semi-Markov costs/support, restart localization/ownership, and prediction history/horizon. These
are not certified externally reselected by merely evaluating the new candidate. The audit does not
claim protected possession truth was used and is not a legal/organizer eligibility ruling.
Retaining the current baseline does not certify its shared provenance.

Final decision: retain the current submission and close this gap candidate as a release rejection
(no submission effect and no external spell-score gain), not as a candidate indefinitely parked
awaiting another dataset. The provenance concern is a separate shared blocker. The default config,
default output, and current CSV remain unchanged.

The reproducible decision and supporting reports are:

- `artifacts/release_gap_decision.json`
- `artifacts/release_gap_external.json`
- `artifacts/release_gap_challenge.json`
- `artifacts/release_gap_provenance.json`

The diagnostic `submissions/possession_gap_release_candidate.csv` is identical to the current
submission and is not a new selected deliverable. The release gate declarations are preserved in
`configs/guarded_gap_release.json`.

### 13.15 Core calibration and reliability-conditioned rescue

The next bounded development experiment varies always-active live evidence and H/A transition
settings on Game 1 only. Dead/restart formulas and preprocessing remain fixed. Four evidence
profiles are followed by six transition/support profiles around the safe evidence winner;
duplicate vectors are evaluated once. This produces nine unique initial vectors. A final
six-vector interaction pass combines selected evidence and transition profiles, preserving the
original thresholds and verifying source data/code/scenario hashes before reusing screening
results. No Game 2-3 or challenge outcomes enter either search.

The strongest interaction is `proximity_focused+strict_responsive`: proximity scale 2.5 m,
two-/four-metre density weights 0.06/0.02, live/strong costs 20/5, strong support four-of-ten frames,
and strong-control distance/margin 1.0/1.5 m at the 50 fps configuration timeline. Its development
objective rises from 0.43908 to 0.46226. Clean no-tag reference spell F1 at IoU 0.5 rises from 0.4540
to 0.4916, and clean paired-recovery correctness rises from 79.17% to 84.90%. It passes 59/60 guards
but fails recovery-masked/degraded-tag IoU 0.5 F1: 0.4071 versus baseline 0.4294 and the 0.4194
minimum. It is retained as an explicit near-miss, not promoted by quietly waiving that guard.

That regression is mainly extra predicted segments: baseline has 70 matched and 97 spurious
reference-region spells, while the near-miss has 69 matched and 111 spurious spells. Its recovery
instant accuracy nevertheless improves from 56.25% to 61.98%, showing why event correctness alone
cannot select a possession decoder.

The user authorized three reliability-conditioned ablations:

1. Use baseline proximity/density scoring on non-observed ball frames.
2. Use baseline H/A transition costs and strong-support criteria/window on those frames.
3. Apply both fallbacks.

Observed source-1 ball frames retain the improved core settings; interpolation, prediction,
missingness, and model-derived sources receive the declared fallback. The baseline values are
copied exactly rather than tuned anew. New reliability flags default off, and tests verify both
all-reliable/all-unreliable equivalence and frame-rate normalization of fallback costs/windows.
The original baseline and near-miss label hashes reproduce across all eight development
scenarios before the new variants are scored.

| Game 1 recovery-masked/degraded-tag condition | IoU 0.5 spell F1 | Matched | Spurious | Failed guards |
| --- | ---: | ---: | ---: | ---: |
| Current baseline | 0.4294 | 70 | 97 | 0 |
| Original near-miss | 0.4071 | 69 | 111 | 1 |
| Geometry fallback | 0.4024 | 68 | 111 | 4 |
| Transition fallback | 0.3953 | 67 | 113 | 7 |
| Both fallbacks | 0.4024 | 68 | 111 | 7 |

The combined fallback improves clean no-tag spell F1 to 0.5028 and has the highest mean objective,
0.46692, but does not repair the target regression. More conservative per-frame policies are
therefore not a sufficient solution to the segment-level fragmentation. None of the three
fallbacks is promoted and no acceptance threshold is relaxed. A temporal-continuity/hysteresis
or segment-level consistency hypothesis would be a separate experiment, not a confirmed fix.

The three-ablation report is `artifacts/development_reliability_possession.json` and its saved
`configs/archive/metrica_reliability_candidate.json` selects the unchanged baseline. Run it with:

```bash
python -m possession.external_reliability_calibration \
  --root /path/to/metrica-sample-data \
  --output-config configs/archive/metrica_reliability_candidate.json \
  --output-report artifacts/development_reliability_possession.json
```

The prior core reports/configurations are retained as immutable development evidence. The default
challenge submission is unchanged; no inspected external game or challenge input is used to
select the rescue. All 121 unit tests pass. The experiment verifies the implementation and records
a negative performance result, not an improvement or full compliance certificate.

### 13.16 Localization before another temporal repair

The next step replays the current baseline and original core near-miss on Game 1, without changing
the scoring definition or inference parameters. Reference/prediction spell construction is shared
with the scorer, and localization matching is checked against the existing one-to-one IoU metric.
All three replayed conditions reproduce their original label hashes. No Game 2-3 or challenge
data is inspected for this analysis.

The net fourteen additional spurious spells in the recovery-masked/degraded-tag condition are
not fourteen identifiable new intervals. One-to-one same-team IoU lineage finds 83 shared
spurious predictions, 28 introduced/degraded predictions, and 14 baseline errors removed/fixed,
yielding a net increase of 14.

| Introduced/degraded predictions in the failing condition | Count |
| --- | ---: |
| Total | 28 |
| Start on observed ball frames | 24 |
| Start in interpolation | 2 |
| Start unobserved/predicted | 2 |
| Start within 0.5 s after reappearance | 1 |
| Start within 1 s after reappearance | 4 |
| Start within 3 s of a tag boundary | 1 |
| Wrong known reference team at start | 14 |
| Correct known reference team at start | 9 |
| Unknown/censored reference at start | 5 |

Cue neighborhoods overlap and are associations, not causal proof. None reaches the predeclared
50% concentration criterion for a reappearance, interpolation, or tag-boundary repair. Seventeen
of the 28 intervals last at least three seconds, so the error is not predominantly subsecond
flicker either.

The controls also matter: clean degraded-tag input introduces/degrades 30 predictions but fixes
27 baseline errors (net +3), whereas masked/degraded input introduces/degrades 28 and fixes 14
(net +14). These counts suggest loss of the candidate's clean-data benefit under missing
recoveries, not a neat new artifact localized at the gap exit. Cross-condition counts alone do
not establish interval identity or causality.

No reappearance buffer or tag-boundary veto is added or tested: the localization does not justify
either as the requested targeted fix. All guards and the current submission remain unchanged.
The next modeling hypothesis should address control-versus-proximity and sequence consistency
on observed tracking, and would require its own external development experiment rather than a
forced local patch.

Run and inspect the localization evidence with:

```bash
python -m possession.external_spell_localization \
  --root /path/to/metrica-sample-data \
  --output artifacts/localization_core_spurious_spells.json
```

The report retains baseline/candidate scores, input and code hashes, period-local mask/reappearance/
tag boundaries, baseline error controls, and annotated spurious intervals. All 124 unit tests pass.
The original core near-miss remains unpromoted; this negative diagnostic is not another model gain.

### 13.17 Control-versus-proximity diagnosis and one fixed cost restore

The next Game 1 diagnosis compares the original near-miss's introduced/degraded spell starts with
its provider-IoU-matched starts. Both groups use the same prediction extents and matching definition.
The primary comparison retains observed decoded H/A handovers, separates wrong-live-team,
correct-at-start, dead-reference, and unknown-reference cases, and reports each half separately.
No Games 2-3 or challenge inputs are evaluated.

Motion measurement needs stable identities: the existing nearest-player velocity proxy differences
nearest-player positions, which can refer to different people in adjacent frames. The diagnostic
instead differences the current nearest player's same track, with the existing player-speed cap.
Ball velocity uses backward differences only across consecutive observed points within a period.
Unknown future ball frames are excluded from contact denominators, not counted as failed contact.
Production tracking features and inference are not modified.

In recovery-masked/degraded-tag input, there are 48 observed matched H/A starts, 21 observed
introduced/degraded H/A starts, and 14 wrong-known-live-team starts in the latter group:

| Cue | Matched starts | Wrong-live-team starts |
| --- | ---: | ---: |
| Median ball/player relative speed | 15.82 m/s (47 available) | 12.00 m/s (14 available) |
| Median ball speed | 15.00 m/s (47 available) | 10.77 m/s (14 available) |
| Current geometric strong control | 0/48 | 1/14 |
| Upcoming strong-control support absent | 19/48 | 10/14 |
| Low same-player future contact fraction | 18/48 | 9/14 |

Fast-motion and current-contact vetoes are not supported: matched starts are also fast and frequently
precede geometric contact. Lower future contact is somewhat enriched among wrong-team starts but
is also common among matched starts. These are within-Game-1 associations, not evidence that a
particular interval is a pass in flight.

The weaker upcoming-support pattern motivates exactly one fixed ablation, not a new feature or
threshold search: restore the near-miss's ordinary H/A transition cost from 20 to baseline 30, while
keeping its strong-supported transition cost at 5. Its proximity/density settings, strong geometry,
future-support window/minimum, dead/restart formulas, and prediction settings stay unchanged.
Costs and frame windows continue to scale from source 50 fps to Metrica 25 fps.

Baseline and original near-miss outputs reproduce across all eight development scenarios. The
existing objective, 1% spell guard, recovery guards, and live-missingness guards are unchanged:

| Setup | Clean/no-tag F1@0.5 | Clean/degraded-tag F1@0.5 | Recovery-masked/degraded-tag F1@0.5 | Objective | Guards passed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.4540 | 0.4726 | 0.4294 | 0.43908 | 60/60 |
| Original core near-miss | 0.4916 | 0.5028 | 0.4071 | 0.46226 | 59/60 |
| Fixed ordinary-cost restore | 0.4854 | 0.5146 | 0.3976 | 0.45703 | 50/60 |

In the failing condition, restoring the cost reduces spurious spells from 111 to 103 but also reduces
matched spells from 69 to 65. Recovery-instant accuracy falls from 61.98% to 53.65%, below baseline
56.25%. Eight failed guards concern recovery-masked conditions; two concern recovery accuracy in
the five-second live-missingness scenarios. Thus the ablation trades away necessary recoveries,
not just false switches.

The restore is rejected. No development config is emitted for this unsafe candidate, no release
check or challenge inference is run, no guard is relaxed, and the selected config/submission remain
unchanged. This closes the single fixed ablation without another rescue sweep. These scores are
provider-reference development scores, not challenge spell-F1 or independent transfer evidence.

Reproduce the diagnosis and fixed ablation with:

```bash
python -m possession.external_spell_localization \
  --root /path/to/metrica-sample-data --control-cues \
  --output artifacts/control_cues_core_spurious_spells.json
python -m possession.external_spell_localization \
  --root /path/to/metrica-sample-data --weak-transition-restore \
  --diagnosis artifacts/control_cues_core_spurious_spells.json \
  --output artifacts/development_weak_transition_restore.json \
  --output-config configs/development_weak_transition_restore.json
```

The diagnosis records per-start cues, missing-value denominators, rank contrasts, matched-boundary
offsets, half-specific comparisons, and source/input hashes. The ablation records all eight scenario
scores and all 60 guards, the one changed parameter, provenance hashes, and explicit baseline
retention. All 129 unit tests pass.

## 14. Experiment Order

Run experiments in this order so every change has an interpretable purpose:

1. Official nearest-player baseline.
2. Correctly distinguish missing ball from dead ball.
3. Short-gap ball interpolation.
4. Possession hysteresis and transition confirmation.
5. Explicit dead-ball model.
6. Smart Tagging priors.
7. Relative player-ball motion.
8. Local team-density features.
9. Confidence-aware handling of medium and long gaps.
10. Optional tactical-video ball recovery.
11. Optional semi-Markov decoder.

Do not combine several unvalidated changes into one experiment.

An experiment is retained when it:

- fixes a specific observed failure mode;
- preserves alignment and output invariants;
- reduces implausibly short possession fragments;
- remains stable under reasonable parameter perturbations;
- does not rely on tuning against the public labels.

## 15. Schedule

### October 31 to November 2: Reproduce and inspect

- Download and validate all match assets.
- Install the starter dependencies.
- Run the official baseline and evaluator.
- Confirm exact frame alignment.
- Produce missingness, proximity, tagging, and transition diagnostics.
- Freeze a baseline output checksum.

Exit criteria:

- all assets load;
- the four official baselines regenerate;
- the validator accepts generated files;
- team orientation and frame indexing are verified.

### November 3 to November 5: Build the feature pipeline

- Implement asset validation and ATD parsing.
- Implement Smart Tagging frame alignment.
- Ball trajectory cleanup in 2D pitch coordinates.
- Compute ball confidence, motion, and team-control features.
- Add synthetic tests for geometry and indexing.

Exit criteria:

- all 199,319 frames receive a feature record;
- no competition data is written into the repository;
- short-gap repair passes synthetic physical-consistency tests.

### November 6 to November 8: Build the temporal decoder

- Implement the three-state state machine.
- Add loose-ball hysteresis.
- Add controlled-possession confirmation.
- Add explicit dead-ball and restart logic.
- Generate the first non-baseline submission.

Exit criteria:

- output is deterministic and valid;
- one-frame and sub-second possession fragments are rare;
- missing-ball intervals do not automatically become `D`.

### November 12 to November 13: Optional video recovery

- Profile which missing-ball intervals are most likely to contain transitions.
- Add tactical-video detection only if the tracking-only system is stable.
- Invert homography and confidence-gate recovered detections.
- Drop the video component if it produces unstable trajectories or complicates reproducibility.

Exit criteria:

- video recovery is either validated and optional, or explicitly excluded from the final system.

### November 14 to November 15: Reproducibility and documentation

- Test installation in a clean environment.
- Pin all dependencies.
- Ensure one documented command regenerates the exact submission.
- Confirm data and derived exports are ignored.
- Draft the technical document with approach, rules, assets, limitations, and reproduction steps.
- Record runtime and hardware requirements.

Exit criteria:

- clean end-to-end run;
- exact output checksum reproduced;
- official validator passes;
- no restricted data appears in Git history or repository contents.

### November 16: Freeze

- Select the final configuration using robustness and engineering evidence, not the public score.
- Generate the final CSV.
- Run the official validator and evaluator.
- Archive the config, logs, aggregate diagnostics, and checksum.
- Finish the technical document.
- Stop changing modeling logic unless a blocking defect is found.

### November 17: Submit

- Verify filenames use the registered team name.
- Upload the possession CSV, public repository URL, and technical document.
- Confirm the form response can be edited before the deadline.

## 16. Team Work Split

For a three-person team:

### Data and geometry

- asset loading;
- ATD parsing;
- Smart Tagging alignment;
- ball trajectory cleanup;
- homography and optional video recovery.

### Modeling and evaluation

- feature definitions;
- state machine and optional semi-Markov decoder;
- synthetic tests;
- diagnostics and ablations;
- robustness analysis.

### Reproducibility and submission

- CLI and configuration;
- dependency pinning;
- clean-environment runs;
- submission validation;
- technical document and final packaging.

All members should review the possession rules and at least one end-to-end run.

## 17. Risks and Mitigations

### Public-window overfitting

Risk: selecting parameters from only 17 possessions.

Mitigation: use football-informed constants, synthetic tests, structural diagnostics, threshold
perturbations, and external non-challenge games where legally and practically available.

### Treating missing ball as dead ball

Risk: inflated dead-ball duration and fragmented live possessions.

Mitigation: model observation confidence separately from game state and maintain the last confirmed
live possessor through uncertain intervals.

### False transitions from touches and duels

Risk: many short possessions and poor spellF1.

Mitigation: require sustained control evidence and preserve possession through loose-ball states.

### Track fragmentation

Risk: identity-dependent logic fails when slots switch.

Mitigation: use team-level nearest-neighbor, density, and motion features instead of player history.

### Video recovery scope

Risk: detector setup and inference consume the remaining schedule.

Mitigation: make video recovery optional, tactical-only, and limited to high-value gaps.

### Non-reproducible final output

Risk: the submitted file cannot be regenerated exactly.

Mitigation: deterministic algorithms, frozen config, pinned dependencies, fixed seeds where
needed, checksums, and a clean end-to-end rehearsal before the freeze.

## 18. Minimum Viable Submission

If time becomes constrained, freeze this scope:

- supplied tactical tracking only;
- short-gap trajectory cleanup;
- nearest-team distance and persistence features;
- Smart Tagging set-piece priors;
- deterministic three-state decoder;
- strict CSV validation;
- one-command reproducibility.

Do not sacrifice a valid, coherent, reproducible submission to add 3D reconstruction or
broadcast-video synchronization.

## 19. Definition of Done

The project is ready to submit when:

- the official validator accepts the exact final CSV;
- the file contains 199,319 correctly ordered frame rows;
- the repository regenerates the exact file from the provided assets;
- dependencies are pinned and install successfully;
- no restricted data or reconstructive derived data is committed;
- aggregate diagnostics show plausible possession structure;
- the result is stable under modest parameter perturbations;
- the technical document explains data, method, possession rules, limitations, and reproduction;
- the final artifacts have been generated and checked before the deadline.
