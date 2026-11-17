import argparse
import bisect
import hashlib
import json
import re
import secrets
import sys
from collections import Counter, defaultdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from archive.research.jsonio import write_json_atomic
from possession.video import inspect_video


BALL_ACTIONS = {
    "PASS", "HIGH PASS", "DRIVE", "PLAYER SUCCESSFUL TACKLE",
    "CROSS", "BALL PLAYER BLOCK", "OUT", "FREE KICK", "SHOT",
    "THROW IN", "HEADER", "GOAL",
}
REVIEW_QUOTAS = {
    "pass_or_cross": 6,
    "defensive_touch": 6,
    "restart": 4,
    "drive_control": 2,
    "unannotated_control": 6,
}
REVIEW_FIELDS = (
    "status", "reviewer", "permission_confirmed", "contact_team",
    "contact_frame", "actor_id", "confidence", "notes",
)
MAX_MEDIA_CHUNK = 4 * 1024 * 1024


TOUCH_REVIEW_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SoccerTrack touch review</title>
<style>
  :root { color-scheme: dark; font: 16px system-ui, sans-serif; background: #101923; color: #eef4fa; }
  body { margin: 0; padding: 1.25rem; max-width: 1500px; margin-inline: auto; }
  h1 { font-size: 1.5rem; margin: 0 0 .4rem; }
  p { color: #bbcad8; line-height: 1.45; }
  .bar, .controls, .form-row { display: flex; flex-wrap: wrap; align-items: center; gap: .65rem; margin: .8rem 0; }
  button, select, input, textarea { font: inherit; color: inherit; background: #1d2c3b; border: 1px solid #71869b; border-radius: .35rem; padding: .45rem .6rem; }
  button { cursor: pointer; background: #24496b; }
  button:hover, button:focus-visible { background: #3473a8; }
  button:disabled { opacity: .5; cursor: default; }
  label { display: inline-flex; gap: .35rem; align-items: center; }
  video { display: block; width: 100%; max-height: 72vh; background: #000; border: 1px solid #486077; border-radius: .35rem; }
  fieldset { border: 1px solid #486077; border-radius: .35rem; padding: .8rem; margin-top: 1rem; }
  legend { color: #f2d879; font-weight: 600; }
  textarea { min-width: min(100%, 30rem); min-height: 3rem; }
  #message { min-height: 1.5rem; color: #f2d879; }
  .warning { border-left: 4px solid #e6ba57; padding: .5rem .75rem; background: #2b2926; }
  @media (max-width: 700px) { body { padding: .7rem; } .form-row > label { width: 100%; } video { max-height: 55vh; } }
</style>
</head>
<body>
<h1>Blinded SoccerTrack touch review</h1>
<p class="warning">Authorized video through a private localhost/VS Code connection only. Review what is physically visible, not the event implied by the window center. Keep the sealed event context closed. Do not make the forwarded port public or upload frames to hosted AI.</p>
<p>“Left” and “right” are <strong>team IDs fixed for this half by their attacking direction at kickoff</strong>, not where a player stands now. If the kits are unclear, leave the team unknown and mark the contact frame. If you cannot tell whether there was a touch, choose <strong>Can't tell</strong>. Never guess a team.</p>
<div class="bar">
  <button id="previous">Previous case</button>
  <label>Case <select id="case"></select></label>
  <button id="next">Next case</button>
  <span id="progress"></span>
</div>
<video id="video" src="/media" controls playsinline preload="metadata"></video>
<div class="controls">
  <button id="play-window">Play window</button>
  <button id="back-second">-1 s</button>
  <button id="back-frame">-1 frame</button>
  <button id="forward-frame">+1 frame</button>
  <button id="forward-second">+1 s</button>
  <button id="fullscreen">Fullscreen video</button>
  <button id="mark-frame">Mark this frame</button>
  <span id="frame-display"></span>
</div>
<p id="window-info"></p>
<fieldset>
  <legend>What did you see?</legend>
  <div class="form-row">
    <label>Your name (once per session) <input id="reviewer" autocomplete="name" placeholder="Reviewer"></label>
    <label><input id="permission" type="checkbox"> I have authorized access to this local video</label>
  </div>
  <p>Pause on the first visible touch, then use a choice below. Team identification is optional. Your choice saves the card and moves to the next.</p>
  <div class="controls">
    <button id="touch-now">Touch at this frame</button>
    <button id="no-touch">No visible touch</button>
    <button id="cannot-tell">Can't tell</button>
  </div>
  <div class="form-row">
    <label>Team, if recognizable <select id="team"><option value="ambiguous">Unknown team</option><option value="left">Team attacking left at kickoff</option><option value="right">Team attacking right at kickoff</option><option value="none">No visible contact</option></select></label>
    <label>Touch frame (set automatically) <input id="contact-frame" type="number" step="1" placeholder="Mark this frame"></label>
  </div>
  <details>
    <summary>Optional details or corrections</summary>
    <div class="form-row">
      <label>Status <select id="review-status"><option value="unreviewed">Unreviewed</option><option value="verified">Reviewed</option><option value="unverifiable">Can't tell</option></select></label>
      <label>Actor ID <input id="actor-id" inputmode="numeric" placeholder="Only if identifiable"></label>
      <label>Confidence <input id="confidence" type="number" min="0" max="1" step=".05" placeholder="Optional 0–1"></label>
    </div>
    <label>Notes <textarea id="notes" placeholder="Optional context"></textarea></label>
  </details>
  <div class="controls"><button id="save">Save changes</button><button id="save-next">Save changes and next</button></div>
</fieldset>
<p id="message" role="status" aria-live="polite"></p>
<script>
const token = "__CSRF_TOKEN__";
const els = Object.fromEntries(["case","video","progress","frame-display","window-info",
  "review-status","reviewer","permission","team","contact-frame","actor-id","confidence",
  "notes","message"].map(id => [id, document.getElementById(id)]));
let caseIds = [], task = null, reviewerName = "", permissionConfirmed = false, saving = false;
const saveActions = ["previous","next","case","touch-now","no-touch","cannot-tell","save","save-next"]
  .map(id => document.getElementById(id));
const frame = () => Math.round(els.video.currentTime * (task ? task.fps : 25));
const startTime = () => task.start_frame / task.fps;
const stopTime = () => (task.end_frame + 1) / task.fps;
const message = text => { els.message.textContent = text; };
async function request(path, options) {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}
async function refreshIndex() {
  const data = await request("/api/index");
  caseIds = data.cases.map(item => item.case_id);
  const previous = els.case.value;
  els.case.replaceChildren();
  data.cases.forEach(item => {
    const option = document.createElement("option");
    option.value = item.case_id;
    option.textContent = `${item.case_id} — ${item.status}`;
    els.case.append(option);
  });
  if (caseIds.includes(previous)) els.case.value = previous;
  els.progress.textContent = `${data.reviewed}/${caseIds.length} reviewed`;
}
async function loadCase() {
  if (task && els.reviewer.value.trim()) reviewerName = els.reviewer.value.trim();
  if (task && els.permission.checked) permissionConfirmed = true;
  els.video.pause();
  task = await request(`/api/task?case=${encodeURIComponent(els.case.value)}`);
  const review = task.review;
  els["review-status"].value = review.status;
  els.reviewer.value = review.reviewer || reviewerName;
  els.permission.checked = review.permission_confirmed === true || permissionConfirmed;
  els.team.value = review.contact_team || "ambiguous";
  els["contact-frame"].value = review.contact_frame ?? "";
  els["actor-id"].value = review.actor_id || "";
  els.confidence.value = review.confidence ?? "";
  els.notes.value = review.notes || "";
  els.video.currentTime = startTime();
  els["window-info"].textContent = `Video frames ${task.start_frame}–${task.end_frame} (0-based; ${(startTime()).toFixed(2)}–${stopTime().toFixed(2)} s). ${task.review_instructions}`;
  updateFrame();
  message("");
}
function updateFrame() {
  if (!task) return;
  els["frame-display"].textContent = `Frame ${frame()} / center ${task.center_frame} · ${els.video.currentTime.toFixed(2)} s`;
  if (!els.video.paused && els.video.currentTime >= stopTime()) {
    els.video.pause();
    els.video.currentTime = startTime();
  }
}
function step(delta) {
  if (!task) return;
  els.video.pause();
  const target = Math.max(task.start_frame, Math.min(task.end_frame, frame() + delta));
  els.video.currentTime = target / task.fps;
  updateFrame();
}
function moveCase(delta) {
  const position = caseIds.indexOf(els.case.value);
  els.case.value = caseIds[Math.max(0, Math.min(caseIds.length - 1, position + delta))];
  return loadCase();
}
async function save(next) {
  if (!task || saving) return;
  const status = els["review-status"].value;
  const team = els.team.value || null;
  const value = els["contact-frame"].value;
  const review = status === "unreviewed"
    ? {status, reviewer:null, permission_confirmed:null, contact_team:null,
       contact_frame:null, actor_id:null, confidence:null, notes:null}
    : {status, reviewer:els.reviewer.value.trim(), permission_confirmed:els.permission.checked,
       contact_team:status === "verified" ? team : null,
       contact_frame:status === "verified" && ["left","right","ambiguous"].includes(team) && value !== ""
         ? Number(value) : null,
       actor_id:status === "verified" && ["left","right"].includes(team) && els["actor-id"].value.trim()
         ? els["actor-id"].value.trim() : null,
       confidence:status === "verified" && els.confidence.value !== ""
         ? Number(els.confidence.value) : null,
       notes:els.notes.value.trim() || null};
  if (status !== "unreviewed" && !review.reviewer) {
    message("Enter a reviewer name before saving.");
    return;
  }
  if (status !== "unreviewed" && !review.permission_confirmed) {
    message("Confirm authorized access to the local video before saving.");
    return;
  }
  if (status === "verified") {
    if (!review.contact_team) { message("Select the contact team, Team unknown, or No visible contact."); return; }
    if (review.contact_team !== "none" && review.contact_frame === null) {
      message("Mark the contact frame for a visible touch, even if the team is unknown.");
      return;
    }
    if (review.confidence !== null && (!Number.isFinite(review.confidence) ||
        review.confidence < 0 || review.confidence > 1)) {
      message("If set, confidence must be between 0 and 1.");
      return;
    }
  }
  saving = true;
  saveActions.forEach(element => { element.disabled = true; });
  let saved = false;
  try {
    await request(`/api/task?case=${encodeURIComponent(task.case_id)}`, {
      method:"POST", headers:{"Content-Type":"application/json","X-Review-Token":token},
      body:JSON.stringify({review}),
    });
    saved = true;
    reviewerName = els.reviewer.value.trim();
    permissionConfirmed = els.permission.checked;
    message(`Saved ${task.case_id}.`);
    await refreshIndex();
    if (next) await moveCase(1);
  } catch (error) {
    message(`${saved ? "Saved, but couldn't load the next case" : "Not saved"}: ${error.message}`);
  } finally {
    saving = false;
    saveActions.forEach(element => { element.disabled = false; });
  }
}
function chooseReview(choice) {
  if (!task || saving) return;
  els.video.pause();
  if (choice === "touch") {
    els["review-status"].value = "verified";
    if (!["left", "right"].includes(els.team.value)) els.team.value = "ambiguous";
    els["contact-frame"].value = Math.max(task.start_frame, Math.min(task.end_frame, frame()));
  } else if (choice === "none") {
    els["review-status"].value = "verified";
    els.team.value = "none";
    els["contact-frame"].value = "";
    els["actor-id"].value = "";
  } else {
    els["review-status"].value = "unverifiable";
    els.team.value = "ambiguous";
    els["contact-frame"].value = "";
    els["actor-id"].value = "";
  }
  save(true);
}
document.getElementById("previous").onclick = () => moveCase(-1).catch(error => message(error.message));
document.getElementById("next").onclick = () => moveCase(1).catch(error => message(error.message));
document.getElementById("play-window").onclick = () => { els.video.currentTime = startTime(); els.video.play().catch(error => message(error.message)); };
document.getElementById("back-second").onclick = () => step(-25);
document.getElementById("back-frame").onclick = () => step(-1);
document.getElementById("forward-frame").onclick = () => step(1);
document.getElementById("forward-second").onclick = () => step(25);
document.getElementById("fullscreen").onclick = () => els.video.requestFullscreen().catch(error => message(error.message));
document.getElementById("mark-frame").onclick = () => {
  if (!task) return;
  els["contact-frame"].value = Math.max(task.start_frame, Math.min(task.end_frame, frame()));
  els["review-status"].value = "verified";
};
document.getElementById("touch-now").onclick = () => chooseReview("touch");
document.getElementById("no-touch").onclick = () => chooseReview("none");
document.getElementById("cannot-tell").onclick = () => chooseReview("unclear");
document.getElementById("save").onclick = () => save(false);
document.getElementById("save-next").onclick = () => save(true);
els.case.onchange = () => loadCase().catch(error => message(error.message));
els.team.onchange = () => {
  if (els.team.value) els["review-status"].value = "verified";
  if (els.team.value === "none") els["contact-frame"].value = "";
  if (["none", "ambiguous"].includes(els.team.value)) els["actor-id"].value = "";
};
els.video.addEventListener("timeupdate", updateFrame);
els.video.addEventListener("seeked", updateFrame);
document.addEventListener("keydown", event => {
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") {
    event.preventDefault();
    save(false);
    return;
  }
  if (["INPUT","TEXTAREA","SELECT"].includes(document.activeElement.tagName)) return;
  if (event.key === "ArrowLeft") { event.preventDefault(); step(-1); }
  if (event.key === "ArrowRight") { event.preventDefault(); step(1); }
  if (event.key === " ") { event.preventDefault(); els.video.paused ? els.video.play() : els.video.pause(); }
});
refreshIndex().then(loadCase).catch(error => message(error.message));
</script>
</body>
</html>"""


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _half_actions(bas_path, match, half, frames, fps):
    data = json.loads(Path(bas_path).read_text())
    if str(data.get("match_id")) != match or float(data.get("fps", 0)) != fps:
        raise ValueError("SoccerTrack BAS match or frame rate differs from video")
    actions = data.get("actions")
    if not isinstance(actions, list):
        raise ValueError("SoccerTrack BAS requires an actions array")
    selected = []
    for action in actions:
        if not isinstance(action, dict) or action.get("label") not in BALL_ACTIONS:
            raise ValueError("SoccerTrack BAS contains an unknown ball action")
        if action.get("team") not in ("left", "right"):
            raise ValueError("SoccerTrack BAS has an unknown acting team")
        clock = action.get("gameTime")
        if not isinstance(clock, str) or not clock.startswith(("1 - ", "2 - ")):
            raise ValueError("SoccerTrack BAS has an invalid half clock")
        if not clock.startswith("{} - ".format(half)):
            continue
        frame = action.get("frame")
        position = action.get("position")
        if (
            not isinstance(frame, int) or isinstance(frame, bool)
            or not isinstance(position, str) or not position.isdigit()
            or not 1 <= frame <= frames
            or frame - 1 != round(int(position) * fps / 1000)
        ):
            raise ValueError("SoccerTrack BAS frame must align to 1-based GSR and 0-based video")
        actor = action.get("player_id")
        if actor is not None and (
            not isinstance(actor, str) or not actor.isdigit()
        ):
            raise ValueError("SoccerTrack BAS actor must be a numeric string or null")
        selected.append(action)
    if not selected:
        raise ValueError("SoccerTrack BAS has no actions in the requested half")
    return selected


def build_soccertrack_alignment(root, match, half):
    if not (isinstance(match, str) and len(match) == 6 and match.isdigit()):
        raise ValueError("SoccerTrack match must be a six-digit identifier")
    if half not in (1, 2):
        raise ValueError("SoccerTrack half must be first or second")
    try:
        import ijson
    except ImportError as error:
        raise RuntimeError(
            "SoccerTrack GSR needs optional streaming ijson dependency"
        ) from error

    root = Path(root)
    label = "1st" if half == 1 else "2nd"
    bas_path = root / "bas" / match / "{}_12_class_events.json".format(match)
    gsr_path = root / "gsr" / match / "{}_{}.json".format(match, label)
    video_path = root / "videos" / match / "{}_panorama_{}_half.mp4".format(
        match, label
    )
    for path in (bas_path, gsr_path, video_path):
        if not path.is_file():
            raise ValueError("{}: authorized SoccerTrack asset is missing".format(path))
    metadata = inspect_video(str(video_path))
    if metadata.fps != 25 or metadata.frames < 1:
        raise ValueError("SoccerTrack aligned video must be nonempty and 25 fps")
    actions = _half_actions(bas_path, match, half, metadata.frames, metadata.fps)

    with gsr_path.open("rb") as handle:
        info = next(ijson.items(handle, "info"))
    if (
        float(info.get("frame_rate", 0)) != metadata.fps
        or int(info.get("seq_length", 0)) != metadata.frames
    ):
        raise ValueError("SoccerTrack GSR header differs from the aligned video")

    image_count = 0
    prefix = None
    with gsr_path.open("rb") as handle:
        for image in ijson.items(handle, "images.item"):
            image_id = image.get("image_id")
            if not isinstance(image_id, str) or not image_id.isdigit():
                raise ValueError("SoccerTrack GSR image ID must be numeric text")
            current_prefix, frame = divmod(int(image_id), 1000000)
            if (
                frame != image_count + 1
                or (prefix is not None and prefix != current_prefix)
                or image.get("width") != metadata.width
                or image.get("height") != metadata.height
            ):
                raise ValueError("SoccerTrack GSR image IDs or dimensions differ from video")
            prefix = current_prefix
            image_count += 1
    if image_count != metadata.frames:
        raise ValueError("SoccerTrack GSR and video frame counts differ")

    targets = defaultdict(list)
    for index, action in enumerate(actions):
        for offset in range(-3, 4):
            frame = action["frame"] + offset
            if 1 <= frame <= metadata.frames:
                targets[frame].append((index, offset))

    exact, nearby, team_match = set(), set(), set()
    roles = Counter()
    annotations = 0
    with gsr_path.open("rb") as handle:
        for item in ijson.items(handle, "annotations.item"):
            annotations += 1
            if item.get("supercategory") != "object":
                continue
            attributes = item.get("attributes") or {}
            role = attributes.get("role")
            if role not in ("player", "goalkeeper", "referee", "other", "ball"):
                raise ValueError("SoccerTrack GSR has an unknown object role")
            roles[role] += 1
            image_id = item.get("image_id")
            if (
                not isinstance(image_id, str) or not image_id.isdigit()
                or int(image_id) // 1000000 != prefix
            ):
                raise ValueError("SoccerTrack GSR annotation references an invalid image")
            frame = int(image_id) % 1000000
            if not 1 <= frame <= image_count:
                raise ValueError("SoccerTrack GSR annotation frame is outside video")
            actor = attributes.get("player_id")
            if actor is None:
                continue
            for index, offset in targets.get(frame, ()):
                action = actions[index]
                if action.get("player_id") is None or str(actor) != action["player_id"]:
                    continue
                nearby.add(index)
                if offset == 0:
                    exact.add(index)
                    if attributes.get("team") == action["team"]:
                        team_match.add(index)

    return {
        "schema_version": 1,
        "dataset": "SoccerTrack v2",
        "match": match,
        "half": half,
        "license": "CC BY 4.0; credit SoccerTrack v2 and Scott et al.",
        "scope": (
            "Shipped BAS-to-GSR actor/frame alignment only; actions are "
            "event proxies, not independently verified physical touches."
        ),
        "video": {
            "frames": metadata.frames,
            "fps": metadata.fps,
            "width": metadata.width,
            "height": metadata.height,
            "sha256": _sha256(video_path),
        },
        "bas_sha256": _sha256(bas_path),
        "gsr_sha256": _sha256(gsr_path),
        "video_frame_index_from_action": "frame - 1 = round(position_ms / 40)",
        "gsr_image_id_suffix": "frame (1-based)",
        "first_half_action_count" if half == 1 else "second_half_action_count": len(actions),
        "actor_ids_present": sum(
            action.get("player_id") is not None for action in actions
        ),
        "actor_present_at_exact_frame": len(exact),
        "actor_present_within_three_frames": len(nearby),
        "actor_team_agrees_at_exact_frame": len(team_match),
        "action_classes": dict(sorted(Counter(
            action["label"] for action in actions
        ).items())),
        "gsr_image_count": image_count,
        "gsr_annotation_count": annotations,
        "gsr_object_roles": dict(sorted(roles.items())),
        "independent_ball_position_truth_available": False,
        "physical_contact_verified": False,
        "challenge_data_used": False,
    }


def select_blinded_touch_windows(actions, frames):
    groups = {
        "pass_or_cross": {"PASS", "HIGH PASS", "CROSS"},
        "defensive_touch": {"BALL PLAYER BLOCK", "PLAYER SUCCESSFUL TACKLE"},
        "restart": {"THROW IN", "FREE KICK"},
        "drive_control": {"DRIVE"},
    }
    if frames < 101:
        raise ValueError("touch review needs at least 101 aligned video frames")
    selected = []
    for group, labels in groups.items():
        candidates = sorted(
            (index, action["frame"] - 1)
            for index, action in enumerate(actions)
            if action["label"] in labels and action.get("player_id") is not None
            and 25 <= action["frame"] - 1 < frames - 25
        )
        candidates.sort(key=lambda item: item[1])
        quota = REVIEW_QUOTAS[group]
        if len(candidates) < quota:
            raise ValueError("{} lacks {} eligible actions".format(group, quota))
        for offset in range(quota):
            target = round(offset * (len(candidates) - 1) / max(1, quota - 1))
            available = [
                (position, index, frame)
                for position, (index, frame) in enumerate(candidates)
                if all(abs(frame - other["center_frame"]) >= 100
                       for other in selected)
            ]
            if not available:
                raise ValueError("{} cannot select spaced review windows".format(group))
            _, index, frame = min(
                available,
                key=lambda item: (
                    abs(item[0] - target), item[2]
                ),
            )
            selected.append({
                "group": group, "action_index": index,
                "center_frame": frame,
            })

    action_frames = sorted(action["frame"] - 1 for action in actions)
    controls = []
    for frame in range(25, frames - 25, 100):
        index = bisect.bisect_left(action_frames, frame)
        neighbors = action_frames[max(0, index - 1):index + 1]
        if (
            all(abs(frame - nearby) > 50 for nearby in neighbors)
            and all(abs(frame - item["center_frame"]) >= 100
                    for item in selected)
        ):
            controls.append(frame)
    if len(controls) < REVIEW_QUOTAS["unannotated_control"]:
        raise ValueError("not enough source-unannotated review controls")
    count = REVIEW_QUOTAS["unannotated_control"]
    for offset in range(count):
        index = round(offset * (len(controls) - 1) / max(1, count - 1))
        selected.append({
            "group": "unannotated_control", "action_index": None,
            "center_frame": controls[index],
        })
    return sorted(selected, key=lambda item: item["center_frame"])


def validate_touch_review_task(task, require_reviewed=False):
    if (
        not isinstance(task, dict)
        or set(task) != {
            "schema_version", "dataset", "case_id", "match", "half",
            "video_path", "start_frame", "center_frame", "end_frame",
            "fps", "review_instructions", "review",
        }
        or task["schema_version"] != 1
        or task["dataset"] != "SoccerTrack v2"
        or not isinstance(task["case_id"], str)
        or re.fullmatch(r"case-\d{2}", task["case_id"]) is None
        or not isinstance(task["video_path"], str)
        or task["fps"] != 25
        or not all(isinstance(task[key], int) for key in
                   ("start_frame", "center_frame", "end_frame"))
        or not task["start_frame"] <= task["center_frame"] <= task["end_frame"]
    ):
        raise ValueError("invalid or unblinded SoccerTrack touch review card")
    review = task["review"]
    if (
        not isinstance(review, dict)
        or set(review) != set(REVIEW_FIELDS)
        or review["status"] not in ("unreviewed", "unverifiable", "verified")
    ):
        raise ValueError("invalid SoccerTrack touch review fields")
    if review["status"] == "unreviewed":
        if any(value is not None for key, value in review.items()
               if key != "status"):
            raise ValueError("unreviewed touch card must not prefill labels")
    else:
        if (
            not isinstance(review["reviewer"], str)
            or not review["reviewer"].strip()
            or review["permission_confirmed"] is not True
        ):
            raise ValueError("touch review requires a permitted human reviewer")
        if review["status"] == "unverifiable":
            if (
                review["contact_team"] is not None
                or review["contact_frame"] is not None
                or review["actor_id"] is not None
                or review["confidence"] is not None
                or (review["notes"] is not None
                    and not isinstance(review["notes"], str))
            ):
                raise ValueError("unverifiable touch card cannot assert contact")
        else:
            team = review["contact_team"]
            frame = review["contact_frame"]
            actor = review["actor_id"]
            confidence = review["confidence"]
            if team not in ("left", "right", "none", "ambiguous"):
                raise ValueError("select contact team, Team unknown or No visible contact")
            if (team != "none") != (frame is not None):
                raise ValueError("visible touch needs a marked contact frame; No visible contact cannot have one")
            if frame is not None and (
                not isinstance(frame, int) or isinstance(frame, bool)
                or not task["start_frame"] <= frame <= task["end_frame"]
            ):
                raise ValueError("contact frame must be a whole number inside this video window")
            if actor is not None and (
                team not in ("left", "right")
                or not isinstance(actor, str) or not actor.isdigit()
            ):
                raise ValueError("actor ID needs a definite team and numeric identifier")
            if confidence is not None and (
                not isinstance(confidence, (int, float))
                or isinstance(confidence, bool)
                or not 0 <= confidence <= 1
            ):
                raise ValueError("optional confidence must be between 0 and 1")
            if review["notes"] is not None and not isinstance(review["notes"], str):
                raise ValueError("touch review notes must be text")
    if require_reviewed and review["status"] == "unreviewed":
        raise ValueError("touch case still needs independent human video review")
    return review["status"]


def build_touch_review_workspace(root, match, half, alignment_path, workspace):
    workspace = Path(workspace).resolve()
    if workspace.exists():
        raise FileExistsError("{}: touch review workspace already exists".format(workspace))
    alignment_path = Path(alignment_path)
    alignment = json.loads(alignment_path.read_text())
    video_path = (
        Path(root) / "videos" / match /
        "{}_panorama_{}_half.mp4".format(
            match, "1st" if half == 1 else "2nd"
        )
    )
    bas_path = Path(root) / "bas" / match / "{}_12_class_events.json".format(match)
    if (
        alignment.get("dataset") != "SoccerTrack v2"
        or alignment.get("match") != match or alignment.get("half") != half
        or alignment.get("physical_contact_verified") is not False
        or not video_path.is_file()
        or _sha256(bas_path) != alignment.get("bas_sha256")
    ):
        raise ValueError("touch review requires the pinned external event alignment")
    frames = alignment["video"]["frames"]
    actions = _half_actions(bas_path, match, half, frames, 25)
    selected = select_blinded_touch_windows(actions, frames)
    cards = []
    private = []
    for index, item in enumerate(selected, 1):
        frame = item["center_frame"]
        case_id = "case-{:02d}".format(index)
        task = {
            "schema_version": 1, "dataset": "SoccerTrack v2",
            "case_id": case_id, "match": match, "half": half,
            "video_path": str(video_path.resolve()), "fps": 25,
            "start_frame": frame - 25, "center_frame": frame,
            "end_frame": frame + 25,
            "review_instructions": (
                "Review only the local authorized video window. Record the first "
                "visible physical ball contact by any player, or mark none/ambiguous. "
                "Do not assume the centre frame is a touch; do not open the sealed "
                "event context or model predictions before review."
            ),
            "review": {
                key: "unreviewed" if key == "status" else None
                for key in REVIEW_FIELDS
            },
        }
        validate_touch_review_task(task)
        cards.append(task)
        event = (
            actions[item["action_index"]]
            if item["action_index"] is not None else None
        )
        private.append({
            "case_id": case_id, "group": item["group"],
            "action_index": item["action_index"],
            "event": event,
        })
    index = {
        "schema_version": 1, "dataset": "SoccerTrack v2",
        "match": match, "half": half,
        "alignment_report_sha256": _sha256(alignment_path),
        "source_video_sha256": alignment["video"]["sha256"],
        "generator_sha256": _sha256(Path(__file__)),
        "selection_policy": {
            "quotas": REVIEW_QUOTAS,
            "minimum_spacing_frames": 100,
            "window_radius_frames": 25,
            "control_exclusion_radius_frames": 50,
            "model_predictions_used": False,
            "physical_contacts_prefilled": False,
        },
        "tasks": [card["case_id"] for card in cards],
    }
    summary = {
        "schema_version": 1,
        "total_cases": len(cards),
        "selection_groups": dict(Counter(item["group"] for item in private)),
        "independently_verified_contacts": 0,
        "physical_touch_review_complete": False,
        "event_interpolated_ball_used": False,
        "challenge_data_used": False,
        "alignment_report_sha256": index["alignment_report_sha256"],
    }
    workspace.mkdir(parents=True)
    for task in cards:
        write_json_atomic(workspace / "tasks" / (task["case_id"] + ".json"), task)
    write_json_atomic(workspace / "index.json", index)
    write_json_atomic(workspace / "selection_private.json", private)
    write_json_atomic(workspace / "summary.json", summary)
    return summary


def validate_touch_review_workspace(workspace, require_reviewed=False):
    workspace = Path(workspace)
    index = json.loads((workspace / "index.json").read_text())
    if (
        index.get("schema_version") != 1
        or index.get("dataset") != "SoccerTrack v2"
        or index.get("selection_policy", {}).get("model_predictions_used") is not False
        or index.get("selection_policy", {}).get("physical_contacts_prefilled") is not False
        or len(index.get("tasks", [])) != sum(REVIEW_QUOTAS.values())
        or len(set(index["tasks"])) != len(index["tasks"])
    ):
        raise ValueError("SoccerTrack review index lacks source-only provenance")
    statuses = Counter()
    for case_id in index["tasks"]:
        if not isinstance(case_id, str) or re.fullmatch(r"case-\d{2}", case_id) is None:
            raise ValueError("invalid touch review case identifier")
        task = json.loads((workspace / "tasks" / (case_id + ".json")).read_text())
        if (
            task.get("case_id") != case_id
            or task.get("match") != index["match"]
            or task.get("half") != index["half"]
        ):
            raise ValueError("touch review task differs from its index")
        statuses[validate_touch_review_task(task, require_reviewed)] += 1
    return {"tasks": len(index["tasks"]), "review_status": dict(statuses)}


def make_touch_review_handler(workspace, video_path, token):
    workspace = Path(workspace).resolve()
    video_path = Path(video_path).resolve()
    index = json.loads((workspace / "index.json").read_text())
    validate_touch_review_workspace(workspace)
    if (
        not index["tasks"]
        or video_path.name != "{}_panorama_{}_half.mp4".format(
            index["match"], "1st" if index["half"] == 1 else "2nd"
        )
        or video_path.parent.name != index["match"]
        or not video_path.is_file()
        or _sha256(video_path) != index["source_video_sha256"]
    ):
        raise ValueError("review server needs the pinned authorized local video")
    task_paths = {
        case_id: workspace / "tasks" / (case_id + ".json")
        for case_id in index["tasks"]
    }
    for path in task_paths.values():
        if Path(json.loads(path.read_text())["video_path"]).resolve() != video_path:
            raise ValueError("review card refers to a different local video")

    class Handler(BaseHTTPRequestHandler):
        def _json(self, data, status=HTTPStatus.OK):
            body = json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _case_path(self, query):
            ids = parse_qs(query).get("case", [])
            if len(ids) != 1 or ids[0] not in task_paths:
                raise ValueError("unknown blinded review case")
            return task_paths[ids[0]]

        def _media(self, head_only=False):
            size = video_path.stat().st_size
            range_header = self.headers.get("Range")
            first, last = 0, size - 1
            status = HTTPStatus.OK
            if range_header is not None:
                match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
                if match is None or not any(match.groups()):
                    self._invalid_range(size)
                    return
                start, end = match.groups()
                if start:
                    first = int(start)
                    last = min(int(end), size - 1) if end else size - 1
                else:
                    suffix = int(end)
                    first = max(0, size - suffix)
                if first >= size or last < first or (not start and not suffix):
                    self._invalid_range(size)
                    return
                last = min(last, first + MAX_MEDIA_CHUNK - 1)
                status = HTTPStatus.PARTIAL_CONTENT
            self.send_response(status)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(last - first + 1))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "private, no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if status == HTTPStatus.PARTIAL_CONTENT:
                self.send_header("Content-Range", "bytes {}-{}/{}".format(first, last, size))
            self.end_headers()
            if head_only:
                return
            try:
                with video_path.open("rb") as media:
                    media.seek(first)
                    remaining = last - first + 1
                    while remaining:
                        chunk = media.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise OSError("authorized video ended while streaming")
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                return

        def _invalid_range(self, size):
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", "bytes */{}".format(size))
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_HEAD(self):
            if urlparse(self.path).path == "/media":
                self._media(head_only=True)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_GET(self):
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/":
                    body = TOUCH_REVIEW_HTML.replace(
                        "__CSRF_TOKEN__", token
                    ).encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header(
                        "Content-Security-Policy",
                        "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                        "style-src 'self' 'unsafe-inline'; media-src 'self'; "
                        "connect-src 'self'; object-src 'none'; base-uri 'none'",
                    )
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("X-Content-Type-Options", "nosniff")
                    self.end_headers()
                    self.wfile.write(body)
                elif parsed.path == "/api/index":
                    cards = [
                        json.loads(path.read_text())
                        for path in task_paths.values()
                    ]
                    self._json({
                        "cases": [{
                            "case_id": task["case_id"],
                            "status": task["review"]["status"],
                        } for task in cards],
                        "reviewed": sum(
                            task["review"]["status"] != "unreviewed"
                            for task in cards
                        ),
                    })
                elif parsed.path == "/api/task":
                    task = json.loads(self._case_path(parsed.query).read_text())
                    self._json({
                        key: task[key] for key in (
                            "case_id", "start_frame", "center_frame",
                            "end_frame", "fps", "review_instructions", "review",
                        )
                    })
                elif parsed.path == "/media":
                    self._media()
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
            except (KeyError, TypeError, ValueError) as error:
                self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            except OSError as error:
                self.log_error("review read failed: %s", error)
                self._json({"error": str(error)}, HTTPStatus.INTERNAL_SERVER_ERROR)

        def do_POST(self):
            self.close_connection = True
            parsed = urlparse(self.path)
            if parsed.path != "/api/task":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            if self.headers.get("X-Review-Token") != token:
                self._json({"error": "invalid local review token"}, HTTPStatus.FORBIDDEN)
                return
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                self._json({"error": "review requires JSON"}, HTTPStatus.UNSUPPORTED_MEDIA_TYPE)
                return
            try:
                path = self._case_path(parsed.query)
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 8192:
                    raise ValueError("invalid review request size")
                submitted = json.loads(self.rfile.read(length))
                if not isinstance(submitted, dict) or set(submitted) != {"review"}:
                    raise ValueError("only blinded review fields can be saved")
                task = json.loads(path.read_text())
                task["review"] = submitted["review"]
                validate_touch_review_task(task)
                write_json_atomic(path, task)
                self._json({
                    "case_id": task["case_id"],
                    "status": task["review"]["status"],
                })
            except (KeyError, TypeError, ValueError) as error:
                self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            except OSError as error:
                self.log_error("review save failed: %s", error)
                self._json({"error": str(error)}, HTTPStatus.INTERNAL_SERVER_ERROR)

        def log_message(self, format_string, *args):
            print(format_string % args)

    return Handler


def serve_touch_review_workspace(workspace, port):
    if not 1 <= port <= 65535:
        raise ValueError("review port must be between 1 and 65535")
    workspace = Path(workspace).resolve()
    validate_touch_review_workspace(workspace)
    index = json.loads((workspace / "index.json").read_text())
    first = json.loads(
        (workspace / "tasks" / (index["tasks"][0] + ".json")).read_text()
    )
    token = secrets.token_urlsafe(32)
    handler = make_touch_review_handler(
        workspace, first["video_path"], token
    )
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    print("Local touch review: http://127.0.0.1:{}/".format(port), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Align locally authorized SoccerTrack BAS actors to GSR frames."
    )
    parser.add_argument("--root")
    parser.add_argument("--match")
    parser.add_argument("--half", type=int, choices=(1, 2))
    parser.add_argument("--output")
    parser.add_argument("--workspace")
    parser.add_argument("--alignment")
    parser.add_argument("--validate-workspace")
    parser.add_argument("--require-reviewed", action="store_true")
    parser.add_argument("--serve-workspace")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    if args.serve_workspace:
        if (
            args.root or args.match or args.half or args.output
            or args.workspace or args.alignment or args.validate_workspace
            or args.require_reviewed
        ):
            parser.error("--serve-workspace cannot be combined with build or validation inputs")
        try:
            serve_touch_review_workspace(args.serve_workspace, args.port)
        except (OSError, ValueError) as error:
            parser.exit(1, "error: {}\n".format(error))
        return 0
    if args.validate_workspace:
        if (
            args.root or args.match or args.half or args.output
            or args.workspace or args.alignment
        ):
            parser.error("--validate-workspace cannot be combined with build inputs")
        try:
            result = validate_touch_review_workspace(
                args.validate_workspace, args.require_reviewed
            )
        except (OSError, ValueError) as error:
            parser.exit(1, "error: {}\n".format(error))
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if not (args.root and args.match and args.half):
        parser.error("--root, --match and --half are required")
    if args.require_reviewed or bool(args.workspace) == bool(args.output):
        parser.error("choose either --output alignment or --workspace review packet")
    if bool(args.workspace) != bool(args.alignment):
        parser.error("--workspace and --alignment must be supplied together")
    try:
        if args.workspace:
            report = build_touch_review_workspace(
                args.root, args.match, args.half,
                args.alignment, args.workspace,
            )
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        report = build_soccertrack_alignment(args.root, args.match, args.half)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    except (OSError, RuntimeError, ValueError, StopIteration) as error:
        parser.exit(1, "error: {}\n".format(error))
    print(json.dumps({
        "actions": report["first_half_action_count"]
        if args.half == 1 else report["second_half_action_count"],
        "actor_exact": report["actor_present_at_exact_frame"],
        "actor_team_agrees": report["actor_team_agrees_at_exact_frame"],
        "physical_contact_verified": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
