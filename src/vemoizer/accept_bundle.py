"""On-disk format of the private acceptance bench (issue #227, epic #220).

A bench lives in a project layer's ``.vemoizer/bench/`` — REAL audio and
references, never in git (AGENTS.md "Never Commit") — as ``manifest.json``
+ ``audio/<id>/<slot>.flac`` (16 kHz mono, invariant #6) +
``reference/<id>.json``. Only reader/writer; the accept CLI lands in #229.
"""

import contextlib
import hashlib
import json
import math
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .audio_contract import SAMPLE_RATE

MANIFEST_VERSION = 1  # the manifest schema version we read and write
UNCLEAR = "unclear"  # speaker placeholder until the owner names the segment

# The two legal audio layouts: a single mix, or the two-track call stems.
_KIND_SLOTS: dict[str, tuple[str, ...]] = {"mix": ("mix",), "two-track": ("me", "them")}

# Ids land in bench-relative paths; a looser pattern lets "../../x" escape.
_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")


class BundleError(RuntimeError):
    """The bundle's audio is missing or failed sha256 verification."""


@dataclass(frozen=True)
class Excerpt:  # an audio window cut out of a real recording
    id: str
    kind: str  # "mix" | "two-track"
    audio: dict[str, str]  # slot -> bench-relative flac path
    sha256: dict[str, str]  # relative path -> hex digest
    window: tuple[float, float]  # [start, end) in seconds of the ORIGINAL file
    source: str  # original sidecar path (provenance)
    glossary_sha256: str | None
    owner: str | None


@dataclass(frozen=True)
class RefSegment:  # times are relative to the excerpt start
    start: float
    end: float
    speaker: str  # a name, or UNCLEAR
    text: str
    edge: bool = False  # straddles the window edge; text needs manual trimming


@dataclass(frozen=True)
class Reference:  # the reviewed ground truth for one excerpt
    id: str
    segments: tuple[RefSegment, ...]
    title_expect: dict[str, list[str]]  # keys must_contain / forbid
    reviewed_by: str | None
    reviewed_on: str | None  # "YYYY-MM-DD"


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON {path}: {exc}") from exc


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_write_json(path: Path, obj: Any) -> None:
    # tmp + os.replace: a half-written manifest would read as corruption.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    text = json.dumps(obj, indent=2, ensure_ascii=False) + "\n"
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _require_str(obj: dict[str, Any], key: str, w: str, or_none: bool = False) -> Any:
    value = obj.get(key)
    if value is None and or_none:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{w}: {key} must be a string{' or null' if or_none else ''}")
    return value


def _require_id(exc_id: str, what: str) -> str:
    if not _ID_RE.fullmatch(exc_id):
        raise ValueError(f"{what}: excerpt id {exc_id!r} must match {_ID_RE.pattern}")
    return exc_id


def _num(v: Any) -> bool:
    # bool IS an int subclass, and NaN/inf poison every span computation.
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _require_span(what: str, name: str, start: Any, end: Any) -> tuple[float, float]:
    if not (_num(start) and _num(end)) or start < 0 or end <= start:
        raise ValueError(f"{what}: {name} must be finite numbers with 0 <= start < end")
    return float(start), float(end)


def _cut_audio(src: Path, dst: Path, start: float, end: float) -> None:
    """Cut [start, end) of *src* into *dst* as 16 kHz mono FLAC — the single
    monkeypatchable ffmpeg seam.

    16-bit FLAC on purpose: the live-capture stems we cut from are FLAC too,
    and every reader gets 16 kHz mono float32 after ffmpeg ingest (invariant
    #6). The on-disk sample format is storage, not the internal contract.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    # -f forces the muxer: the atomic temp target ends in ".tmp", no inference.
    # -ac/-ar after -i: before it they are input options and do not downmix.
    argv = ["ffmpeg", "-nostdin", "-v", "error", "-y"]
    argv += ["-ss", f"{start:.6f}", "-i", str(src), "-t", f"{end - start:.6f}"]
    argv += ["-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "flac", "-f", "flac"]
    argv += [str(dst)]
    try:
        proc = subprocess.run(argv, capture_output=True, check=False)
    except FileNotFoundError as exc:  # ffmpeg is the one system dependency
        raise BundleError("ffmpeg not found on PATH; install ffmpeg") from exc
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", errors="replace")[-400:]
        raise BundleError(f"ffmpeg cut failed for {src} [{start}, {end}): {tail}")


def _manifest_entries(bench: Path, *, required: bool) -> list[Excerpt]:
    """Manifest excerpts; a missing manifest is fatal when required.

    Entries are validated here so load and freeze share one gate: corrupt
    entries fail loudly instead of being silently dropped on re-freeze.
    """
    path = bench / "manifest.json"
    if not path.is_file():
        if required:
            raise BundleError(f"bench manifest not found: {path}")
        return []
    m = _load_json(path)
    if not isinstance(m, dict) or m.get("version", 1) != MANIFEST_VERSION:
        raise ValueError(f"unsupported manifest version in {path}")
    entries = m.get("excerpts", [])
    if not isinstance(entries, list):
        raise ValueError(f"manifest.excerpts must be a list in {path}")
    return [_excerpt_from_manifest(e) for e in entries]


def load_bundle(bench_dir: Path) -> tuple[list[Excerpt], dict[str, Reference]]:
    """Load and validate a bench: every manifest audio file is sha256-checked.

    BundleError: missing or tampered manifest/audio. ValueError: schema
    violations, e.g. an excerpt without a reference.
    """
    bench = Path(bench_dir)
    excerpts = _manifest_entries(bench, required=True)
    references: dict[str, Reference] = {}
    for excerpt in excerpts:
        for rel, digest in excerpt.sha256.items():
            path = bench / rel
            if not path.is_file():
                raise BundleError(f"excerpt {excerpt.id}: missing audio file {rel}")
            if _sha256_file(path) != digest:
                raise BundleError(f"excerpt {excerpt.id}: sha256 mismatch for {rel}")
        ref_path = bench / "reference" / f"{excerpt.id}.json"
        if not ref_path.is_file():
            raise ValueError(f"excerpt {excerpt.id!r} has no reference file {ref_path}")
        references[excerpt.id] = _reference_from_json(_load_json(ref_path), excerpt.id)
    return excerpts, references


def _excerpt_from_manifest(entry: Any) -> Excerpt:
    if not isinstance(entry, dict):
        raise ValueError(f"manifest excerpt entry must be an object: {entry!r}")
    base = "manifest excerpt"
    exc_id = _require_id(_require_str(entry, "id", base), base)
    what = f"{base} {exc_id}"
    kind = _require_str(entry, "kind", what)
    if kind not in _KIND_SLOTS:
        raise ValueError(f"{what}: kind must be one of {sorted(_KIND_SLOTS)}")
    audio = entry.get("audio")
    if not isinstance(audio, dict) or tuple(audio) != _KIND_SLOTS[kind]:
        raise ValueError(f"{what}: audio keys must be {list(_KIND_SLOTS[kind])}")
    for rel in audio.values():
        p = Path(str(rel))
        if not isinstance(rel, str) or p.is_absolute() or ".." in p.parts:
            raise ValueError(f"{what}: audio path {rel!r} must be bench-relative")
    sha = entry.get("sha256")
    if not isinstance(sha, dict) or set(sha) != set(audio.values()):
        raise ValueError(f"{what}: sha256 keys must be exactly the audio paths")
    w = entry.get("window")
    w = w if isinstance(w, list) and len(w) == 2 else (None, None)
    w0, w1 = _require_span(what, "window", *w)
    for key in ("glossary_sha256", "owner"):
        _require_str(entry, key, what, or_none=True)
    sha_map = {str(k): str(v) for k, v in sha.items()}
    src = _require_str(entry, "source", what)
    tail = (entry.get("glossary_sha256"), entry.get("owner"))
    return Excerpt(exc_id, kind, dict(audio), sha_map, (w0, w1), src, *tail)


def _reference_from_json(obj: Any, expected_id: str) -> Reference:
    what = f"reference {expected_id!r}"
    if not isinstance(obj, dict) or _require_str(obj, "id", what) != expected_id:
        raise ValueError(f"{what}: must be an object whose id matches the excerpt")
    for key in ("reviewed_by", "reviewed_on"):
        _require_str(obj, key, what, or_none=True)
    raw = obj.get("segments")
    if not isinstance(raw, list):
        raise ValueError(f"{what}: segments must be a list")
    title = obj.get("title_expect")
    if not isinstance(title, dict) or any(
        not isinstance(title.get(k), list)
        or not all(isinstance(v, str) for v in title[k])
        for k in ("must_contain", "forbid")
    ):
        raise ValueError(f"{what}: title_expect needs str-list must_contain/forbid")
    segs = []
    for d in raw:
        if not isinstance(d, dict):
            raise ValueError(f"{what}: segment must be an object: {d!r}")
        s0, e0 = _require_span(what, "segment start/end", d.get("start"), d.get("end"))
        spk = _require_str(d, "speaker", f"{what} segment")
        txt = _require_str(d, "text", f"{what} segment")  # null must not become "None"
        edge = d.get("edge", False)
        if not isinstance(edge, bool):
            raise ValueError(f"{what}: segment edge must be a bool when present")
        segs.append(RefSegment(s0, e0, spk, txt, edge))
    who, when = obj.get("reviewed_by"), obj.get("reviewed_on")
    return Reference(expected_id, tuple(segs), title, who, when)


def unreviewed(references: dict[str, Reference]) -> list[str]:
    """Ids whose reference has no reviewer; the CLI blocks --check on these."""
    return [rid for rid, ref in references.items() if ref.reviewed_by is None]


def freeze_excerpt(
    *,
    sidecar_path: Path,
    audio: dict[str, Path],
    window: tuple[float, float],
    bench_dir: Path,
    excerpt_id: str,
    glossary_path: Path | None = None,
    owner: str | None = None,
) -> Excerpt:
    """Cut ``window`` from *audio* into the bench and draft its reference.

    *audio* is ``{"mix": path}`` or ``{"me": path, "them": path}``. The draft
    is unreviewed; a reviewed reference is never overwritten. A re-freeze is
    all-or-nothing: bench files change only after every read and cut succeeds.
    """
    _require_id(excerpt_id, "freeze")
    kind = next((k for k, s in _KIND_SLOTS.items() if set(audio) == set(s)), None)
    if kind is None:
        raise ValueError(f"audio keys must be mix or me+them, got {sorted(audio)}")
    start, end = _require_span("freeze", "window", window[0], window[1])
    bench = Path(bench_dir)
    ref_path = bench / "reference" / f"{excerpt_id}.json"
    if ref_path.exists() and _load_json(ref_path).get("reviewed_by"):
        raise ValueError(f"reference {excerpt_id!r} is reviewed; refusing to overwrite")
    sidecar = _load_json(Path(sidecar_path))
    if not isinstance(sidecar, dict) or not isinstance(sidecar.get("segments"), list):
        raise ValueError(f"sidecar {sidecar_path} must carry a 'segments' list")
    # Draft before touching the bench: a malformed sidecar must fail with zero writes.
    draft = _draft_reference(excerpt_id, sidecar, (start, end), owner)
    # Parse the existing manifest up front too: a malformed one must also
    # fail before any audio or reference is written.
    by_id = {e["id"]: e for e in map(asdict, _manifest_entries(bench, required=False))}
    # Cut every slot to a temp; swap all in only once every cut succeeded.
    audio_rel: dict[str, str] = {}
    digests: dict[str, str] = {}
    swapped: list[tuple[Path, Path]] = []
    with contextlib.ExitStack() as cleaner:
        for slot in _KIND_SLOTS[kind]:
            rel = f"audio/{excerpt_id}/{slot}.flac"
            final = bench / rel
            tmp = final.with_name(final.name + ".tmp")
            cleaner.callback(tmp.unlink, missing_ok=True)  # no half-cut survives
            _cut_audio(Path(audio[slot]), tmp, start, end)
            digests[rel] = _sha256_file(tmp)
            audio_rel[slot] = rel
            swapped.append((tmp, final))
        for tmp, final in swapped:
            os.replace(tmp, final)
    entry: dict[str, Any] = {
        "id": excerpt_id,
        "kind": kind,
        "audio": audio_rel,
        "sha256": digests,
        "window": [start, end],
        "source": str(sidecar_path),
        "glossary_sha256": _sha256_file(glossary_path) if glossary_path else None,
        "owner": owner,
    }
    _atomic_write_json(ref_path, draft)
    by_id[excerpt_id] = entry
    out = {"version": MANIFEST_VERSION, "excerpts": list(by_id.values())}
    _atomic_write_json(bench / "manifest.json", out)
    # Re-read through the loader so freeze only ever returns valid entries.
    return _excerpt_from_manifest(entry)


def _draft_reference(
    excerpt_id: str,
    sidecar: dict[str, Any],
    window: tuple[float, float],
    owner: str | None,
) -> dict[str, Any]:
    """Draft segments from the corrected sidecar: keep those overlapping the
    window, clip them to it, shift times to start at 0, map speaker labels
    (unnamed -> "unclear"; *owner* wins on the "me" track).

    A segment straddling a window edge keeps its FULL text while its times
    are clipped, so it gets ``"edge": true`` for the owner to trim by hand.
    Missing/blank text drops the segment: a null must never become "None".
    """
    win_start, win_end = window
    names = sidecar.get("speaker_names") or {}
    segs: list[RefSegment] = []
    for d in sidecar["segments"]:
        if not isinstance(d, dict) or not all(k in d for k in ("start", "end", "text")):
            raise ValueError(f"sidecar segment needs start/end/text: {d!r}")
        s, e = _require_span("sidecar segment", "start/end", d["start"], d["end"])
        if e <= win_start or s >= win_end:
            continue
        text = d["text"]
        if not isinstance(text, str) or not text.strip():
            continue
        label = d.get("speaker")
        speaker = names.get(label) if isinstance(label, str) else None
        if owner is not None and d.get("track") == "me":
            speaker = owner  # the owner's own mic stem beats any diarization label
        edge = d["start"] < win_start or d["end"] > win_end
        s0, e0 = max(s, win_start) - win_start, min(e, win_end) - win_start
        segs.append(RefSegment(s0, e0, speaker or UNCLEAR, text, edge))
    empty = {"must_contain": [], "forbid": []}
    return asdict(Reference(excerpt_id, tuple(segs), empty, None, None))
