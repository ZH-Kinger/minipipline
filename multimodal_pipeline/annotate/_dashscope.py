"""DashScope (Aliyun Qwen-VL) REST client + shared frame-extraction helper.

Pure stdlib (``urllib`` + ``json`` + ``base64``) so we don't pull in the
``dashscope`` SDK; the REST endpoint is stable enough to talk to directly.

Used by ``annotate.language`` and ``annotate.actions`` to share frame
extraction and API plumbing.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from .._system import check_ffmpeg
from ._cache import AnnotationCache


_DEFAULT_ENDPOINT = (
    "https://dashscope.aliyuncs.com/api/v1/services/aigc/"
    "multimodal-generation/generation"
)


class DashScopeError(RuntimeError):
    """Raised for any DashScope-side failure (config, network, response)."""


def require_api_key() -> str:
    key = os.environ.get("MMPIPE_DASHSCOPE_API_KEY", "").strip()
    if not key:
        raise DashScopeError(
            "MMPIPE_DASHSCOPE_API_KEY is not set. Add it to .env (see "
            ".env.example) before using backend=dashscope."
        )
    return key


def extract_clip_frames(
    video_path: Path,
    t_start_s: float,
    t_end_s: float,
    max_frames: int,
    long_edge_px: int,
    *,
    frame_indices: list[int] | None = None,
) -> list[bytes]:
    """Extract JPEG bytes from a clip via ffmpeg.

    Two modes:
      - ``frame_indices`` given: extract exactly those source-frame indices
        (used by the motion-peak sampler). The list should be sorted; values
        are interpreted as 0-based frame numbers in the source video.
      - ``frame_indices`` None: uniformly sample ``max_frames`` over
        ``[t_start_s, t_end_s)`` (legacy behaviour).

    Returned bytes are JPEG-encoded and ready to base64-embed in a DashScope
    request.
    """
    if frame_indices is not None and len(frame_indices) > 0:
        return _extract_by_index(video_path, frame_indices, long_edge_px)

    if t_end_s <= t_start_s:
        return []
    bundle = check_ffmpeg()
    duration = max(t_end_s - t_start_s, 0.05)
    rate = max_frames / duration
    with tempfile.TemporaryDirectory(prefix="mmpipe_dashscope_") as tmpdir:
        out_pattern = str(Path(tmpdir) / "f_%03d.jpg")
        cmd = [
            str(bundle.ffmpeg),
            "-v", "error",
            "-ss", f"{t_start_s:.6f}",
            "-i", str(video_path),
            "-t", f"{duration:.6f}",
            "-vf",
            f"fps={rate:.4f},scale='if(gt(iw,ih),{long_edge_px},-2)':"
            f"'if(gt(iw,ih),-2,{long_edge_px})'",
            "-frames:v", str(max_frames),
            "-q:v", "5",
            out_pattern,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        out: list[bytes] = []
        for p in sorted(Path(tmpdir).glob("f_*.jpg")):
            out.append(p.read_bytes())
    return out


def _extract_by_index(
    video_path: Path, frame_indices: list[int], long_edge_px: int,
) -> list[bytes]:
    """Extract a specific set of source frames via ffmpeg select filter.

    Uses ``select='eq(n,X)+eq(n,Y)+...'`` so ffmpeg decodes the whole stream
    once and emits only the chosen frames. Output order matches source frame
    order (ffmpeg select preserves input order regardless of expression order).
    """
    bundle = check_ffmpeg()
    select_expr = "+".join(f"eq(n\\,{i})" for i in sorted(set(int(x) for x in frame_indices)))
    vf = (
        f"select='{select_expr}',"
        f"scale='if(gt(iw,ih),{long_edge_px},-2)':"
        f"'if(gt(iw,ih),-2,{long_edge_px})'"
    )
    with tempfile.TemporaryDirectory(prefix="mmpipe_dashscope_") as tmpdir:
        out_pattern = str(Path(tmpdir) / "f_%03d.jpg")
        cmd = [
            str(bundle.ffmpeg),
            "-v", "error",
            "-i", str(video_path),
            "-vf", vf,
            "-vsync", "vfr",
            "-q:v", "5",
            out_pattern,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        out: list[bytes] = []
        for p in sorted(Path(tmpdir).glob("f_*.jpg")):
            out.append(p.read_bytes())
    return out


def call_qwen_vl(
    *,
    api_key: str,
    model: str,
    prompt: str,
    frame_bytes_list: list[bytes],
    temperature: float = 0.2,
    top_p: float = 0.8,
    max_tokens: int = 96,
    endpoint: str | None = None,
    timeout_s: float = 30.0,
    cache: AnnotationCache | None = None,
) -> str:
    """Call DashScope's multimodal generation endpoint, return the text output.

    Encodes each frame as ``data:image/jpeg;base64,...`` and ships them as
    ``image`` content items alongside a single text turn. Returns the model's
    plain-text reply (first choice, first content item that is text).

    When ``cache`` is provided, identical inputs (model + prompt + gen params
    + frame content hashes) short-circuit the network call.
    """
    if cache is not None:
        cached = cache.get(
            model=model, prompt=prompt, frame_bytes_list=frame_bytes_list,
            temperature=temperature, top_p=top_p, max_tokens=max_tokens,
        )
        if cached is not None:
            return cached

    endpoint = endpoint or os.environ.get("MMPIPE_DASHSCOPE_ENDPOINT", "").strip() or _DEFAULT_ENDPOINT

    content: list[dict] = []
    for blob in frame_bytes_list:
        b64 = base64.b64encode(blob).decode("ascii")
        content.append({"image": f"data:image/jpeg;base64,{b64}"})
    content.append({"text": prompt})

    body = {
        "model": model,
        "input": {"messages": [{"role": "user", "content": content}]},
        "parameters": {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "result_format": "message",
        },
    }
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        try:
            err_body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            err_body = ""
        raise DashScopeError(
            f"DashScope HTTP {exc.code} {exc.reason}: {err_body[:400]}"
        ) from exc
    except urllib.error.URLError as exc:
        raise DashScopeError(f"DashScope network error: {exc.reason}") from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DashScopeError(f"DashScope returned non-JSON: {raw[:200]!r}") from exc

    output = payload.get("output") or {}
    choices = output.get("choices") or []
    if not choices:
        raise DashScopeError(f"DashScope response had no choices: {payload}")
    message = choices[0].get("message") or {}
    msg_content = message.get("content")
    extracted: str | None = None
    if isinstance(msg_content, str):
        extracted = msg_content.strip()
    elif isinstance(msg_content, list):
        for item in msg_content:
            if isinstance(item, dict) and "text" in item:
                extracted = str(item["text"]).strip()
                break
    if extracted is None:
        raise DashScopeError(f"Could not extract text from DashScope response: {payload}")
    if cache is not None:
        try:
            cache.put(
                model=model, prompt=prompt, frame_bytes_list=frame_bytes_list,
                temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                response_text=extracted,
            )
        except Exception:
            # Cache write failure must not poison a successful API call.
            pass
    return extracted


__all__ = [
    "DashScopeError",
    "extract_clip_frames",
    "call_qwen_vl",
    "require_api_key",
]
