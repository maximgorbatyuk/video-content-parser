#!/usr/bin/env python3
"""
Interactive batch video analyzer for a locally deployed vision LLM
(e.g. llama-server with Qwen2.5-VL or LM Studio with a VL model).

Configuration lives in `.env` next to this script:

    SOURCE_FOLDER=~/photo-projects/istanbul/video   # absolute path; ~ expanded
    OUTPUT_FOLDER=results                           # relative to current working dir
    LLM_URL=http://127.0.0.1:1234

For each video in SOURCE_FOLDER the script:
  1. Samples N evenly-spaced frames using PyAV (ffmpeg).
  2. Base64-encodes them as JPEGs.
  3. Sends them to the LLM's OpenAI-compatible /v1/chat/completions endpoint
     together with a structured prompt.
  4. Saves one JSON per video named after the source video.

The script is resumable: on "Continue analysis" any video whose result JSON
already exists is skipped. "Analyze from scratch" overwrites everything.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Iterable

import av
import requests
from dotenv import load_dotenv
from PIL import Image
from rich.align import Align
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}

PROMPT = (
    "You are analyzing a personal travel/home video. Below are evenly-spaced "
    "frames from a single video, in chronological order. Describe the video "
    "as a whole and respond in this EXACT JSON schema with no extra commentary "
    "and no markdown fences:\n"
    "{\n"
    '  "summary": "1-2 sentence description of what the video is about",\n'
    '  "setting": "guess at the location / type of place (e.g. \'street market in Istanbul\', \'hotel room\', \'unknown\')",\n'
    '  "people": "brief description of any people visible (count, age range, activity)",\n'
    '  "objects": ["list", "of", "notable", "objects", "or", "landmarks"],\n'
    '  "actions": ["list", "of", "main", "actions", "or", "events"],\n'
    '  "mood": "overall mood / vibe of the footage",\n'
    '  "tags": ["short", "keyword", "tags", "for", "search"]\n'
    "}\n"
    "Be concise. If something is not visible, say \"unknown\" or use [].\n"
)

console = Console()


# ---- config --------------------------------------------------------------- #

class Config:
    def __init__(self) -> None:
        here = Path(__file__).resolve().parent
        load_dotenv(here / ".env")

        src = os.getenv("SOURCE_FOLDER", "").strip()
        out = os.getenv("OUTPUT_FOLDER", "results").strip()
        url = os.getenv("LLM_URL", "http://127.0.0.1:1234").strip()

        if not src:
            raise SystemExit(
                "SOURCE_FOLDER is not set. Copy .env.example to .env and fill it in."
            )

        self.source: Path = Path(os.path.expanduser(src)).resolve()
        self.output: Path = (Path.cwd() / out).resolve()
        self.llm_url: str = url.rstrip("/")

        self.frames: int = int(os.getenv("FRAMES", "8"))
        self.max_side: int = int(os.getenv("MAX_SIDE", "448"))
        self.timeout: int = int(os.getenv("TIMEOUT", "600"))
        self.temperature: float = float(os.getenv("TEMPERATURE", "0.1"))
        self.max_tokens: int = int(os.getenv("MAX_TOKENS", "1024"))
        self.disable_thinking: bool = os.getenv("DISABLE_THINKING", "1") not in ("0", "false", "False")


# ---- frame extraction ----------------------------------------------------- #

def probe_duration_seconds(container: "av.container.InputContainer") -> int:
    """Return integer duration of the video in seconds (best-effort)."""
    if container.duration:
        return int(round(container.duration / av.time_base))
    try:
        stream = container.streams.video[0]
        if stream.duration and stream.time_base:
            return int(round(float(stream.duration * stream.time_base)))
    except Exception:
        pass
    return 0


def sample_frames(video_path: Path, n: int, max_side: int) -> tuple[list[Image.Image], int]:
    """Return (frames, duration_seconds). Frames are downscaled so longest side <= max_side."""
    with av.open(str(video_path)) as container:
        duration_s = probe_duration_seconds(container)
        stream = container.streams.video[0]
        total = stream.frames or 0
        if total <= 0:
            duration = float(container.duration or 0) / av.time_base
            fps = float(stream.average_rate or 24)
            total = max(1, int(duration * fps))

        if n == 1:
            targets = [total // 2]
        else:
            step = (total - 1) / (n - 1)
            targets = [int(round(i * step)) for i in range(n)]
        target_set = set(targets)

        frames: dict[int, Image.Image] = {}
        for i, frame in enumerate(container.decode(video=0)):
            if i in target_set:
                img = frame.to_image()
                if max(img.size) > max_side:
                    img.thumbnail((max_side, max_side), Image.LANCZOS)
                frames[i] = img
                if len(frames) == len(target_set):
                    break

    return [frames[i] for i in sorted(frames)], duration_s


def encode_jpeg_b64(img: Image.Image, quality: int = 85) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---- LLM client ----------------------------------------------------------- #

def strip_thinking(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL | re.IGNORECASE)
    if "<think>" in cleaned.lower() and "</think>" not in cleaned.lower():
        idx = cleaned.lower().find("<think>")
        cleaned = cleaned[:idx]
    return cleaned.strip()


def call_server(
    server: str,
    frames: Iterable[Image.Image],
    prompt: str,
    timeout: int,
    temperature: float,
    max_tokens: int,
    disable_thinking: bool,
) -> str:
    content = []
    for img in frames:
        b64 = encode_jpeg_b64(img)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    content.append({"type": "text", "text": prompt})

    payload = {
        "model": "qwen-vl",
        "messages": [{"role": "user", "content": content}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if disable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    r = requests.post(f"{server}/v1/chat/completions", json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    raw = data["choices"][0]["message"]["content"]
    return strip_thinking(raw)


def server_alive(server: str) -> bool:
    """Try /health first, fall back to /v1/models (works for LM Studio)."""
    for path in ("/health", "/v1/models"):
        try:
            r = requests.get(f"{server}{path}", timeout=3)
            if r.ok:
                return True
        except Exception:
            continue
    return False


# ---- analysis pipeline ---------------------------------------------------- #

def parse_json_loose(text: str) -> tuple[dict | None, str | None]:
    text = text.strip()
    try:
        return json.loads(text), None
    except json.JSONDecodeError as e:
        if "{" in text and "}" in text:
            try:
                return json.loads(text[text.index("{") : text.rindex("}") + 1]), None
            except json.JSONDecodeError as e2:
                return None, str(e2)
        return None, str(e)


def analyze_one(video_path: Path, cfg: Config) -> dict:
    frames, duration_s = sample_frames(video_path, cfg.frames, cfg.max_side)
    if not frames:
        raise RuntimeError("could not decode any frames from this video")
    raw = call_server(
        cfg.llm_url, frames, PROMPT, cfg.timeout, cfg.temperature, cfg.max_tokens,
        disable_thinking=cfg.disable_thinking,
    )
    parsed, err = parse_json_loose(raw)
    return {
        "video": video_path.name,
        "frames_used": len(frames),
        "video_length": duration_s,
        "parsed": parsed,
        "raw": raw,
        "parse_error": err,
    }


def list_videos(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in VIDEO_EXTS)


def count_results(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for _ in folder.glob("*.json"))


# ---- interactive UI ------------------------------------------------------- #

def render_dashboard(cfg: Config, n_videos: int, n_results: int, llm_ok: bool) -> Panel:
    pct = (n_results / n_videos * 100) if n_videos else 0.0

    paths = Table.grid(padding=(0, 1))
    paths.add_column(style="bold cyan", justify="right")
    paths.add_column()
    paths.add_row("Source:", str(cfg.source))
    paths.add_row("Output:", str(cfg.output))
    paths.add_row("LLM URL:", cfg.llm_url)

    stats = Table.grid(padding=(0, 1))
    stats.add_column(style="bold cyan", justify="right")
    stats.add_column()
    stats.add_row("Videos found:", str(n_videos))
    stats.add_row("Results on disk:", str(n_results))
    stats.add_row("Progress:", f"{pct:5.1f}%  ({n_results}/{n_videos})")
    status_text = Text("UP", style="bold green") if llm_ok else Text("DOWN", style="bold red")
    stats.add_row("LLM status:", status_text)

    body = Table.grid(padding=(0, 0))
    body.add_column()
    body.add_row(paths)
    body.add_row(Text(""))
    body.add_row(stats)

    return Panel(
        Align.left(body),
        title="[bold]Video Content Parser[/bold]",
        border_style="cyan",
        padding=(1, 2),
    )


def render_menu() -> Panel:
    menu = Table.grid(padding=(0, 1))
    menu.add_column(style="bold yellow", justify="right")
    menu.add_column()
    menu.add_row("[1]", "Continue analysis  (skip videos that already have results)")
    menu.add_row("[2]", "Analyze from scratch  (overwrite existing results)")
    menu.add_row("[3]", "Exit")
    return Panel(menu, title="Menu", border_style="yellow", padding=(1, 2))


def show_main_view_and_choose(cfg: Config) -> str:
    """Render dashboard + menu, return one of '1'/'2'/'3'."""
    while True:
        n_videos = len(list_videos(cfg.source))
        n_results = count_results(cfg.output)
        llm_ok = server_alive(cfg.llm_url)

        console.clear()
        console.print(render_dashboard(cfg, n_videos, n_results, llm_ok))
        console.print(render_menu())

        if not cfg.source.exists():
            console.print(
                f"[bold red]Source folder does not exist:[/bold red] {cfg.source}"
            )
        if not llm_ok:
            console.print(
                f"[bold yellow]Warning:[/bold yellow] cannot reach LLM at {cfg.llm_url}. "
                "Start it before running analysis."
            )

        choice = console.input("\n[bold]Select option [1/2/3]:[/bold] ").strip()
        if choice in ("1", "2", "3"):
            return choice
        console.print("[red]Invalid choice.[/red]")
        time.sleep(0.6)


# ---- run loop ------------------------------------------------------------- #

def short_summary(result: dict) -> str:
    parsed = result.get("parsed") or {}
    if result.get("parse_error"):
        return f"[yellow]parse_error[/yellow] — raw[:80]: {(result.get('raw') or '')[:80]!r}"
    summary = parsed.get("summary", "")
    tags = parsed.get("tags") or []
    return f"{summary[:100]}  [dim]tags: {', '.join(tags[:5])}[/dim]"


def run_analysis(cfg: Config, overwrite: bool) -> None:
    videos = list_videos(cfg.source)
    if not videos:
        console.print(f"[red]No videos found in[/red] {cfg.source}")
        return

    cfg.output.mkdir(parents=True, exist_ok=True)

    # Decide work set up front so the progress bar reflects only what will run.
    work: list[Path] = []
    skipped = 0
    for v in videos:
        out_json = cfg.output / f"{v.stem}.json"
        if out_json.exists() and not overwrite:
            skipped += 1
            continue
        work.append(v)

    if not work:
        console.print(
            f"[green]Nothing to do — all {len(videos)} videos already have results.[/green]"
        )
        return

    console.print(
        f"[bold]Starting:[/bold] {len(work)} to process "
        f"({skipped} already done, {len(videos)} total). "
        f"Press [bold]Ctrl+C[/bold] to stop — already-saved results are kept."
    )

    processed = 0
    failed = 0
    interrupted = False
    t0 = time.time()

    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )

    try:
        with progress:
            task_id = progress.add_task("analyzing", total=len(work))
            for v in work:
                out_json = cfg.output / f"{v.stem}.json"
                progress.update(task_id, description=f"analyzing {v.name}")

                try:
                    result = analyze_one(v, cfg)
                except KeyboardInterrupt:
                    interrupted = True
                    break
                except Exception as e:
                    failed += 1
                    err = {
                        "video": v.name,
                        "error": str(e),
                        "trace": traceback.format_exc(),
                    }
                    out_json.write_text(json.dumps(err, indent=2, ensure_ascii=False))
                    console.log(f"[red]FAIL[/red] {v.name}: {e}")
                    progress.advance(task_id)
                    continue

                out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False))
                processed += 1
                console.log(f"[green]OK[/green] {v.name}  →  {short_summary(result)}")
                progress.advance(task_id)
    except KeyboardInterrupt:
        interrupted = True

    if interrupted:
        console.print(
            "\n[yellow]Stopped by user. Re-run and choose [1] Continue to resume.[/yellow]"
        )

    dt = time.time() - t0
    remaining = len(work) - processed - failed

    report = Table(title="Run report", show_header=True, header_style="bold cyan")
    report.add_column("Metric", style="bold")
    report.add_column("Value", justify="right")
    report.add_row("Total videos in source", str(len(videos)))
    report.add_row("Already had results (skipped)", str(skipped))
    report.add_row("Processed this run", str(processed))
    report.add_row("Failed this run", str(failed))
    report.add_row("Not reached (stopped early)", str(remaining))
    report.add_row("Elapsed", f"{dt:.1f}s")
    console.print(report)


# ---- main ----------------------------------------------------------------- #

def main() -> int:
    try:
        cfg = Config()
    except SystemExit as e:
        console.print(f"[red]{e}[/red]")
        return 2

    try:
        choice = show_main_view_and_choose(cfg)
        if choice == "3":
            console.print("Bye.")
            return 0

        if not server_alive(cfg.llm_url):
            console.print(
                f"[red]Cannot reach LLM at {cfg.llm_url}. Start it and try again.[/red]"
            )
            return 2

        overwrite = (choice == "2")
        if overwrite:
            console.print("[bold red]Mode: analyze from scratch (overwriting existing results)[/bold red]")
        else:
            console.print("[bold green]Mode: continue (skipping existing results)[/bold green]")

        run_analysis(cfg, overwrite=overwrite)
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled.[/yellow]")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
