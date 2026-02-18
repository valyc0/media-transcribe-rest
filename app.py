import os
import shutil
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel


app = FastAPI(title="Video Speech Extractor", version="1.0.0")


@dataclass
class WordToken:
    start: float
    end: float
    text: str
    speaker: str


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def extract_audio(video_path: str, audio_path: str) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-ac",
        "1",
        "-ar",
        "16000",
        "-vn",
        audio_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Errore ffmpeg: {result.stderr.strip()}")


def transcribe_audio(audio_path: str, model_name: str) -> tuple[list[dict], str]:
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, info = model.transcribe(audio_path, word_timestamps=True, vad_filter=True)

    segment_list: list[dict] = []
    for segment in segments:
        words = []
        for word in segment.words or []:
            words.append(
                {
                    "start": _safe_float(word.start),
                    "end": _safe_float(word.end),
                    "word": (word.word or "").strip(),
                }
            )

        segment_list.append(
            {
                "start": _safe_float(segment.start),
                "end": _safe_float(segment.end),
                "text": (segment.text or "").strip(),
                "words": words,
            }
        )

    return segment_list, info.language


def diarize_audio(audio_path: str, hf_token: str) -> list[dict]:
    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "Diarizzazione non disponibile: installa requirements-diarization.txt"
        ) from exc

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=hf_token
    )
    diarization = pipeline(audio_path)

    spans: list[dict] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        spans.append(
            {
                "start": _safe_float(turn.start),
                "end": _safe_float(turn.end),
                "speaker": str(speaker),
            }
        )

    return spans


def overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def assign_speakers(transcribed_segments: list[dict], diarization_spans: list[dict]) -> list[WordToken]:
    tokens: list[WordToken] = []

    for segment in transcribed_segments:
        words = segment.get("words") or []

        if not words and segment.get("text"):
            words = [
                {
                    "start": segment.get("start", 0.0),
                    "end": segment.get("end", 0.0),
                    "word": segment.get("text", "").strip(),
                }
            ]

        for word in words:
            w_start = _safe_float(word.get("start"))
            w_end = _safe_float(word.get("end"), w_start)
            w_text = (word.get("word") or "").strip()

            if not w_text:
                continue

            best_speaker = "SPEAKER_00"
            best_overlap = 0.0

            for span in diarization_spans:
                ov = overlap(w_start, w_end, span["start"], span["end"])
                if ov > best_overlap:
                    best_overlap = ov
                    best_speaker = span["speaker"]

            tokens.append(
                WordToken(
                    start=w_start,
                    end=w_end,
                    text=w_text,
                    speaker=best_speaker,
                )
            )

    return sorted(tokens, key=lambda t: (t.start, t.end))


def build_minute_chunks(tokens: list[WordToken], duration_seconds: float) -> list[dict]:
    if not tokens:
        return []

    by_minute: dict[int, list[WordToken]] = defaultdict(list)
    for token in tokens:
        minute_idx = int(token.start // 60)
        by_minute[minute_idx].append(token)

    minute_keys = sorted(by_minute.keys())
    chunks: list[dict] = []

    for minute_idx in minute_keys:
        minute_tokens = by_minute[minute_idx]

        utterances: list[dict] = []
        current = None

        for token in minute_tokens:
            if current is None or current["speaker"] != token.speaker:
                if current is not None:
                    utterances.append(current)
                current = {
                    "speaker": token.speaker,
                    "start": token.start,
                    "end": token.end,
                    "text": token.text,
                }
            else:
                current["end"] = token.end
                current["text"] = f"{current['text']} {token.text}".strip()

        if current is not None:
            utterances.append(current)

        speakers = sorted({u["speaker"] for u in utterances})

        chunks.append(
            {
                "minute_index": minute_idx,
                "minute_start": minute_idx * 60,
                "minute_end": min((minute_idx + 1) * 60, int(duration_seconds) + 1),
                "text": " ".join(t.text for t in minute_tokens).strip(),
                "speakers": speakers,
                "interlocutors": utterances,
            }
        )

    return chunks


def build_full_text(transcribed_segments: list[dict]) -> str:
    parts: list[str] = []
    for segment in transcribed_segments:
        text = (segment.get("text") or "").strip()
        if text:
            parts.append(text)
    return " ".join(parts).strip()


async def process_transcription(file: UploadFile, model_name: str) -> dict:
    hf_token = os.getenv("HF_TOKEN", "").strip()
    diarization_enabled = False
    diarization_error = ""

    with tempfile.TemporaryDirectory(prefix="video-transcribe-") as tmpdir:
        video_path = os.path.join(tmpdir, file.filename or "input-video")
        audio_path = os.path.join(tmpdir, "audio.wav")

        with open(video_path, "wb") as out_file:
            shutil.copyfileobj(file.file, out_file)

        try:
            extract_audio(video_path, audio_path)
            segments, language = transcribe_audio(audio_path, model_name=model_name)

            if hf_token:
                try:
                    diarization_spans = diarize_audio(audio_path, hf_token)
                    diarization_enabled = True
                except RuntimeError as exc:
                    diarization_spans = []
                    diarization_error = str(exc)
            else:
                diarization_spans = []

            tokens = assign_speakers(segments, diarization_spans)
            duration_seconds = max((t.end for t in tokens), default=0.0)
            chunks = build_minute_chunks(tokens, duration_seconds)
            full_text = build_full_text(segments)

            result = {
                "language": language,
                "duration_seconds": round(duration_seconds, 2),
                "chunks": chunks,
                "full_text": full_text,
                "diarization_enabled": diarization_enabled,
            }

            if diarization_error:
                result["diarization_error"] = diarization_error

            return result
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe_video(
    file: UploadFile = File(...),
    model_name: str = "small",
) -> JSONResponse:
    result = await process_transcription(file=file, model_name=model_name)
    return JSONResponse(content=result)


@app.post("/transcribe/full-text")
async def transcribe_video_full_text(
    file: UploadFile = File(...),
    model_name: str = "small",
) -> JSONResponse:
    result = await process_transcription(file=file, model_name=model_name)

    response = {
        "language": result["language"],
        "duration_seconds": result["duration_seconds"],
        "full_text": result["full_text"],
        "diarization_enabled": result["diarization_enabled"],
    }

    if "diarization_error" in result:
        response["diarization_error"] = result["diarization_error"]

    return JSONResponse(content=response)
