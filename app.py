import os
import json
import re
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel
import requests


app = FastAPI(title="Video Speech Extractor", version="1.0.0")


@dataclass
class WordToken:
    start: float
    end: float
    text: str
    speaker: str


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


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


def _extract_json(text: str) -> dict:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\\s*", "", cleaned)
        cleaned = re.sub(r"\\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", cleaned)
    if not match:
        raise ValueError("Risposta LLM non in formato JSON valido")

    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Risposta LLM JSON non valida")
    return parsed


def assign_interlocutors_with_ollama(segments: list[dict], language: str) -> list[dict]:
    if not segments:
        return []

    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434").strip().rstrip("/")
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1:8b").strip()
    ollama_timeout = _safe_float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "600"), 600.0)
    ollama_required = _env_bool("OLLAMA_REQUIRED", default=True)

    compact_segments = []
    for index, seg in enumerate(segments):
        compact_segments.append(
            {
                "index": index,
                "start": round(_safe_float(seg.get("start")), 2),
                "end": round(_safe_float(seg.get("end")), 2),
                "text": (seg.get("text") or "").strip(),
            }
        )

    prompt = (
        "Sei un analista di dialoghi. "
        "Ricevi segmenti cronologici di una trascrizione con start/end/text. "
        "Assegna ogni segmento a un interlocutore stabile (interlocutore_1, interlocutore_2, ...). "
        "Usa il minor numero di interlocutori coerente col testo. "
        "Rispondi SOLO JSON con questo schema: "
        '{"segments":[{"index":0,"interlocutor":"interlocutore_1"}],"interlocutors":[{"id":"interlocutore_1","description":"breve profilo"}]}. '
        f"Lingua prevalente: {language}. "
        f"Segmenti: {json.dumps(compact_segments, ensure_ascii=False)}"
    )

    try:
        response = requests.post(
            f"{ollama_base_url}/api/generate",
            json={
                "model": ollama_model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.1},
            },
            timeout=max(5, int(ollama_timeout)),
        )
        response.raise_for_status()
        raw = response.json().get("response", "")
        parsed = _extract_json(raw)
    except Exception as exc:
        if ollama_required:
            raise RuntimeError(f"Errore chiamata Ollama: {exc}") from exc
        return []

    assignments = parsed.get("segments") or []
    by_index: dict[int, str] = {}
    for item in assignments:
        try:
            idx = int(item.get("index"))
        except (TypeError, ValueError):
            continue
        interlocutor = str(item.get("interlocutor") or "").strip()
        if not interlocutor:
            continue
        by_index[idx] = interlocutor

    enriched: list[dict] = []
    for index, segment in enumerate(segments):
        interlocutor = by_index.get(index, "interlocutore_1")
        enriched.append(
            {
                "start": _safe_float(segment.get("start")),
                "end": _safe_float(segment.get("end"), _safe_float(segment.get("start"))),
                "text": (segment.get("text") or "").strip(),
                "interlocutor": interlocutor,
            }
        )

    return enriched


def build_minute_chunks_from_llm_segments(llm_segments: list[dict], duration_seconds: float) -> list[dict]:
    if not llm_segments:
        return []

    by_minute: dict[int, list[dict]] = defaultdict(list)
    for seg in llm_segments:
        minute_idx = int(_safe_float(seg.get("start")) // 60)
        by_minute[minute_idx].append(seg)

    chunks: list[dict] = []
    for minute_idx in sorted(by_minute.keys()):
        minute_segments = sorted(by_minute[minute_idx], key=lambda s: (_safe_float(s.get("start")), _safe_float(s.get("end"))))
        utterances: list[dict] = []

        current = None
        for seg in minute_segments:
            speaker = str(seg.get("interlocutor") or "interlocutore_1")
            start = _safe_float(seg.get("start"))
            end = _safe_float(seg.get("end"), start)
            text = (seg.get("text") or "").strip()

            if not text:
                continue

            if current is None or current["speaker"] != speaker:
                if current is not None:
                    utterances.append(current)
                current = {
                    "speaker": speaker,
                    "start": start,
                    "end": end,
                    "text": text,
                }
            else:
                current["end"] = end
                current["text"] = f"{current['text']} {text}".strip()

        if current is not None:
            utterances.append(current)

        speakers = sorted({u["speaker"] for u in utterances})
        chunks.append(
            {
                "minute_index": minute_idx,
                "minute_start": minute_idx * 60,
                "minute_end": min((minute_idx + 1) * 60, int(duration_seconds) + 1),
                "text": " ".join(s.get("text", "") for s in minute_segments).strip(),
                "speakers": speakers,
                "interlocutors": utterances,
            }
        )

    return chunks


async def process_transcription(file: UploadFile, model_name: str, include_segments: bool = False) -> dict:
    hf_token = os.getenv("HF_TOKEN", "").strip()
    diarization_enabled = False
    diarization_error = ""
    python_start = time.perf_counter()

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
                "python_text_processing_seconds": round(time.perf_counter() - python_start, 3),
            }

            if include_segments:
                result["_segments"] = segments

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


@app.post("/transcribe/interlocutors")
async def transcribe_video_with_llm_interlocutors(
    file: UploadFile = File(...),
    model_name: str = "small",
) -> JSONResponse:
    total_start = time.perf_counter()
    result = await process_transcription(file=file, model_name=model_name, include_segments=True)
    segments = result.get("_segments") or []

    if not segments:
        raise HTTPException(status_code=500, detail="Nessun segmento disponibile per identificare gli interlocutori")

    llm_start = time.perf_counter()
    try:
        llm_segments = assign_interlocutors_with_ollama(segments=segments, language=str(result.get("language", "")))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if not llm_segments:
        raise HTTPException(status_code=503, detail="Assegnazione interlocutori LLM non disponibile")

    duration_seconds = _safe_float(result.get("duration_seconds"))
    llm_chunks = build_minute_chunks_from_llm_segments(llm_segments=llm_segments, duration_seconds=duration_seconds)
    llm_elapsed = round(time.perf_counter() - llm_start, 3)
    total_elapsed = round(time.perf_counter() - total_start, 3)

    response = {
        "language": result.get("language"),
        "duration_seconds": result.get("duration_seconds"),
        "diarization_enabled": result.get("diarization_enabled"),
        "extracted_text_python": result.get("full_text"),
        "llm_chunks": llm_chunks,
        "chunks": llm_chunks,
        "full_text": result.get("full_text"),
        "llm_interlocutor_detection": True,
        "processing_times_seconds": {
            "python_text_processing": result.get("python_text_processing_seconds", 0.0),
            "llm_processing": llm_elapsed,
            "total": total_elapsed,
        },
    }

    if "diarization_error" in result:
        response["diarization_error"] = result["diarization_error"]

    return JSONResponse(content=response)
