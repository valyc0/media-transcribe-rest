# REST API estrazione testo da video

Servizio REST in FastAPI che:
- riceve un file video,
- estrae l'audio,
- trascrive il parlato,
- suddivide il risultato in chunk da 1 minuto,
- identifica gli interlocutori (speaker diarization) opzionale.

## Requisiti

- Python 3.11+
- `ffmpeg` installato nel sistema
- (Opzionale) token Hugging Face in `HF_TOKEN` per diarizzazione speaker

### Installazione rapida

```bash
cd /home/valerio/lavoro/appo/aa
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Abilitare diarizzazione speaker (opzionale)

```bash
pip install -r requirements-diarization.txt
export HF_TOKEN=tuo_token_hf
```

## Avvio server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## Avvio con Docker

Build immagine:

```bash
cd /home/valerio/lavoro/appo/aa
docker build -t video-transcriber:latest .
```

Run container:

```bash
docker run --rm -p 8000:8000 video-transcriber:latest
```

Run container con diarizzazione speaker (token Hugging Face):

```bash
docker build --build-arg INSTALL_DIARIZATION=true -t video-transcriber:latest .
docker run --rm -p 8000:8000 -e HF_TOKEN=tuo_token_hf video-transcriber:latest
```

Nota: l'immagine Docker standard è ottimizzata per startup veloce e non include `pyannote.audio`.
Per la diarizzazione in Docker usa il build arg `INSTALL_DIARIZATION=true`.

## Avvio con Docker Compose

Avvio standard:

```bash
cd /home/valerio/lavoro/appo/aa
docker compose up --build
```

Con endpoint interlocutori LLM (Ollama locale):

```bash
cd /home/valerio/lavoro/appo/aa
export OLLAMA_BASE_URL=http://host.docker.internal:11434
export OLLAMA_MODEL=llama3.1:8b
docker compose up --build
```

Avvio con diarizzazione speaker:

```bash
cd /home/valerio/lavoro/appo/aa
export HF_TOKEN=tuo_token_hf
INSTALL_DIARIZATION=true docker compose up --build
```

Avvio rapido con script:

```bash
cd /home/valerio/lavoro/appo/aa
./start.sh
```

Avvio script con diarizzazione:

```bash
cd /home/valerio/lavoro/appo/aa
HF_TOKEN=tuo_token_hf ./start.sh --diarization
```

Trascrizione via script (chunk per minuto):

```bash
cd /home/valerio/lavoro/appo/aa
./transcribe.sh /percorso/video.mp4 small
```

Trascrizione via script (solo testo completo):

```bash
cd /home/valerio/lavoro/appo/aa
./transcribe.sh --full-text /percorso/video.mp4 small
```

Stop:

```bash
docker compose down
```

## Endpoint

### Health

```http
GET /health
```

### Trascrizione video

```http
POST /transcribe
Content-Type: multipart/form-data
```

Parametri:
- `file`: file video (mp4, mov, mkv, ...)
- `model_name` (query, opzionale): modello Whisper (`tiny`, `base`, `small`, `medium`, `large-v3`), default `small`

Esempio con `curl`:

```bash
curl -X POST "http://localhost:8000/transcribe?model_name=small" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/percorso/video.mp4"
```

### Trascrizione testo completo

```http
POST /transcribe/full-text
Content-Type: multipart/form-data
```

Parametri:
- `file`: file video (mp4, mov, mkv, ...)
- `model_name` (query, opzionale): modello Whisper (`tiny`, `base`, `small`, `medium`, `large-v3`), default `small`

Esempio con `curl`:

```bash
curl -X POST "http://localhost:8001/transcribe/full-text?model_name=small" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/percorso/video.mp4"
```

### Trascrizione con interlocutori LLM

Identifica gli interlocutori (`interlocutore_1`, `interlocutore_2`, ...) usando una LLM locale via Ollama e restituisce i tempi per ogni intervento.

```http
POST /transcribe/interlocutors
Content-Type: multipart/form-data
```

Parametri:
- `file`: file video (mp4, mov, mkv, ...)
- `model_name` (query, opzionale): modello Whisper (`tiny`, `base`, `small`, `medium`, `large-v3`), default `small`

Variabili ambiente Ollama:
- `OLLAMA_BASE_URL` (default `http://host.docker.internal:11434`)
- `OLLAMA_MODEL` (default `llama3.1:8b`)
- `OLLAMA_TIMEOUT_SECONDS` (default `600`)
- `OLLAMA_REQUIRED` (default `true`) se `true`, in caso di errore LLM l'endpoint risponde `503`

Esempio con `curl`:

```bash
curl -X POST "http://localhost:8001/transcribe/interlocutors?model_name=small" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/percorso/video.mp4"
```

Campi principali della risposta:
- `extracted_text_python`: testo completo estratto da Whisper/Faster-Whisper (Python)
- `llm_chunks`: chunk temporali con `minute_start`, `minute_end` e `interlocutors` (`interlocutore_1`, `interlocutore_2`, ...)
- `chunks`: alias compatibile di `llm_chunks`
- `processing_times_seconds`: tempi in secondi con `python_text_processing`, `llm_processing`, `total`

## Esempio risposta JSON

```json
{
  "language": "it",
  "duration_seconds": 137.5,
  "diarization_enabled": true,
  "chunks": [
    {
      "minute_index": 0,
      "minute_start": 0,
      "minute_end": 60,
      "text": "...",
      "speakers": ["SPEAKER_00", "SPEAKER_01"],
      "interlocutors": [
        {
          "speaker": "SPEAKER_00",
          "start": 2.1,
          "end": 7.8,
          "text": "Buongiorno a tutti"
        }
      ]
    }
  ]
}
```

## Note importanti

- Se `HF_TOKEN` non è impostato, la trascrizione funziona ma gli speaker saranno assegnati a un interlocutore di default (`SPEAKER_00`).
- Se `HF_TOKEN` è impostato ma `pyannote.audio` non è installato, la trascrizione funziona e nel JSON trovi `diarization_error`.
- Il primo avvio può essere lento perché scarica i modelli.
- La qualità dipende da audio, rumore di fondo e lingua.
