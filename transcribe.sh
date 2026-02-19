#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Uso: $0 [--full-text|-f|--interlocutors|-i] /percorso/video.mp4 [model_name]"
  echo "Esempi:"
  echo "  $0 /home/valerio/lavoro/appo/aa/videoplayback.mp4 small"
  echo "  $0 --full-text /home/valerio/lavoro/appo/aa/videoplayback.mp4 small"
  echo "  $0 -f /home/valerio/lavoro/appo/aa/videoplayback.mp4 small"
  echo "  $0 --interlocutors /home/valerio/lavoro/appo/aa/videoplayback.mp4 small"
  echo "  $0 -i /home/valerio/lavoro/appo/aa/videoplayback.mp4 small"
}

FULL_TEXT=false
INTERLOCUTORS=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --full-text|-f)
      FULL_TEXT=true
      shift
      ;;
    --interlocutors|-i)
      INTERLOCUTORS=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      break
      ;;
  esac
done

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

VIDEO_PATH="$1"
MODEL_NAME="${2:-small}"
API_URL="${API_URL:-http://localhost:${HOST_PORT:-8001}}"
OUT_FILE="${OUT_FILE:-transcription_$(date +%Y%m%d_%H%M%S).json}"

ENDPOINT_PATH="/transcribe"
if [[ "$FULL_TEXT" == "true" ]]; then
  ENDPOINT_PATH="/transcribe/full-text"
fi
if [[ "$INTERLOCUTORS" == "true" ]]; then
  ENDPOINT_PATH="/transcribe/interlocutors"
fi

if [[ ! -f "$VIDEO_PATH" ]]; then
  echo "Errore: file non trovato: $VIDEO_PATH"
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "Errore: curl non trovato nel PATH"
  exit 1
fi

echo "Invio file: $VIDEO_PATH"
echo "Endpoint: ${API_URL}${ENDPOINT_PATH}?model_name=${MODEL_NAME}"

HTTP_CODE=$(curl -sS -w "%{http_code}" -o "$OUT_FILE" \
  -X POST "${API_URL}${ENDPOINT_PATH}?model_name=${MODEL_NAME}" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@${VIDEO_PATH}")

if [[ "$HTTP_CODE" -ge 200 && "$HTTP_CODE" -lt 300 ]]; then
  echo "Trascrizione completata. Output salvato in: $OUT_FILE"
  if command -v jq >/dev/null 2>&1; then
    echo "--- Anteprima JSON ---"
    jq '.' "$OUT_FILE"
  fi
else
  echo "Errore API (HTTP $HTTP_CODE). Risposta salvata in: $OUT_FILE"
  exit 1
fi
