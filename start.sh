#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

INSTALL_DIARIZATION="${INSTALL_DIARIZATION:-false}"

if [[ "${1:-}" == "--diarization" ]]; then
  INSTALL_DIARIZATION="true"
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Errore: docker non trovato nel PATH"
  exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
  echo "Errore: docker compose non disponibile"
  exit 1
fi

echo "Avvio servizio video-transcriber con Docker Compose..."
INSTALL_DIARIZATION="$INSTALL_DIARIZATION" docker compose up --build -d

HOST_PORT="${HOST_PORT:-8001}"
echo "Servizio avviato su http://localhost:${HOST_PORT}"
if [[ "$INSTALL_DIARIZATION" == "true" ]]; then
  echo "Modalità diarizzazione: ATTIVA (build più lenta)"
else
  echo "Modalità diarizzazione: DISATTIVA (build rapida)"
fi
echo "Per vedere i log: docker compose logs -f"
echo "Per fermare: docker compose down"
