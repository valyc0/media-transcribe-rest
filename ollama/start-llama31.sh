#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_NAME="llama3.1:8b"

echo "🚀 Avvio setup Ollama + modello ${MODEL_NAME}"
echo ""

"${SCRIPT_DIR}/start.sh"

echo ""
echo "📥 Verifica/installazione modello ${MODEL_NAME}..."

docker exec ollama ollama pull "${MODEL_NAME}"

echo ""
echo "✅ Modello ${MODEL_NAME} pronto"
echo ""
echo "🧪 Test rapido:"
echo "   docker exec -it ollama ollama run ${MODEL_NAME} \"Ciao, presentati in una riga\""
