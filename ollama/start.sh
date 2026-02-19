#!/bin/bash

echo "🦙 Avvio Ollama con modello llama3.2:1b"
echo ""

# Ferma eventuali container esistenti
echo "🧹 Pulizia container esistenti..."
docker compose down > /dev/null 2>&1

# Avvia Ollama
echo "1️⃣  Avvio container Ollama..."
docker compose up -d

echo ""
echo "⏳ Attendo che Ollama sia pronto (max 30 secondi)..."
COUNTER=0
until curl -s http://localhost:11434/api/tags > /dev/null 2>&1 || [ $COUNTER -eq 30 ]; do
    echo -n "."
    sleep 1
    COUNTER=$((COUNTER+1))
done
echo ""

if [ $COUNTER -eq 30 ]; then
    echo "❌ Timeout: Ollama non risponde dopo 30 secondi"
    echo "   Controlla i log con: docker compose logs ollama"
    exit 1
fi

echo "✅ Ollama è pronto!"
echo ""

# Verifica se il modello esiste già
echo "2️⃣  Verifica modello llama3.2:1b..."
MODEL_EXISTS=$(docker exec ollama ollama list | grep -c "llama3.2:1b" || true)

if [ "$MODEL_EXISTS" -gt 0 ]; then
    echo "✅ Modello llama3.2:1b già presente"
else
    echo "📥 Download modello llama3.2:1b in corso..."
    echo "   (Questa operazione può richiedere qualche minuto alla prima esecuzione)"
    docker exec ollama ollama pull llama3.2:1b
    
    if [ $? -eq 0 ]; then
        echo "✅ Modello llama3.2:1b scaricato con successo!"
    else
        echo "❌ Errore durante il download del modello"
        exit 1
    fi
fi

echo ""
echo "3️⃣  Modelli disponibili:"
docker exec ollama ollama list

echo ""
echo "✅ Setup completato!"
echo ""
echo "🔗 Ollama è in esecuzione su: http://localhost:11434"
echo ""
echo "Comandi utili:"
echo "  - Verifica stato: docker compose ps"
echo "  - Log: docker compose logs -f ollama"
echo "  - Stop: docker compose down"
echo "  - Scarica altro modello: docker exec ollama ollama pull <nome-modello>"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧪 Per testare il modello, esegui:"
echo "   docker exec -it ollama ollama run llama3.2:1b \"Ciao, presentati in una riga\""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
