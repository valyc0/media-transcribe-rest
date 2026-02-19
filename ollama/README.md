# Ollama Docker Setup

Setup Docker per Ollama con modello `llama3.2:1b` precaricato.

## Quick Start

```bash
# Rendi eseguibile lo script
chmod +x start.sh

# Avvia Ollama e scarica il modello
./start.sh
```

## Configurazione

Il docker-compose avvia:
- **Ollama Server** sulla porta `11434`
- Download automatico del modello `llama3.2:1b`

## Comandi Utili

```bash
# Avvia Ollama
docker-compose up -d

# Verifica stato
docker-compose ps

# Log
docker-compose logs -f ollama

# Lista modelli disponibili
docker exec ollama ollama list

# Testa il modello in modalità interattiva
docker exec -it ollama ollama run llama3.2:1b

# Scarica un altro modello
docker exec ollama ollama pull mistral

# Stop
docker-compose down

# Stop e rimuovi volumi (cancella modelli)
docker-compose down -v
```

## Uso con RAG System

1. Avvia Ollama:
```bash
cd ollama
./start.sh
```

2. Configura il RAG system per usare Ollama:
```bash
cd ..
export LLM_PROVIDER=ollama
export OLLAMA_MODEL=llama3.2:1b
./start.sh
```

## Modelli Disponibili

- `llama3.2:1b` - Velocissimo, 1B parametri (~1GB)
- `llama3.2` - Default 3B parametri (~2GB)
- `llama3.1` - Più potente, 8B parametri (~4.7GB)
- `mistral` - Ottimo rapporto qualità/velocità, 7B (~4GB)
- `phi3` - Compatto, 3.8B parametri (~2.3GB)

Scarica altri modelli:
```bash
docker exec ollama ollama pull <model-name>
```

## Note

- I modelli vengono salvati nel volume Docker `ollama_data`
- Alla prima esecuzione il download del modello può richiedere qualche minuto
- Porta: `11434` (default Ollama)
