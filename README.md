# mvp-aiops-agent

MVP agent for code/config RAG based on local vector index (FAISS) + SQLite payload.

## Setup
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt

## Build index
python -m agent.cli index --repo /path/to/repo --out ./data/index/payments

## Run retrieval
python -m agent.cli run --index ./data/index/payments --incident ./data/incidents/incident.sample.json --topk 12