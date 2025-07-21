#!/bin/bash

set -e  # Para parar em caso de erro
set -u  # Para avisar se usar variáveis não definidas

echo "==== Instalando Ollama ===="
curl -fsSL https://ollama.com/install.sh | sh

echo "==== Iniciando servidor Ollama em background ===="
ollama serve &

sleep 2  # Pequena pausa para garantir que o servidor suba

echo "==== Baixando modelos ===="
# Modelos de Chat: 
# - llama3.2
# - cogito:3b
# - 

# Modelos de Embedding:
# - mxbai-embed-large
# - nomic-embed-text


ollama pull llama3.2:latest
ollama pull mxbai-embed-large

# echo "==== Clonando repositório ===="
# git clone https://github.com/brenofandrade/ollama_pdf_rag.git

# cd ollama_pdf_rag

echo "==== Criando ambiente virtual ===="
python3 -m venv env
source env/bin/activate

echo "==== Removendo python-blinker se instalado ===="
sudo apt-get remove -y python-blinker || true

echo "==== Atualizando pip ===="
python -m pip install --upgrade pip

echo "==== Instalando dependências do projeto ===="
pip install -r requirements.txt

echo "==== Iniciando aplicação Streamlit ===="
streamlit run app.py \
    --server.address 0.0.0.0 \
    --server.port 7860 \
    --server.enableCORS false \
    --server.headless true \
    --server.enableWebsocketCompression false
