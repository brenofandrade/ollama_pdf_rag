# ollama_pdf_rag
Execute os comandos abaixo em blocos separados:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

```bash
ollama serve &
```

```bash
ollama pull llama3.2:latest
```

```bash
ollama pull mxbai-embed-large
```

```bash
git clone https://github.com/brenofandrade/ollama_pdf_rag.git
```

```bash
python -m venv env
```

```bash
source env/bin/activate
```

```bash
apt remove python-blinker
```

```bash
python -m pip install --upgrade pip
```

```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 7860 --server.enableCORS false --server.headless true --server.enableWebsocketCompression false
```

### SETUP with docker container

```bash
docker build -t streamlit-app .
```


```bash
docker run -d -p 7860:7860 streamlit-app
```