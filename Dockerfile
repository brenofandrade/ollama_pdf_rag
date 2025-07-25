FROM python:3.10-slim

RUN apt update

WORKDIR /app

COPY . /app

RUN python -m venv /opt/venv 

RUN pip install --upgrade pip && pip install -r requirements.txt

EXPOSE 7860 

CMD streamlit run app.py \
    --server.address 0.0.0.0 \
    --server.port 7860 \
    --server.enableCORS false \
    --server.headless true \
    --server.enableWebsocketCompression false