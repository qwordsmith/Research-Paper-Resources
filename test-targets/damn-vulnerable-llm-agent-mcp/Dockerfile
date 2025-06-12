FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip install python-dotenv

COPY * /app/
RUN pip3 install -r requirements.txt

COPY config.toml /root/.streamlit/config.toml

EXPOSE 8502

HEALTHCHECK CMD curl --fail http://localhost:8502/_stcore/health

ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8502", "--server.address=0.0.0.0"]
