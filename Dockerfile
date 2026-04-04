# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY my_env.py .
COPY inference.py .
COPY graders/ ./graders/
COPY data/ ./data/

# Set environment defaults
ENV API_BASE_URL=https://router.huggingface.co/v1
ENV MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
# ENV HF_TOKEN=hf_xxxxx

CMD ["python", "inference.py"]
