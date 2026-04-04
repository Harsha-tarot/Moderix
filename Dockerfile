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
ENV GEMINI_MODEL_NAME=gemini-1.5-flash
# ENV GEMINI_API_KEY=YOUR_API_KEY

CMD ["python", "inference.py"]
