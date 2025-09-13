# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.10.11
FROM python:${PYTHON_VERSION}-slim as base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Create a non-privileged user with a valid home directory
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/home/appuser" \
    --shell "/sbin/nologin" \
    --uid "${UID}" \
    appuser && \
    mkdir -p /home/appuser/.cache && \
    chown -R appuser /home/appuser

# Set environment variable so Hugging Face and transformers can write cache
ENV HF_HOME=/home/appuser/.cache/huggingface
ENV TRANSFORMERS_CACHE=/home/appuser/.cache/transformers

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt

# Copy source code
COPY . .

# Set user and expose port
USER appuser
EXPOSE 8000

# Run app
CMD ["python", "app.py"]
