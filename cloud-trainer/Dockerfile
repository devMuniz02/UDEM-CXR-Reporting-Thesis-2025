# Use Vertex AI PyTorch base image
FROM us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest

# Install system packages (optional, good for cv2, matplotlib, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

# Create workdir
WORKDIR /app

# Copy dependency list and install first (caches layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY trainer /app/trainer
COPY utils /app/utils

# Make utils importable
ENV PYTHONPATH=/app

# Default command (can be overridden by Vertex)
ENTRYPOINT ["python", "-m", "trainer.train"]

# # Use Google Cloud Workbench base image (provides Jupyter startup scripts)
# FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/workbench-container:latest

# # Install system packages (optional, good for cv2, matplotlib, etc.)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     libgl1 libglib2.0-0 ffmpeg git && \
#     rm -rf /var/lib/apt/lists/*

# # Create workdir
# WORKDIR /app

# # Copy dependency list and install first (caches layer)
# COPY requirements.txt .
# RUN pip install --no-cache-dir --upgrade-strategy only-if-needed -r requirements.txt

# # Copy code
# COPY trainer /app/trainer
# COPY utils /app/utils

# # Make utils importable
# ENV PYTHONPATH=/app

# # Start JupyterLab directly using the workbench-provided script.
# ENTRYPOINT ["/run_jupyter.sh"]
