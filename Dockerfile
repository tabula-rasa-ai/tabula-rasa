FROM python:3.11-slim AS base

WORKDIR /app

# Install PyTorch CPU-only (GPU users can override with CUDA variant)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install project dependencies
COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Install optional extras
RUN pip install --no-cache-dir "psutil>=5.0"

# ── Final stage (smaller image) ─────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy pre-built deps from base
COPY --from=base /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=base /usr/local/bin /usr/local/bin

# Project source
COPY src/ src/

# Configuration
COPY config.py .

# Core scripts
COPY api_server.py .
COPY train_specialist.py .
COPY auto_train.py .

# Egefalos package (sleep cycle, hippocampus, socratic, router, etc.)
COPY egefalos/ egefalos/

# Dashboard static files
COPY Dashboard/ Dashboard/

# Runtime directories (model checkpoints, data, exports)
RUN mkdir -p specialists checkpoints data exports memory

EXPOSE 8000 8002

CMD ["python", "api_server.py"]
