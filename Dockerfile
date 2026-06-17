FROM python:3.11-slim AS base

WORKDIR /app

# Install PyTorch CPU-only (GPU users can override with CUDA variant)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install project dependencies
COPY pyproject.toml requirements-lock.txt ./
RUN pip install --no-cache-dir -r requirements-lock.txt

# ── Final stage (smaller image) ─────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy pre-built deps from base
COPY --from=base /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=base /usr/local/bin /usr/local/bin

# Install the package itself
COPY src/ src/
COPY pyproject.toml requirements-lock.txt ./
RUN pip install --no-cache-dir -e .

# Scripts (CLI tools, training, etc.)
COPY scripts/ scripts/

# Dashboard static files
COPY Dashboard/ Dashboard/

# Runtime directories (model checkpoints, data, exports)
RUN mkdir -p specialists checkpoints data exports memory

EXPOSE 8000 8002

CMD ["tabula-rasa", "serve"]
