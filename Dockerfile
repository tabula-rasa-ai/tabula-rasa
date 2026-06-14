FROM python:3.11-slim

WORKDIR /app

# Install PyTorch (CPU version for smaller image; GPU users can install CUDA variant)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY start_tabula_rasa.bat .
COPY api_server.py .
COPY train_specialist.py .
COPY auto_train.py .
COPY specialist_router.py .
COPY specialist_network.py .
COPY self_improve.py .

# Copy egefalos package
COPY egefalos/ egefalos/

# Create directories for runtime artifacts
RUN mkdir -p checkpoints specialists data exports memory

# Expose API port
EXPOSE 8000

# Default: start the API server
CMD ["python", "api_server.py"]
