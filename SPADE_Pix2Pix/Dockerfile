# ==========================================================
# SPADE Fine-tuning Dockerfile
# Author: Gayatri Patil
# ==========================================================

# ----- Base Image (PyTorch with CUDA) -----
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn8-runtime

# ----- Set working directory -----
WORKDIR /workspace

# ----- Copy files -----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Optional: tqdm color fix for Docker
ENV PYTHONUNBUFFERED=1

# ----- Copy source code -----
COPY src/ src/
COPY data/ data/
COPY outputs_spade_finetune/ outputs_spade_finetune/

# ----- Default command -----
CMD ["python", "src/train_spade.py"]
