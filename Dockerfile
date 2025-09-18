FROM python:3.12.9-slim-bookworm AS dev

# Upgrade pip
RUN pip3 install --upgrade pip pip-tools

# Set working directory
WORKDIR /workspace

# Copy dependency files
COPY pyproject.toml /workspace/

RUN python -m piptools compile --generate-hashes --resolver=backtracking pyproject.toml -o requirements.txt && \
    pip install -r requirements.txt && \
    pip cache purge

COPY config.yml /workspace/
COPY inference_perf /workspace/inference_perf

# Set PYTHONPATH
ENV PYTHONPATH=/workspace

# Run inference-perf
CMD ["python", "inference_perf/main.py", "--config_file", "config.yml"]
