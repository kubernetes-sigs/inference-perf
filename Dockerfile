FROM python:3.12.9-slim-bookworm AS dev

# Install PDM
RUN pip3 install --upgrade pip pdm

# Set working directory
WORKDIR /workspace

# Copy dependency files
COPY pyproject.toml pdm.lock /workspace/

# Install dependencies using PDM with the lock file
RUN pdm sync --prod --no-editable && \
    pip cache purge

COPY config.yml /workspace/
COPY inference_perf /workspace/inference_perf

# Set PYTHONPATH
ENV PYTHONPATH=/workspace

# Run inference-perf using PDM's virtual environment
CMD ["pdm", "run", "python", "inference_perf/main.py", "--config_file", "config.yml"]
