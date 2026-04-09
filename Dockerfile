# Build stage - install dependencies
FROM python:3.12.11-alpine3.22 AS builder

# Install PDM
RUN pip install --no-cache-dir pdm

WORKDIR /workspace

# Copy dependency files
COPY pyproject.toml pdm.lock ./

# Copy source code (needed for PDM to resolve the project)
COPY inference_perf ./inference_perf

# Install dependencies using PDM (this will create .venv and install all prod dependencies)
RUN pdm install --prod --no-lock --no-editable && \
    pip cache purge

# Runtime stage - minimal image
FROM python:3.12.11-alpine3.22

WORKDIR /workspace

# Copy installed dependencies from builder (PDM's virtual environment)
COPY --from=builder /workspace/.venv /workspace/.venv

# Copy application code
COPY config.yml ./
COPY inference_perf ./inference_perf
COPY tools ./tools

# Set PYTHONPATH and PATH to use virtual environment
ENV PYTHONPATH=/workspace
ENV PATH="/workspace/.venv/bin:$PATH"

# Entrypoint generates synthetic traces if SYNTHETIC_TRACE_CONFIG is set,
# then execs the CMD (inference-perf or whatever the K8s job overrides).
ENTRYPOINT ["/workspace/tools/entrypoint.sh"]
CMD ["python", "inference_perf/main.py", "--config_file", "config.yml"]
