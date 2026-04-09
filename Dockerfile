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

# Generate synthetic traces at build time (deterministic — same seed = same files).
# These are baked into the image so no runtime generation is needed.
# To regenerate with different distributions, edit tools/default_trace_config.yaml and rebuild.
RUN python tools/generate_synthetic_traces.py --config tools/default_trace_config.yaml

CMD ["python", "inference_perf/main.py", "--config_file", "config.yml"]
