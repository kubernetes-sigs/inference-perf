FROM python:3.13.7-slim-bookworm AS dev

# Upgrade pip
RUN pip3 install --upgrade pip

# Set working directory
WORKDIR /workspace

# Copy project files
COPY inference_perf/ /workspace/inference_perf/
COPY pyproject.toml /workspace/

# Install dependencies & clean cache
RUN pip install . && pip cache purge

# Run inference-perf
CMD ["inference-perf", "--config_file", "config.yml"]
