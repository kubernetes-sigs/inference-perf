FROM python:3.12.9-slim-bookworm AS dev

RUN apt-get update -y \
    && apt-get install -y python3-pip

# Upgrade pip
RUN pip3 install --upgrade pip

# Set working directory
WORKDIR /workspace

# Copy project files
COPY . /workspace

# Install dependencies
RUN pip install -e .

# Run inference-perf
CMD ["inference-perf", "--config_file", "config.yml"]
