FROM python:3.9.20-slim-bookworm as dev

RUN apt-get update -y \
    && apt-get install -y python3-pip

# Install PDM
RUN pip3 install --upgrade pip
RUN pip3 install pdm

# Set working directory
WORKDIR /workspace

# Copy project files
COPY . /workspace

# Install dependencies using PDM
RUN pdm install

# Run inference-perf (example, adjust as needed)
CMD ["pdm", "run", "inference-perf", "--config_file", "config.yml"]
