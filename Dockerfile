FROM ubuntu:22.04

# Set environment variables to prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

WORKDIR /app

# Install system dependencies and add deadsnakes PPA for Python 3.12
RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    libsndfile1 \
    ffmpeg \
    git \
    curl \
    wget \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pyaudio  \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.12
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# Create symbolic links to ensure python and pip commands work
RUN ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/python3.12 /usr/bin/python3

# Install distutils via pip (since the package isn't available)
RUN python -m pip install setuptools

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy the application code
COPY . .

# Make sure the temp_audio directory exists
RUN mkdir -p temp_audio

# Create a script to download models at runtime
RUN echo '#!/bin/bash \n\
ollama serve & \n\
sleep 5 \n\
ollama pull llama3.2:1b \n\
ollama pull llama3:8b \n\
ollama pull gemma \n\
streamlit run app.py --server.port=8501 --server.address=0.0.0.0 \n\
' > /app/start.sh && chmod +x /app/start.sh

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run the application
CMD ["/app/start.sh"]