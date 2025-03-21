FROM python:3.12.3

WORKDIR /

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Make sure the temp_audio directory exists
RUN mkdir -p temp_audio

# Make sure the model audio files exist (placeholder command)
# Note: Make sure audio1.wav and audio2.wav are included in your build context
# or adjust this section to download or generate them

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run the application
CMD ["sh", "-c", "ollama serve & streamlit run app.py"]