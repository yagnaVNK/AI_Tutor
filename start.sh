#!/bin/bash

# Start the Ollama service in the background
echo "Starting Ollama service..."
ollama serve &

# Wait for Ollama service to fully start
echo "Waiting for Ollama service to initialize..."
sleep 5

# Pull the required models
echo "Pulling llama3.2:1b model..."
ollama pull llama3.2:1b

echo "Pulling llama3:8b model..."
ollama pull llama3:8b

echo "Pulling gemma model..."
ollama pull gemma

# Login to Hugging Face
echo "Logging in to Hugging Face..."
if [ -n "$HUGGING_FACE_HUB_TOKEN" ]; then
    # Pass the token and automatically answer 'n' to the prompt
    echo -e "$HUGGING_FACE_HUB_TOKEN\nn" | huggingface-cli login
else
    echo "HUGGING_FACE_HUB_TOKEN environment variable not set. Skipping login."
fi

# Wait a moment for the login to complete
sleep 2

# Start the Streamlit application
echo "Starting Streamlit application..."
streamlit run app_old.py --server.port=8501 --server.address=0.0.0.0