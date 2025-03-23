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

# Start the Streamlit application
echo "Starting Streamlit application..."
streamlit run app.py --server.port=8501 