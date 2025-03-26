# Open Source RAG Multimodal LLM

This project combines multiple open-source AI models to create a conversational assistant with text and voice capabilities. It features:

- Text-based conversation with LLaMA models
- Speech-to-text using Whisper
- Text-to-speech using CSM 1B
- Streamlit web interface

## Features

- **Multiple LLM Options**: Choose between llama3.2:1b, llama3:8b, and gemma models
- **Voice Input**: Record audio that gets transcribed and sent to the LLM
- **Voice Output**: Automated text-to-speech for all assistant responses
- **Conversation History**: Maintains a chat history for context in responses
- **File Upload**: Upload files to discuss with the AI assistant
- **Adaptive TTS**: Handles long responses by breaking them into appropriate chunks

## Requirements

- Python 3.12
- [Ollama](https://ollama.com/) for running LLM models
- GPU recommended for faster processing

## Setup

### Using Docker (Recommended)

1. Build the Docker image:
   ```bash
   docker build -t rag-llm-app .
   ```

2. Run the Docker container:
   ```bash
   docker run -p 8501:8501 rag-llm-app
   ```

3. Access the application at http://localhost:8501

### Manual Setup

1. Install Python 3.12
2. Install Ollama following instructions at https://ollama.com/
3. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Download required models:
   ```bash
   ollama pull llama3.2:1b
   ollama pull llama3:8b
   ollama pull gemma
   ```
5. Run the application:
   ```bash
   streamlit run app.py
   ```

## Environment Variables

Create a `.env` file in the project root with the following variables (optional):

```
HUGGING_FACE_HUB_TOKEN=your_huggingface_token
```

## Project Structure

- `app.py`: Main application file with Streamlit UI
- `generator.py`: Text-to-speech model loading and inference (not shown in the provided code)
- `audio1.wav`, `audio2.wav`: Sample audio files for TTS context
- `requirements.txt`: Python dependencies
- `Dockerfile`: Docker configuration for containerization
- `temp_audio/`: Directory for storing generated audio files

## Usage

1. Open the application in your web browser
2. Choose your preferred LLM model in the sidebar
3. Adjust TTS settings if needed
4. Type messages in the chat input or record audio using the microphone
5. View responses in text and listen to the generated audio
6. Upload files to discuss with the AI assistant

## How It Works

1. **Text Input**: User messages are sent to the selected LLM via Ollama
2. **Voice Input**: 
   - Audio is recorded and saved to a temporary file
   - Whisper model transcribes the audio to text
   - Transcribed text is sent to the LLM
3. **Response Generation**:
   - LLM generates text responses
   - Text is processed by the CSM 1B TTS model
   - Long responses are split into chunks for better TTS quality
4. **Output**:
   - Text responses are displayed in the chat
   - Audio responses are played through the browser

## Customization

- Adjust the `max_chunk_length` parameter to control how text is split for TTS
- Modify the system prompt in the `call_llm` function to change the assistant's behavior
- Add more LLM models to the selection by updating the `models` list

## Notes

- The TTS model requires specific audio context files (audio1.wav and audio2.wav)
- Audio files are stored temporarily and cleared when the chat is reset
- For best performance, use a machine with a GPU
- The application uses Streamlit's built-in audio recording feature

## Troubleshooting

- If audio recording doesn't work, ensure your browser has permission to access the microphone
- If models fail to load, check that Ollama is running and the models are downloaded
- For TTS issues, verify that the audio context files exist in the correct location

