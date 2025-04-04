import streamlit as st
import ollama
import torch
import torchaudio
import base64
import os
import re
import numpy as np
import whisper
import wave
import tempfile
from dataclasses import dataclass
from typing import List, Optional
from generator import load_csm_1b
import os
from dotenv import load_dotenv
from huggingface_hub import login

# Load environment variables from .env file
load_dotenv()
HF_TOKEN = os.environ.get('HUGGING_FACE_HUB_TOKEN')
# Make sure token exists before trying to use it
if HF_TOKEN:
    login(token=HF_TOKEN)
else:
    # Handle the case when no token is available
    print("Warning: No Hugging Face token found")


# Define the Segment class for TTS
@dataclass
class Segment:
    text: str
    speaker: int
    audio: Optional[torch.Tensor] = None


def load_prompt_audio(audio_path: str, target_sample_rate: int) -> torch.Tensor:
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = audio_tensor.squeeze(0)
    # Resample is lazy so we can always call it
    audio_tensor = torchaudio.functional.resample(
        audio_tensor, orig_freq=sample_rate, new_freq=target_sample_rate
    )
    return audio_tensor


def prepare_prompt(text: str, speaker: int, audio_path: str, sample_rate: int) -> Segment:
    audio_tensor = load_prompt_audio(audio_path, sample_rate)
    return Segment(text=text, speaker=speaker, audio=audio_tensor)


# Function to call LLM
def call_llm(prompt, model="gemma3:1b", chat_history=[]):
    try:
        system_prompt = "You are a funny helpful ai assistant who gives the answers in a short and conversational way. DO not add any punctuation in the output text"
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend([{"role": role, "content": content} for role, content in chat_history])
        messages.append({"role": "user", "content": prompt})
        response = ollama.chat(
            model=model,
            messages=messages,
        )
        return response["message"]["content"]
    except Exception as e:
        return f"Error: {e}"


# Function to get available models
def get_available_models():
    try:
        models = ollama.list_models()
        return [model['name'] for model in models]
    except Exception as e:
        return [f"Error fetching models: {e}"]


# Function to split text into reasonable chunks for TTS
def split_text_into_chunks(text, max_chunk_length=200):
    # Split by sentences or natural pauses for better TTS results
    # Try to break at periods, then commas, then spaces
    chunks = []
    
    # First split by sentences (periods, question marks, exclamation points)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    current_chunk = ""
    for sentence in sentences:
        # If adding this sentence would exceed max length, store current chunk and start new one
        if len(current_chunk) + len(sentence) > max_chunk_length and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
    
    # Add the last chunk if there's anything left
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # If any chunk is still too long, split it by commas
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > max_chunk_length:
            comma_parts = chunk.split(", ")
            sub_chunk = ""
            for part in comma_parts:
                if len(sub_chunk) + len(part) > max_chunk_length and sub_chunk:
                    final_chunks.append(sub_chunk.strip())
                    sub_chunk = part
                else:
                    if sub_chunk:
                        sub_chunk += ", " + part
                    else:
                        sub_chunk = part
            if sub_chunk:
                final_chunks.append(sub_chunk.strip())
        else:
            final_chunks.append(chunk)
    
    # If any chunk is still too long, just hard split it
    result_chunks = []
    for chunk in final_chunks:
        if len(chunk) > max_chunk_length:
            # Hard split by words
            words = chunk.split()
            sub_chunk = ""
            for word in words:
                if len(sub_chunk) + len(word) > max_chunk_length and sub_chunk:
                    result_chunks.append(sub_chunk.strip())
                    sub_chunk = word
                else:
                    if sub_chunk:
                        sub_chunk += " " + word
                    else:
                        sub_chunk = word
            if sub_chunk:
                result_chunks.append(sub_chunk.strip())
        else:
            result_chunks.append(chunk)
    
    return result_chunks


# Function to generate audio from text
def generate_audio(text, speaker_id=1, max_chunk_length=200):
    # Initialize generator if not already in session state
    if 'tts_generator' not in st.session_state:
        st.session_state.tts_generator = load_csm_1b()
    
    generator = st.session_state.tts_generator
    
    # Create context for generation
    context1 = """Think of it like this imagine you're trying to describe a bunch of different objects to someone who's never seen them before you've got to use words that are unique to each object so they can understand what you mean. In computer vision we do something similar with images these are called feature vectors or descriptors because they help computers match an image to the ones it's seen many times before"""
    context2 = """The codebook is like a list of these feature vectors it's used to build the embeddings which are basically vector representations of each image in the codebook, Embeddings are how we map objects to a high dimensional space where objects that are similar are close together and objects that are different are far apart think of it like a big graph with images as nodes and edges between nodes, when you look at the same object multiple times they'll form a connection to each other."""
    
    # Prepare prompt segments
    prompt_a = prepare_prompt(context1, 1, "audio1.wav", generator.sample_rate)
    prompt_b = prepare_prompt(context2, 1, "audio2.wav", generator.sample_rate)
    prompt_segments = [prompt_a, prompt_b]
    
    # Check if text is too long and needs to be split
    text_chunks = split_text_into_chunks(text, max_chunk_length)
    
    if len(text_chunks) == 1:
        # If we only have one chunk, generate normally
        audio_tensor = generator.generate(
            text=text,
            speaker=speaker_id,
            context=prompt_segments,
            max_audio_length_ms=25_000
        )
        
        # Save audio to file in a temp directory
        os.makedirs("temp_audio", exist_ok=True)
        audio_file_path = f"temp_audio/response_{hash(text)}.wav"
        torchaudio.save(
            audio_file_path,
            audio_tensor.unsqueeze(0).cpu(),
            generator.sample_rate
        )
        
        return audio_file_path
    else:
        # Process each chunk and concatenate the results
        audio_tensors = []
        
        # Show progress bar when processing multiple chunks
        progress_text = f"Generating audio for {len(text_chunks)} segments..."
        progress_bar = st.progress(0)
        
        for i, chunk in enumerate(text_chunks):
            # Update progress
            progress_bar.progress((i / len(text_chunks)))
            st.text(f"Processing chunk {i+1}/{len(text_chunks)}: {chunk[:30]}...")
            
            # Generate audio for this chunk
            chunk_audio = generator.generate(
                text=chunk,
                speaker=speaker_id,
                context=prompt_segments,
                max_audio_length_ms=25_000
            )
            audio_tensors.append(chunk_audio)
        
        # Complete the progress
        progress_bar.progress(1.0)
        st.text("Merging audio segments...")
        
        # Concatenate all audio tensors
        full_audio = torch.cat(audio_tensors, dim=0)
        
        # Save the complete audio file
        os.makedirs("temp_audio", exist_ok=True)
        audio_file_path = f"temp_audio/response_{hash(text)}.wav"
        torchaudio.save(
            audio_file_path,
            full_audio.unsqueeze(0).cpu(),
            generator.sample_rate
        )
        
        return audio_file_path


# Function to get base64 encoded audio for HTML audio player
def get_audio_base64(audio_file):
    with open(audio_file, "rb") as f:
        audio_bytes = f.read()
    audio_b64 = base64.b64encode(audio_bytes).decode()
    return audio_b64


# Function to create an HTML audio player
def get_audio_player_html(audio_b64):
    audio_html = f"""
    <audio controls style="height:50px; width:100%">
        <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav">
        Your browser does not support the audio element.
    </audio>
    """
    return audio_html


# Function to save audio bytes to a file
def save_audio_to_file(uploaded_file, filename):
    """Saves uploaded audio file to a WAV file."""
    try:
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Read bytes from the uploaded file
        audio_bytes = uploaded_file.read()
        
        # Save the audio bytes to a file
        with open(filename, 'wb') as f:
            f.write(audio_bytes)
        
        return True
    except Exception as e:
        st.error(f"Error saving audio: {e}")
        return False


# Function to transcribe audio using Whisper
def transcribe_audio(audio_path, whisper_model):
    """Transcribes the audio using Whisper."""
    try:
        with st.spinner("Transcribing your speech..."):
            result = whisper_model.transcribe(audio_path)
            return result['text']
    except Exception as e:
        st.error(f"Error during transcription: {e}")
        return None


def main():
    st.set_page_config(page_title="RAG LLM", layout="wide")
    st.title("Open Source RAG multimodel LLM")
    
    # Initialize Whisper model in session state if not already there
    if 'whisper_model' not in st.session_state:
        with st.spinner("Loading Whisper model..."):
            st.session_state.whisper_model = whisper.load_model("base")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "audio_files" not in st.session_state:
        st.session_state.audio_files = {}
    
    # Sidebar settings
    with st.sidebar:
        st.header("Settings")
        
        # LLM settings
        st.subheader("LLM Settings")
        models = ["gemma3:1b", "llama3.2:1b"]
        model = st.selectbox("Choose LLaMA Model", models)
        
        # TTS settings
        st.subheader("Text-to-Speech Settings")
        max_chunk_length = st.slider("Max Chunk Length", 100, 500, 200, 
                                  help="Maximum number of characters per audio chunk. Longer texts will be split into chunks of this size.")
        
        # File uploader
        st.header("Upload Files")
        uploaded_file = st.file_uploader(
            "Upload a file to analyze",
            type=None,  # Accept all file types
            accept_multiple_files=False,
            help="Upload any file that you want to analyze or discuss"
        )
        if uploaded_file is not None:
            file_details = {"Filename": uploaded_file.name, "Filetype": uploaded_file.type, "Size": f"{uploaded_file.size / 1024:.2f} KB"}
            st.write("File Details:")
            st.json(file_details)
            # Add the file upload as a message
            if st.button("Add file to chat"):
                st.session_state.messages.append({"role": "user", "content": f"[Uploaded file: {uploaded_file.name}]"})
                st.rerun()
    
    # Display chat messages
    st.subheader("Chat")
    chat_container = st.container()
    with chat_container:
        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
                # If it's an assistant message, add audio playback option
                if message["role"] == "assistant" and i in st.session_state.audio_files:
                    audio_file = st.session_state.audio_files[i]
                    audio_b64 = get_audio_base64(audio_file)
                    st.markdown(get_audio_player_html(audio_b64), unsafe_allow_html=True)
    
    # Input section - putting audio and text input close together
    st.subheader("Input")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Audio input
        uploaded_audio = st.audio_input("Record audio")
        
        if uploaded_audio is not None:
            # Create a temporary file to store the recording
            temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            
            # Save the uploaded audio file to disk
            with st.spinner("Processing your recording..."):
                if save_audio_to_file(uploaded_audio, temp_audio_file):
                    # Transcribe the recorded audio
                    transcribed_text = transcribe_audio(temp_audio_file, st.session_state.whisper_model)
                    
                    if transcribed_text:
                        st.success(f"Transcribed: {transcribed_text}")
                        
                        # Add transcribed text as user message
                        if st.button("Add transcription to chat"):
                            st.session_state.messages.append({"role": "user", "content": transcribed_text})
                            
                            # Get response from LLM
                            with st.spinner("Getting response..."):
                                chat_history = [(msg["role"], msg["content"]) for msg in st.session_state.messages[:-1]]
                                response = call_llm(transcribed_text, model, chat_history)
                            
                            # Add assistant response to chat
                            message_idx = len(st.session_state.messages)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                            
                            # Generate audio for the response
                            with st.spinner("Generating audio..."):
                                audio_file = generate_audio(response, speaker_id=1, max_chunk_length=max_chunk_length)
                                st.session_state.audio_files[message_idx] = audio_file
                            
                            st.rerun()
    
    with col2:
        # Text input for chat
        prompt = st.chat_input("Type your message here...")
        
        if prompt:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Get response from LLM
            with st.spinner("Getting response..."):
                chat_history = [(msg["role"], msg["content"]) for msg in st.session_state.messages[:-1]]
                response = call_llm(prompt, model, chat_history)
            
            # Add assistant response to chat
            message_idx = len(st.session_state.messages)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Generate audio for the response
            with st.spinner("Generating audio..."):
                audio_file = generate_audio(response, speaker_id=1, max_chunk_length=max_chunk_length)
                st.session_state.audio_files[message_idx] = audio_file
            
            # Rerun to update the UI
            st.rerun()
    
    # Optional: Clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.audio_files = {}
        # Clean up temp audio files
        if os.path.exists("temp_audio"):
            for file in os.listdir("temp_audio"):
                os.remove(os.path.join("temp_audio", file))
        st.rerun()


if __name__ == "__main__":
    main()