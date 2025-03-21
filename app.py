import streamlit as st
import ollama
import torch
import torchaudio
import base64
import os
from dataclasses import dataclass
from typing import List, Optional
from generator import load_csm_1b

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
def call_llm(prompt, model="llama3:8b", chat_history=[]):
    try:
        system_prompt = "You are a funny helpful ai assistant who gives the answers in a short and conversational way. DO not add any punctuation like : or ; in the output text"
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend([{"role": role, "content": content} for role, content in chat_history])
        messages.append({"role": "user", "content": prompt})
        response = ollama.chat(
            model=model,
            messages=messages
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

# Function to generate audio from text
def generate_audio(text, speaker_id=1):
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
    
    # Generate audio for the text
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

def main():
    st.set_page_config(page_title="LLM Chatbot using Ollama", layout="centered")
    st.title("🤖 Chatbot using Ollama LLaMA")
    
    st.sidebar.title("Settings")
    models = ["llama3.2:1b", "llama3:8b","gemma3:1b" ]
    model = st.sidebar.selectbox("Choose LLaMA Model", models)
    
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "audio_files" not in st.session_state:
        st.session_state.audio_files = {}
    
    # Display chat history
    for i, chat in enumerate(st.session_state.chat_history):
        role, content = chat
        if role == "user":
            st.chat_message("user").markdown(content)
        else:
            # For assistant messages, display message with audio button
            col1, col2 = st.columns([0.9, 0.1])
            with col1:
                message_container = st.chat_message("assistant")
                message_container.markdown(content)
                
                # Check if audio exists for this message
                if i in st.session_state.audio_files:
                    audio_file = st.session_state.audio_files[i]
                    audio_b64 = get_audio_base64(audio_file)
                    message_container.markdown(get_audio_player_html(audio_b64), unsafe_allow_html=True)
            
            with col2:
                # Copy button
                if st.button("📋", key=f"copy_{i}"):
                    st.session_state.clipboard_text = content
                    st.success("Copied to clipboard!")
                
                # Speaker button - generate audio if not already generated
                if st.button("🔊", key=f"speak_{i}"):
                    if i not in st.session_state.audio_files:
                        with st.spinner("Generating audio..."):
                            audio_file = generate_audio(content)
                            st.session_state.audio_files[i] = audio_file
                    
                    # Display audio player
                    audio_file = st.session_state.audio_files[i]
                    audio_b64 = get_audio_base64(audio_file)
                    st.markdown(get_audio_player_html(audio_b64), unsafe_allow_html=True)
    
    # Chat input
    prompt = st.chat_input("Ask something...")
    if prompt:
        # Display user message
        st.chat_message("user").markdown(prompt)
        st.session_state.chat_history.append(("user", prompt))
        
        # Get response from LLM
        response = call_llm(prompt, model, st.session_state.chat_history)
        
        # Display assistant message
        message_idx = len(st.session_state.chat_history)
        message_container = st.chat_message("assistant")
        message_container.markdown(response)
        
        # Generate audio for the response
        with st.spinner("Generating audio..."):
            audio_file = generate_audio(response)
            st.session_state.audio_files[message_idx] = audio_file
        
        # Display audio player
        audio_b64 = get_audio_base64(audio_file)
        message_container.markdown(get_audio_player_html(audio_b64), unsafe_allow_html=True)
        
        # Add to chat history
        st.session_state.chat_history.append(("assistant", response))
        
        # Force rerun to update UI
        st.rerun()

if __name__ == "__main__":
    main()