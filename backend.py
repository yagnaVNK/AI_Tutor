from flask import Flask, request, jsonify
from flask_cors import CORS
import os, io, base64, tempfile
import numpy as np
import whisper
import ollama
from generator import load_csm_1b
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
HF_TOKEN = os.environ.get('HF_TOKEN')
if HF_TOKEN:
    from huggingface_hub import login
    login(token=HF_TOKEN)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load models once
whisper_model = whisper.load_model("base")
tts_generator = load_csm_1b()

# System prompt for LLM
SYSTEM_PROMPT = (
    "You are a funny helpful ai assistant who gives the answers in a "
    "short and conversational way. DO not add any punctuation in the output text"
)

# Helper to call the LLM
def call_llm(prompt, model, chat_history):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += [{"role": role, "content": content} for role, content in chat_history]
    messages.append({"role": "user", "content": prompt})
    response = ollama.chat(model=model, messages=messages)
    return response["message"]["content"]

@app.route('/models', methods=['GET'])
def list_models():
    try:
        # ollama.list() returns ("models", [Model, ...])
        _, model_list = ollama.list()
        names = [m.model for m in model_list]
        return jsonify(names)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json or {}
    prompt = data.get("prompt", "")
    model = data.get("model", "gemma3:1b")
    history = data.get("chat_history", [])
    try:
        resp = call_llm(prompt, model, history)
        return jsonify({"response": resp})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    f = request.files["file"]
    tmp_path = f"{tempfile.mktemp()}.wav"
    f.save(tmp_path)
    try:
        result = whisper_model.transcribe(tmp_path)
    finally:
        os.remove(tmp_path)
    return jsonify({"text": result["text"]})

@app.route('/generate_audio', methods=['POST'])
def generate_audio():
    data = request.json or {}
    text = data.get("text", "")
    speaker = data.get("speaker_id", 1)
    try:
        audio_tensor = tts_generator.generate(
            text=text,
            speaker=speaker,
            context=[],
            max_audio_length_ms=25_000
        )
        arr = audio_tensor.cpu().numpy()
        pcm = (arr * 32767).astype(np.int16)
        buf = io.BytesIO()
        import wave
        wf = wave.open(buf, "wb")
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(tts_generator.sample_rate)
        wf.writeframes(pcm.tobytes())
        wf.close()
        audio_b64 = base64.b64encode(buf.getvalue()).decode()
        return jsonify({"audio": audio_b64})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Run on port 5000 in debug mode
    app.run(host="0.0.0.0", port=5000, debug=True)
