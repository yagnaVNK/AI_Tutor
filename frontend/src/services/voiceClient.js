import { VOICE_WS_URL } from "../config";
import { getIdToken } from "./firebase";

export class VoiceClient {
  constructor({ conversationId = null, handlers = {} } = {}) {
    this.conversationId = conversationId;
    this.handlers = handlers;
    this.ws = null;
    this.mediaStream = null;
    this.recorder = null;
    this.audioCtx = null;
    this.playbackTime = 0;
    this.sampleRate = 24000;
    this.connected = false;
    this.audioSources = []; // track active audio sources for interruption
  }

  async connect() {
    const token = await getIdToken(true);
    if (!token) throw new Error("Not signed in");
    const url = new URL(VOICE_WS_URL);
    url.searchParams.set("token", token);
    if (this.conversationId) {
      url.searchParams.set("conversation_id", this.conversationId);
    }

    this.ws = new WebSocket(url.toString());
    this.ws.binaryType = "arraybuffer";

    await new Promise((resolve, reject) => {
      this.ws.onopen = () => {
        this.connected = true;
        resolve();
      };
      this.ws.onerror = (e) => reject(e);
    });

    this.ws.onmessage = (event) => this._onMessage(event);
    this.ws.onclose = () => {
      this.connected = false;
      this._emit("close");
    };
  }

  disconnect() {
    this.stopRecording();
    this.stopPlayback();
    if (this.ws) {
      try { this.ws.close(); } catch (_) {}
      this.ws = null;
    }
    if (this.audioCtx) {
      try { this.audioCtx.close(); } catch (_) {}
      this.audioCtx = null;
    }
  }

  async startRecording() {
    if (!this.connected) throw new Error("WebSocket not connected");
    this.mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
    const mime = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
      ? "audio/webm;codecs=opus"
      : "audio/webm";

    this._send({ type: "start_audio", mime, language: "en" });

    this.recorder = new MediaRecorder(this.mediaStream, { mimeType: mime });
    this._recordingStart = performance.now();
    this._chunkCount = 0;
    this.recorder.ondataavailable = async (e) => {
      if (e.data && e.data.size > 0 && this.ws && this.connected) {
        const buf = await e.data.arrayBuffer();
        this.ws.send(buf);
        this._chunkCount += 1;
      }
    };
    this.recorder.start(250);
  }

  // Stops the recorder and resolves once the final dataavailable chunk has
  // been delivered, then sends end_audio. We also enforce a short minimum
  // recording window so the user can't release the mic before MediaRecorder
  // has produced a single 250 ms chunk (which is what causes "no audio").
  async stopRecording() {
    const recorder = this.recorder;
    const stream = this.mediaStream;
    this.recorder = null;
    this.mediaStream = null;

    if (recorder && recorder.state !== "inactive") {
      const minMs = 400;
      const elapsed = performance.now() - (this._recordingStart || 0);
      if (elapsed < minMs) {
        await new Promise((r) => setTimeout(r, minMs - elapsed));
      }

      await new Promise((resolve) => {
        const onStop = () => resolve();
        recorder.addEventListener("stop", onStop, { once: true });
        try { recorder.stop(); } catch (_) { resolve(); }
      });
    }

    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
    }

    if (this.connected) {
      if ((this._chunkCount || 0) > 0) {
        this._send({ type: "end_audio" });
      } else {
        this._emit("error", "no audio captured");
      }
    }
    this._chunkCount = 0;
  }

  stopPlayback() {
    this.audioSources.forEach((src) => {
      try {
        src.stop();
      } catch (e) {}
    });
    this.audioSources = [];
    if (this.audioCtx) {
      this.playbackTime = this.audioCtx.currentTime + 0.05;
    }
  }

  sendText(text) {
    this._send({ type: "text", text });
  }

  setSystemPrompt(prompt) {
    this._send({ type: "set_system_prompt", prompt });
  }

  // -------------------------------------------------------------------
  // Internals
  // -------------------------------------------------------------------
  _send(payload) {
    if (this.ws && this.connected) this.ws.send(JSON.stringify(payload));
  }

  _emit(name, data) {
    const fn = this.handlers[name];
    if (fn) fn(data);
  }

  _onMessage(event) {
    if (typeof event.data === "string") {
      let payload;
      try { payload = JSON.parse(event.data); } catch (_) { return; }
      switch (payload.type) {
        case "ready":
          this.conversationId = payload.conversation_id;
          this._emit("ready", payload);
          break;
        case "transcript":
          this._emit("transcript", payload.text);
          break;
        case "assistant_chunk":
          this._emit("assistantChunk", payload.text);
          break;
        case "assistant_done":
          this._emit("assistantDone", payload.text);
          break;
        case "tts_meta":
          this._initAudio(payload.sample_rate || 24000);
          break;
        case "interrupt":
          this.stopPlayback();
          this._emit("interrupt", payload);
          break;
        case "error":
          this._emit("error", payload.detail);
          break;
        default:
          break;
      }
      return;
    }
    this._playPcmChunk(event.data);
  }

  _initAudio(sampleRate) {
    this.sampleRate = sampleRate;
    if (!this.audioCtx) {
      const Ctx = window.AudioContext || window.webkitAudioContext;
      this.audioCtx = new Ctx({ sampleRate });
      this.playbackTime = this.audioCtx.currentTime + 0.05;
    }
  }

  _playPcmChunk(arrayBuffer) {
    if (!this.audioCtx) this._initAudio(this.sampleRate);
    // Defensive: Int16Array requires an even byte length. Drop a stray
    // trailing byte rather than throwing — any misalignment is an upstream
    // bug, but losing one sample is preferable to crashing playback.
    let buf = arrayBuffer;
    if (buf.byteLength % 2) {
      buf = arrayBuffer.slice(0, arrayBuffer.byteLength - 1);
    }
    const i16 = new Int16Array(buf);
    if (i16.length === 0) return;
    const f32 = new Float32Array(i16.length);
    for (let i = 0; i < i16.length; i += 1) f32[i] = i16[i] / 0x8000;

    const buffer = this.audioCtx.createBuffer(1, f32.length, this.sampleRate);
    buffer.copyToChannel(f32, 0);

    const src = this.audioCtx.createBufferSource();
    src.buffer = buffer;
    src.connect(this.audioCtx.destination);
    
    // Keep track so we can stop it if interrupted
    this.audioSources.push(src);
    src.onended = () => {
      this.audioSources = this.audioSources.filter((s) => s !== src);
    };

    const now = this.audioCtx.currentTime;
    if (this.playbackTime < now) this.playbackTime = now + 0.02;
    src.start(this.playbackTime);
    this.playbackTime += buffer.duration;
  }
}
