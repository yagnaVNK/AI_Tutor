import streamlit as st
import streamlit.components.v1 as components
import os
import base64
import uuid
from typing import Optional, Tuple


def audio_recorder(
    recording_color: str = "#f44336",
    neutral_color: str = "#4CAF50",
    height: int = 190
) -> Optional[bytes]:
    """
    Creates an audio recorder component for Streamlit.
    
    Parameters:
    ----------
    recording_color : str, optional
        CSS color for the recording button when active
    neutral_color : str, optional
        CSS color for the recording button when inactive
    height : int, optional
        Height of the component in pixels
    
    Returns:
    -------
    Optional[bytes]:
        The recorded audio as bytes if recording was submitted, None otherwise
    """
    # Create HTML for the audio recorder
    html = f"""
    <div style="padding: 20px; border: 1px solid #ccc; border-radius: 10px;">
        <style>
            button {{
                padding: 12px 24px;
                border-radius: 5px;
                border: none;
                cursor: pointer;
                margin: 5px;
                font-size: 16px;
            }}
            .record-btn {{
                background-color: {neutral_color};
                color: white;
            }}
            .record-btn.recording {{
                background-color: {recording_color};
            }}
            .submit-btn {{
                background-color: #2196F3;
                color: white;
            }}
        </style>
        
        <button id="recordBtn" class="record-btn">Start Recording</button>
        <div id="timer" style="margin: 10px 0; font-size: 18px; font-weight: bold;">00:00</div>
        <div id="status" style="margin: 10px 0;">Ready to record</div>
        <div id="audioContainer"></div>
        <div id="submitContainer"></div>
        
        <script>
            (function() {{
                // Variables
                const recordBtn = document.getElementById('recordBtn');
                const status = document.getElementById('status');
                const timer = document.getElementById('timer');
                const audioContainer = document.getElementById('audioContainer');
                const submitContainer = document.getElementById('submitContainer');
                
                let mediaRecorder;
                let audioChunks = [];
                let audioBlob;
                let isRecording = false;
                let startTime;
                let timerInterval;
                
                // Update timer
                function updateTimer() {{
                    const elapsed = Date.now() - startTime;
                    const seconds = Math.floor((elapsed / 1000) % 60).toString().padStart(2, '0');
                    const minutes = Math.floor((elapsed / 1000 / 60) % 60).toString().padStart(2, '0');
                    timer.textContent = `${{minutes}}:${{seconds}}`;
                }}
                
                // Function to start recording
                async function startRecording() {{
                    try {{
                        // Get audio stream with default microphone
                        const stream = await navigator.mediaDevices.getUserMedia({{
                            audio: {{
                                echoCancellation: true,
                                noiseSuppression: true,
                                autoGainControl: true
                            }}
                        }});
                        
                        // Create recorder
                        mediaRecorder = new MediaRecorder(stream);
                        audioChunks = [];
                        
                        // Handle data
                        mediaRecorder.ondataavailable = (e) => {{
                            audioChunks.push(e.data);
                        }};
                        
                        // Handle recording stopped
                        mediaRecorder.onstop = () => {{
                            // Create audio element
                            audioBlob = new Blob(audioChunks, {{ type: 'audio/webm' }});
                            const audioUrl = URL.createObjectURL(audioBlob);
                            
                            // Create audio player
                            audioContainer.innerHTML = '';
                            const audio = document.createElement('audio');
                            audio.controls = true;
                            audio.src = audioUrl;
                            audio.style.width = '100%';
                            audioContainer.appendChild(audio);
                            
                            // Create submit button
                            submitContainer.innerHTML = '';
                            const submitBtn = document.createElement('button');
                            submitBtn.className = 'submit-btn';
                            submitBtn.textContent = 'Submit Recording';
                            submitBtn.onclick = submitRecording;
                            submitContainer.appendChild(submitBtn);
                            
                            status.textContent = 'Recording complete';
                        }};
                        
                        // Start recording
                        mediaRecorder.start(1000);
                        isRecording = true;
                        recordBtn.textContent = 'Stop Recording';
                        recordBtn.classList.add('recording');
                        status.textContent = 'Recording... (Click Stop when done)';
                        
                        // Start timer
                        startTime = Date.now();
                        timerInterval = setInterval(updateTimer, 1000);
                        
                    }} catch (err) {{
                        status.textContent = 'Error: ' + err.message;
                        console.error('Recording error:', err);
                    }}
                }}
                
                // Function to stop recording
                function stopRecording() {{
                    if (mediaRecorder && isRecording) {{
                        mediaRecorder.stop();
                        mediaRecorder.stream.getTracks().forEach(track => track.stop());
                        isRecording = false;
                        recordBtn.textContent = 'Start Recording';
                        recordBtn.classList.remove('recording');
                        clearInterval(timerInterval);
                    }}
                }}
                
                // Function to submit recording
                function submitRecording() {{
                    if (!audioBlob) return;
                    
                    // Convert to base64
                    const reader = new FileReader();
                    reader.readAsDataURL(audioBlob);
                    reader.onloadend = function() {{
                        const base64data = reader.result.split(',')[1];
                        
                        // Try multiple methods to send data back to Streamlit
                        try {{
                            if (window.Streamlit) {{
                                window.Streamlit.setComponentValue(base64data);
                            }} else {{
                                window.parent.postMessage({{
                                    type: 'streamlit:setComponentValue',
                                    value: base64data
                                }}, '*');
                            }}
                            
                            status.textContent = 'Recording submitted successfully!';
                            submitContainer.innerHTML = '';
                            
                        }} catch (err) {{
                            status.textContent = 'Error submitting: ' + err.message;
                            console.error('Submit error:', err);
                        }}
                    }};
                }}
                
                // Set up record button
                recordBtn.addEventListener('click', () => {{
                    if (isRecording) {{
                        stopRecording();
                    }} else {{
                        startRecording();
                    }}
                }});
            }})();
        </script>
    </div>
    """
    
    # Display the HTML component - this returns a value from the JavaScript
    component_value = components.html(html, height=height)
    
    # Process the returned value
    if component_value is not None and isinstance(component_value, str):
        try:
            # Decode base64 to bytes
            audio_bytes = base64.b64decode(component_value)
            return audio_bytes
        except Exception as e:
            st.error(f"Error processing audio: {str(e)}")
    
    return None


# Example usage
if __name__ == "__main__":
    st.set_page_config(page_title="Audio Recorder Component", layout="wide")
    st.title("Audio Recorder Component Example")
    
    # Create a directory for recordings
    save_dir = "audio_recordings"
    os.makedirs(save_dir, exist_ok=True)
    
    # Use the audio recorder component
    st.markdown("### Recording with Default Microphone")
    st.markdown("This app uses your system's default microphone. Simply click 'Start Recording' to begin.")
    
    # Get audio data from component
    audio_data = audio_recorder()
    
    # Process the returned value
    if audio_data:
        # Generate a unique ID for this recording
        recording_id = str(uuid.uuid4())
        
        # Save to file
        filename = os.path.join(save_dir, f"recording_{recording_id}.webm")
        with open(filename, "wb") as f:
            f.write(audio_data)
        
        # Display success and audio player
        st.success("✅ Recording successful!")
        st.audio(audio_data, format="audio/webm")
        
        # Download button
        st.download_button(
            label="Download Recording",
            data=audio_data,
            file_name=f"recording_{recording_id}.webm",
            mime="audio/webm"
        )
        
        # Display additional information
        st.info(f"Recording saved to {filename}")
        st.code(f"Audio data length: {len(audio_data)} bytes")