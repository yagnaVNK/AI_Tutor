import os
import streamlit as st
import streamlit.components.v1 as components
import uuid

def mic_recorder(save_path="recordings"):
    """
    Creates a microphone recorder component for Streamlit with device selection.
    
    Args:
        save_path (str, optional): Directory to save recordings. Defaults to "recordings".
    
    Returns:
        str or None: Path to the saved audio file if recording is complete, None otherwise
    """
    # Create save directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Generate a unique ID for this instance
    component_id = str(uuid.uuid4())
    
    # Define the HTML/JS component for microphone recording
    mic_html = f"""
    <div id="mic-recorder-{component_id}">
        <style>
            .mic-button {{
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 15px 32px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 4px 2px;
                cursor: pointer;
                border-radius: 8px;
                transition: background-color 0.3s;
            }}
            .mic-button.recording {{
                background-color: #f44336;
            }}
            .timer {{
                font-size: 18px;
                margin-top: 10px;
            }}
            .audio-container {{
                margin-top: 20px;
            }}
            #device-selector-{component_id} {{
                margin: 10px 0;
                padding: 8px;
                width: 100%;
                border-radius: 4px;
                border: 1px solid #ccc;
            }}
            .device-label {{
                margin-bottom: 5px;
                font-weight: bold;
            }}
            .control-panel {{
                margin-bottom: 15px;
                padding: 15px;
                border: 1px solid #ddd;
                border-radius: 8px;
                background-color: #f9f9f9;
            }}
        </style>

        <div class="control-panel">
            <div class="device-label">Select Microphone:</div>
            <select id="device-selector-{component_id}" class="form-control">
                <option value="">Loading devices...</option>
            </select>
            
            <div style="margin-top: 15px;">
                <button id="micButton-{component_id}" class="mic-button">Start Recording</button>
                <button id="refreshDevices-{component_id}" style="margin-left: 10px; padding: 15px 20px; background-color: #2196F3; color: white; border: none; border-radius: 8px; cursor: pointer;">Refresh Devices</button>
            </div>
        </div>
        
        <div id="timer-{component_id}" class="timer">00:00</div>
        <div id="status-{component_id}"></div>
        <div id="audio-container-{component_id}" class="audio-container"></div>
        <a id="download-link-{component_id}" style="display:none"></a>

        <script>
            (function() {{
                // Define variables for recording
                let mediaRecorder;
                let audioChunks = [];
                let isRecording = false;
                let stream;
                let startTime;
                let timerInterval;
                let audioBlob;
                let audioUrl;
                
                // Get DOM elements
                const deviceSelector = document.getElementById('device-selector-{component_id}');
                const micButton = document.getElementById('micButton-{component_id}');
                const refreshButton = document.getElementById('refreshDevices-{component_id}');
                const timerElement = document.getElementById('timer-{component_id}');
                const statusElement = document.getElementById('status-{component_id}');
                const audioContainer = document.getElementById('audio-container-{component_id}');
                const downloadLink = document.getElementById('download-link-{component_id}');
                
                // Load available audio input devices
                async function loadAudioDevices() {{
                    try {{
                        deviceSelector.innerHTML = '<option value="">Loading devices...</option>';
                        
                        // Request device permissions to get access to device list
                        await navigator.mediaDevices.getUserMedia({{ audio: true }});
                        
                        // Get all media devices
                        const devices = await navigator.mediaDevices.enumerateDevices();
                        
                        // Filter for audio input devices only
                        const audioInputDevices = devices.filter(device => device.kind === 'audioinput');
                        
                        // Clear the select options
                        deviceSelector.innerHTML = '';
                        
                        // Add each audio input device to the select
                        audioInputDevices.forEach(device => {{
                            const option = document.createElement('option');
                            option.value = device.deviceId;
                            // Use a fallback name if label is not available
                            if (device.label) {{
                                option.text = device.label;
                            }} else {{
                                option.text = "Microphone " + device.deviceId.substring(0, 5) + "...";
                            }}
                            deviceSelector.appendChild(option);
                        }});
                        
                        if (audioInputDevices.length === 0) {{
                            deviceSelector.innerHTML = '<option value="">No microphones found</option>';
                            statusElement.textContent = "Error: No microphones detected";
                        }}
                    }} catch (err) {{
                        console.error("Error accessing media devices:", err);
                        deviceSelector.innerHTML = '<option value="">Error loading devices</option>';
                        statusElement.textContent = "Error: " + err.message;
                    }}
                }}
                
                // Update timer display
                function updateTimer() {{
                    const currentTime = new Date();
                    const elapsedTime = new Date(currentTime - startTime);
                    const minutes = elapsedTime.getUTCMinutes().toString().padStart(2, '0');
                    const seconds = elapsedTime.getUTCSeconds().toString().padStart(2, '0');
                    timerElement.textContent = `${{minutes}}:${{seconds}}`;
                }}
                
                // Start recording
                async function startRecording() {{
                    try {{
                        const selectedDeviceId = deviceSelector.value;
                        
                        if (!selectedDeviceId) {{
                            statusElement.textContent = "Error: Please select a microphone";
                            return;
                        }}
                        
                        // Request microphone permission with specific device
                        const constraints = {{
                            audio: {{
                                deviceId: {{ exact: selectedDeviceId }},
                                echoCancellation: true,
                                noiseSuppression: true,
                                autoGainControl: true
                            }}
                        }};
                        
                        stream = await navigator.mediaDevices.getUserMedia(constraints);
                        
                        // Set up media recorder
                        const options = {{ mimeType: 'audio/webm' }};
                        
                        try {{
                            mediaRecorder = new MediaRecorder(stream, options);
                        }} catch (e) {{
                            // Fallback if the preferred format is not supported
                            mediaRecorder = new MediaRecorder(stream);
                        }}
                        
                        audioChunks = [];
                        
                        // Collect audio chunks when data is available
                        mediaRecorder.ondataavailable = event => {{
                            audioChunks.push(event.data);
                        }};
                        
                        // Handle recording stopped
                        mediaRecorder.onstop = () => {{
                            // Create audio blob
                            audioBlob = new Blob(audioChunks);
                            audioUrl = URL.createObjectURL(audioBlob);
                            
                            // Create audio player
                            const audioElement = document.createElement('audio');
                            audioElement.src = audioUrl;
                            audioElement.controls = true;
                            audioElement.style.width = '100%';
                            
                            // Clear and add the audio element
                            audioContainer.innerHTML = '';
                            audioContainer.appendChild(audioElement);
                            
                            // Create download link
                            downloadLink.href = audioUrl;
                            downloadLink.download = "recording.webm";
                            downloadLink.textContent = "Download Recording";
                            downloadLink.style.display = "block";
                            downloadLink.className = "mic-button";
                            downloadLink.style.marginTop = "10px";
                            downloadLink.style.textDecoration = "none";
                            downloadLink.style.textAlign = "center";
                            audioContainer.appendChild(downloadLink);
                            
                            statusElement.textContent = "Recording saved locally. Click Download to save it.";
                        }};
                        
                        // Start recording
                        mediaRecorder.start(1000); // Collect data every second
                        isRecording = true;
                        micButton.textContent = "Stop Recording";
                        micButton.classList.add("recording");
                        
                        // Get the selected device name
                        const selectedOption = deviceSelector.options[deviceSelector.selectedIndex];
                        const deviceName = selectedOption ? selectedOption.text : "selected microphone";
                        
                        statusElement.textContent = "Recording... (using " + deviceName + ")";
                        
                        // Disable device selector while recording
                        deviceSelector.disabled = true;
                        refreshButton.disabled = true;
                        
                        // Start timer
                        startTime = new Date();
                        timerInterval = setInterval(updateTimer, 1000);
                        
                    }} catch (err) {{
                        console.error("Error accessing microphone:", err);
                        statusElement.textContent = "Error: " + err.message;
                    }}
                }}
                
                // Stop recording
                function stopRecording() {{
                    if (mediaRecorder && isRecording) {{
                        mediaRecorder.stop();
                        stream.getTracks().forEach(track => track.stop());
                        clearInterval(timerInterval);
                        isRecording = false;
                        micButton.textContent = "Start Recording";
                        micButton.classList.remove("recording");
                        
                        // Re-enable device selector
                        deviceSelector.disabled = false;
                        refreshButton.disabled = false;
                    }}
                }}
                
                // Toggle recording when button is clicked
                micButton.addEventListener('click', () => {{
                    if (isRecording) {{
                        stopRecording();
                    }} else {{
                        startRecording();
                    }}
                }});
                
                // Refresh device list
                refreshButton.addEventListener('click', () => {{
                    statusElement.textContent = "Refreshing device list...";
                    loadAudioDevices();
                }});
                
                // Initialize device list
                loadAudioDevices();
            }})();
        </script>
    </div>
    """
    
    # Display the component
    components.html(mic_html, height=400)
    
    st.info("Select your preferred microphone from the dropdown menu, then click 'Start Recording'.")
    
    return None

# Example usage
if __name__ == "__main__":
    st.title("Microphone Recorder with Device Selection")
    
    st.write("Select your microphone and record audio")
    
    # Display the microphone recorder component
    mic_recorder(save_path="audio_recordings")