import streamlit as st
from agent.agent_setup import setup_agent
import json
from pathlib import Path
import pandas as pd
import requests
import os
from pydub import AudioSegment
from pydub.playback import play
import tempfile
import pyttsx3
import numpy as np

import sounddevice as sd
import scipy.io.wavfile
import asyncio



# --- Load schema ---
base_path = Path(__file__).resolve().parent
schema_path = base_path / 'data' / 'database_schema.json'
with schema_path.open('r', encoding='utf-8') as f:
    database_schema = json.load(f)

# --- Setup agent ---
agent_executor, memory = setup_agent(database_schema)

# --- Export Conversation Function ---
def export_conversation():
    conversation_data = []
    name_map = {"User": "You", "Agent": "Clarix"}

    for i, (speaker, msg) in enumerate(st.session_state.chat_history):
        display_name = name_map.get(speaker, speaker)
        if speaker == "User":
            conversation_data.append({
                "speaker": speaker,
                "message": msg,
                "feedback": "NA"
            })
        elif speaker == "Agent":
            # Find corresponding feedback
            feedback_entry = next(
                (entry for entry in st.session_state.feedback_log if entry["agent_response"] == msg),
                None
            )
            feedback = feedback_entry["feedback"] if feedback_entry else "No Feedback"
            conversation_data.append({
                "speaker": speaker,
                "message": msg,
                "feedback": feedback
            })

    df = pd.DataFrame(conversation_data)
    return df

# --- Helper Functions ---
def record_audio(duration=5, fs=16000, silence_threshold=500):
    st.markdown(f"🎙️ Listening for {duration} seconds...")

    try:
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()

        if audio is None or audio.size == 0:
            return "there is no voice detected"

        # Flatten and check average amplitude
        audio_flat = audio.flatten()
        avg_amplitude = np.mean(np.abs(audio_flat))

        if avg_amplitude < silence_threshold:
            return "there is no voice detected"

        return fs, audio

    except Exception as e:
        st.error(f"Recording failed: {e}")
        return "there is no voice detected"

def transcribe_with_assemblyai(audio, fs):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name
        scipy.io.wavfile.write(path, fs, audio)

    # Upload audio
    assembly_key = os.getenv("ASSEMBLYAI_API_KEY")
    headers = {'authorization': assembly_key}
    with open(path, 'rb') as f:
        response = requests.post('https://api.assemblyai.com/v2/upload', headers=headers, files={"file": f})
    audio_url = response.json()['upload_url']

    # Transcription request
    json_data = {"audio_url": audio_url}
    response = requests.post("https://api.assemblyai.com/v2/transcript", json=json_data, headers=headers)
    transcript_id = response.json()['id']

    # Poll for result
    status = "queued"
    while status not in ("completed", "error"):
        result = requests.get(f"https://api.assemblyai.com/v2/transcript/{transcript_id}", headers=headers).json()
        status = result["status"]

    if status == "completed":
        return result["text"]
    else:
        st.error("Transcription failed.")
        return ""

    
def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


# --- Main UI ---
def run_ui2():
    st.set_page_config(page_title="PRIZM Agent Chat", layout="wide")
    st.markdown("<h1 style='color:#002b36; '>Ask Clarix anything! </h1>", unsafe_allow_html=True)

    # Initialize session state variables
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "feedback_log" not in st.session_state:
        st.session_state.feedback_log = []

    if "conversation_ended" not in st.session_state:
        st.session_state.conversation_ended = False
    
    st.markdown(
        """
        <style>
        body {
            background-color: #fdf6e3; /* Cream/off-white */
            color: #002b36;           /* Dark blue/navy */
        }
        .stApp {
            background-color: #fdf6e3;
            color: #002b36;
        }
        /* Style Streamlit buttons */
        .stButton > button {
            background-color: #002b36;
            color: #fdf6e3;
            border: none;
            padding: 0.5em 1em;
            border-radius: 6px;
            font-weight: bold;
            transition: 0.3s;
        }

        .stButton > button:hover {
            background-color: #014f86;  /* Slightly lighter on hover */
            color: white;
            cursor: pointer;
        }

        /* Optional: style text input/textarea */
        textarea, input {
            color: #002b36 !important;
            background-color: #fefcf5 !important;
        }
            /* Change text color of radio options */
        div.stRadio label {
            color: #1E3A8A !important;
            font-size: 18px !important;
            font-weight: 500 !important;
        }
        div[role="radiogroup"] label span {
            color: #1E3A8A !important;
            font-size: 18px !important;
            font-weight: 600 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
        )



    # Layout: left (agent + input), right (chat history)
    left_col, right_col = st.columns([3, 2])

    # --- Left Side: Agent Icon and Input ---
    with left_col:
        icon_path = Path(__file__).resolve().parent / "agent_icon.png"
        c1, c2, c3 = st.columns([2, 3, 1])
        with c2:
            st.image(str(icon_path), width=250)
        
        mode = st.radio("Choose Input Mode", ["Text", "Voice"])
        text_input = ""

        if mode == "Text":
                        text_input = st.chat_input("What do you want to ask Clarix:")
                        if text_input:
                            response = agent_executor.invoke({"input": text_input})
                            st.session_state.chat_history.append(("User", text_input))
                            st.session_state.chat_history.append(("Agent", response["output"]))
                            speak(response["output"])

        elif mode == "Voice":
            if st.button("🎤 Record"):
                fs, audio = record_audio()
                user_input = transcribe_with_assemblyai(audio, fs)
                st.success(f"Transcribed: {user_input}")
                speak(response["output"])





    # --- Right Side: Chat History Display ---
    with right_col:
        st.markdown("### Your Chat with Clarix")
        chat_container = """
        <div style='
            height: 500px;
            overflow-y: auto;
            padding-right: 10px;
            border: 1px solid #ccc;
            background-color: #fff;
            border-radius: 10px;
        '>
        """

        chat_html = ""
    # Display chat messages
        name_map = {"User": "You", "Agent": "Clarix"}

        for i, (speaker, msg) in enumerate(st.session_state.chat_history):
            display_name = name_map.get(speaker, speaker)

            align = "right" if speaker == "User" else "left"
            bubble_color = "#f0f0f0" if speaker == "User" else "#e7f4f9"
            text_color = "#002b36"

            st.markdown(
                f"""
                <div style="text-align: {align};">
                    <div style="
                        display: inline-block;
                        background-color: {bubble_color};
                        padding: 12px;
                        border-radius: 10px;
                        margin-bottom: 10px;
                        max-width: 80%;
                        text-align: left;
                    ">
                        <strong style="color: {text_color};">{display_name}:</strong><br>{msg}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


                # Only show thumbs on latest agent response
            if speaker == "Agent" and i == len(st.session_state.chat_history) - 1 and not st.session_state.conversation_ended:
                    col1, col2, _ = st.columns([2, 2, 6])  # two small columns
                    with col1:
                        if st.button("👍", key=f"thumbs_up_{i}"):
                            st.session_state.feedback_log.append({
                                "user_input": st.session_state.chat_history[i-1][1],  # user message before agent
                                "agent_response": msg,
                                "feedback": "thumbs_up"
                            })
                            st.success("Thanks for the thumbs up!")
                    with col2:
                        if st.button("👎", key=f"thumbs_down_{i}"):
                            st.session_state.feedback_log.append({
                                "user_input": st.session_state.chat_history[i-1][1],
                                "agent_response": msg,
                                "feedback": "thumbs_down"
                            })
                            st.success("Thanks for the feedback!")

    # End conversation button
    if not st.session_state.conversation_ended:
        if st.button("End Conversation"):
            st.session_state.conversation_ended = True

    # After ending, show download
    if st.session_state.conversation_ended:
        df = export_conversation()
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Conversation CSV",
            data=csv,
            file_name="conversation_log.csv",
            mime="text/csv",
        )
        st.success("Conversation ended! Download your chat log above.")
        st.stop()
