
import streamlit as st
from streamlit_mic_recorder import speech_to_text
from agent.agent_setup import setup_agent
import json
from pathlib import Path
import pandas as pd
import re
import os
import asyncio
import edge_tts
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()

# --- TTS Function using Edge ---
def speak(text):
    async def run():
        communicate = edge_tts.Communicate(text, voice="en-US-AriaNeural")
        await communicate.save("output.mp3")
        os.system("afplay output.mp3")
    asyncio.run(run())

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
    for i, (speaker, msg) in enumerate(st.session_state.chat_history):
        if speaker == "User":
            conversation_data.append({"speaker": speaker, "message": msg, "feedback": "NA"})
        elif speaker == "Agent":
            feedback_entry = next((entry for entry in st.session_state.feedback_log if entry["agent_response"] == msg), None)
            feedback = feedback_entry["feedback"] if feedback_entry else "No Feedback"
            conversation_data.append({"speaker": speaker, "message": msg, "feedback": feedback})
    return pd.DataFrame(conversation_data)

# --- Main UI ---
def run_ui():
    st.set_page_config(page_title="PRIZM Agent Chat", layout="centered")
    st.title("Chat Agent")

    # --- Session State Initialization ---
    for key, default in {
        "chat_history": [],
        "feedback_log": [],
        "conversation_ended": False,
        "pending_tts": None
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

    st.subheader("Choose input mode")
    input_mode = st.radio("Input mode:", ["Text", "Voice"], horizontal=True)

    user_input = ""

    if input_mode == "Text":
        user_input = st.chat_input("Ask the agent something...")
    else:
        with st.spinner("🎙️ Recording..."):
            def callback():
                if st.session_state.my_stt_output:
                    st.write(f"📝 Transcribed: {st.session_state.my_stt_output}")
            speech_to_text(key='my_stt', language='en', start_prompt="🎤 Start Recording", stop_prompt="🛑 Stop Recording", just_once=True, callback=callback)
        user_input = st.session_state.get("my_stt_output", "")

    # --- Handle input and agent response ---
    if user_input:
        response = agent_executor.invoke({"input": user_input})
        st.session_state.chat_history.append(("User", user_input))
        st.session_state.chat_history.append(("Agent", response["output"]))

        # Defer speech until after UI renders
        if input_mode == "Voice":
            st.session_state.pending_tts = response["output"]
            st.rerun()

    # --- Display chat ---
    for i, (speaker, msg) in enumerate(st.session_state.chat_history):
        with st.chat_message(speaker):
            if isinstance(msg, str) and msg.startswith("![Time Series]("):
                match = re.search(r"base64,([A-Za-z0-9+/=]+)", msg)
                if match:
                    st.image(f"data:image/png;base64,{match.group(1)}")
                else:
                    st.markdown(msg)
            else:
                st.markdown(msg)

            # Feedback UI
            if speaker == "Agent" and i == len(st.session_state.chat_history) - 1 and not st.session_state.conversation_ended:
                col1, col2, _ = st.columns([1, 1, 6])
                with col1:
                    if st.button("👍", key=f"thumbs_up_{i}"):
                        st.session_state.feedback_log.append({
                            "user_input": st.session_state.chat_history[i - 1][1],
                            "agent_response": msg,
                            "feedback": "thumbs_up"
                        })
                        st.success("Thanks for the thumbs up!")
                with col2:
                    if st.button("👎", key=f"thumbs_down_{i}"):
                        st.session_state.feedback_log.append({
                            "user_input": st.session_state.chat_history[i - 1][1],
                            "agent_response": msg,
                            "feedback": "thumbs_down"
                        })
                        st.success("Thanks for the feedback!")

    # --- Speak after rendering ---
    if st.session_state.pending_tts:
        speak(st.session_state.pending_tts)
        st.session_state.pending_tts = None  # Reset

    # --- Conversation End ---
    if not st.session_state.conversation_ended:
        if st.button("End Conversation"):
            st.session_state.conversation_ended = True

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
