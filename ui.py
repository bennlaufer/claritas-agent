import streamlit as st
from agent.agent_setup import setup_agent
import json
from pathlib import Path
import pandas as pd

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

# --- Main UI ---
def run_ui():
    st.set_page_config(page_title="PRIZM Agent Chat", layout="centered")
    st.title("Chat Agent")

    # Initialize session state variables
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "feedback_log" not in st.session_state:
        st.session_state.feedback_log = []

    if "conversation_ended" not in st.session_state:
        st.session_state.conversation_ended = False

    # User input
    user_input = st.chat_input("Ask the agent something...")

    if user_input:
        response = agent_executor.invoke({"input": user_input})
        st.session_state.chat_history.append(("User", user_input))
        st.session_state.chat_history.append(("Agent", response["output"]))

    # Display chat messages
    for i, (speaker, msg) in enumerate(st.session_state.chat_history):
        with st.chat_message(speaker):
            st.markdown(msg)

            # Only show thumbs on latest agent response
            if speaker == "Agent" and i == len(st.session_state.chat_history) - 1 and not st.session_state.conversation_ended:
                col1, col2, _ = st.columns([1, 1, 6])  # two small columns
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
