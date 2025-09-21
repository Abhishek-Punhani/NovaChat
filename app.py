import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import json

from normalization import transcribe_audio
from ingest import ingest_segments_to_chroma
from query_engine import choose_tool, query_chroma
from profiler import create_conversation_profile
from outputs import (
    build_text_response,
    build_llm_chart_response,
    build_speaker_activity_chart,
    build_speaker_turn_count_chart,
    build_sentiment_trend_chart,
    build_holistic_analysis_chart,
    build_audio_response
)
from pdf_builder import build_pdf_response
from transcript_parser import parse_transcript_file

st.set_page_config(page_title="Multimodal AI Agent", layout="wide")
st.title("Multimodal AI Agent for Enterprise Conversations 🚀")

tools = [
    {"name": "create_holistic_analysis_chart", "description": "The most powerful analytical tool. Use for any complex or abstract query that requires understanding the entire conversation, like categorizing segments ('financial vs. academic terms') or summarizing time spent on topics."},
    {"name": "create_speaker_turn_count_chart", "description": "A simple tool that generates a chart counting the NUMBER OF TIMES each speaker spoke. Use ONLY for direct questions like 'How many times did each speaker talk?'."},
    {"name": "create_speaker_activity_chart", "description": "A simple tool that generates a chart showing the total DURATION (in seconds) of each speaker's talk time. Use ONLY for questions about who spoke the most or for how long."},
    {"name": "create_sentiment_trend_chart", "description": "A specialized tool that creates a line chart of sentiment over time. Use ONLY for direct questions about mood or sentiment."},
    {"name": "create_keyword_mention_chart", "description": "A simple tool that counts the frequency of specific, concrete keywords mentioned in the query. Use ONLY for direct questions like 'How often was the word fee mentioned?'."},
    {"name": "summarize_text", "description": "The default tool for general questions or text summaries."},
    {"name": "generate_pdf_report", "description": "Creates a downloadable PDF report. Use for requests to 'download a report'."},
    {"name": "generate_audio_summary", "description": "Generates an audio narration of a summary. Use for requests to 'read the summary aloud'."}
]

PROFILE_FILE = "active_profile.json"
TRANSCRIPT_CACHE_FILE = "transcript_cache.json"

if "current_profile" not in st.session_state:
    if os.path.exists(PROFILE_FILE):
        with open(PROFILE_FILE, "r") as f: st.session_state.current_profile = json.load(f)
        if os.path.exists(TRANSCRIPT_CACHE_FILE):
            with open(TRANSCRIPT_CACHE_FILE, "r") as f: st.session_state.full_transcript_segments = json.load(f)
    else:
        st.session_state.current_profile = None
        st.session_state.full_transcript_segments = []

st.sidebar.header("Process New Call Data")
uploaded_file = st.sidebar.file_uploader("Upload a file", type=["mp3", "wav", "txt", "csv", "json"])

if st.session_state.current_profile:
    st.sidebar.write("Current Active Profile:"); st.sidebar.json(st.session_state.current_profile)

if uploaded_file is not None:
    file_path = os.path.join("temp_uploads", uploaded_file.name)
    os.makedirs("temp_uploads", exist_ok=True)
    with open(file_path, "wb") as f: f.write(uploaded_file.getbuffer())
    if st.sidebar.button("Process File"):
        with st.spinner("Processing..."):
            try:
                file_ext = uploaded_file.name.split('.')[-1].lower()
                segments = parse_transcript_file(file_path) if file_ext in ["txt", "csv", "json"] else transcribe_audio(file_path)
                ingest_segments_to_chroma(segments, file_id=uploaded_file.name)
                full_text = " ".join([seg['text'] for seg in segments])
                profile = create_conversation_profile(full_text)
                profile['source_file'] = uploaded_file.name
                st.session_state.current_profile = profile
                st.session_state.full_transcript_segments = segments
                with open(PROFILE_FILE, "w") as f: json.dump(profile, f)
                with open(TRANSCRIPT_CACHE_FILE, "w") as f: json.dump(segments, f)
                st.sidebar.success("File processed and profiled!")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"An error occurred: {e}")

if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]): st.markdown(message["content"])

def display_chart(response):
    if "error" not in response:
        data = response.get("data")
        is_empty = data is None or (isinstance(data, pd.Series) and data.empty) or (isinstance(data, dict) and not data)
        if is_empty:
            st.warning("The AI could not find any data to plot for this query.")
            return
        chart_type = response.get("chart_type")
        fig, ax = plt.subplots()
        if chart_type == "bar":
            df = pd.DataFrame.from_dict(data, orient='index', columns=['value'])
            df.plot(kind='bar', ax=ax, legend=False); plt.xticks(rotation=45, ha="right")
        elif chart_type == "pie":
            ax.pie(data.values(), labels=data.keys(), autopct='%1.1f%%', startangle=90); ax.axis('equal')
        elif chart_type == "line":
            data.plot(kind='line', ax=ax, legend=False); plt.axhline(0, color='grey', linewidth=0.8)
        ax.set_title(response.get("title", "Chart"))
        ax.set_xlabel(response.get("x_label", ""))
        ax.set_ylabel(response.get("y_label", ""))
        plt.tight_layout(); st.pyplot(fig)
    else:
        st.error(response.get("error"))

if prompt := st.chat_input("Ask anything about your calls..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            profile = st.session_state.get("current_profile")
            if not profile:
                st.warning("Please process a file first."); st.stop()

            chosen_tool = choose_tool(prompt, tools, profile)
            
            if chosen_tool == "create_holistic_analysis_chart":
                full_transcript = st.session_state.get("full_transcript_segments", [])
                response = build_holistic_analysis_chart(prompt, full_transcript)
                display_chart(response)
            else:
                top_k = 75 if any(k in chosen_tool for k in ["activity", "sentiment", "turn_count"]) else 20
                retrieved = query_chroma(prompt, top_k=top_k)

                if "chart" in chosen_tool:
                    # --- MODIFICATION: Pass the 'prompt' to the builder functions ---
                    if chosen_tool == "create_speaker_turn_count_chart":
                        response = build_speaker_turn_count_chart(prompt, retrieved)
                    elif chosen_tool == "create_sentiment_trend_chart":
                        response = build_sentiment_trend_chart(prompt, retrieved)
                    elif chosen_tool == "create_speaker_activity_chart":
                        response = build_speaker_activity_chart(prompt, retrieved)
                    else: 
                        response = build_llm_chart_response(prompt, retrieved)
                    display_chart(response)
                
                elif chosen_tool == "generate_pdf_report":
                    response = build_pdf_response(prompt, retrieved, profile)
                    with open(response["pdf_path"], "rb") as f:
                        st.download_button("Download Report", f, "report.pdf", "application/pdf")
                elif chosen_tool == "generate_audio_summary":
                    text_response = build_text_response(prompt, retrieved)
                    summary_text = text_response.get("text_summary")
                    if summary_text and not summary_text.startswith("Error:"):
                        audio_response = build_audio_response(summary_text)
                        st.audio(audio_response["audio_path"])
                    else: st.error("Could not generate audio.")
                else: # Catches "summarize_text"
                    response = build_text_response(prompt, retrieved)
                    st.markdown(response.get("text_summary", "Sorry, I couldn't find an answer."))

    st.session_state.messages.append({"role": "assistant", "content": "Done."})