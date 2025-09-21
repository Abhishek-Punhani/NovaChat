import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    {"name": "generate_pdf_report", "description": "Creates a downloadable PDF report. Use for requests to 'generate a report', 'create a PDF', or 'download a report'."},
    {"name": "generate_audio_summary", "description": "Generates an audio narration of a summary. Use for requests to 'read the summary aloud'."}
]

PROFILES_DIR = "conversation_profiles"
TRANSCRIPTS_DIR = "transcript_cache"

# Initialize session state for multiple files
if "conversation_files" not in st.session_state:
    st.session_state.conversation_files = {}
if "active_conversation" not in st.session_state:
    st.session_state.active_conversation = None

# Create directories
os.makedirs(PROFILES_DIR, exist_ok=True)
os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)
os.makedirs("temp_uploads", exist_ok=True)

def process_single_file(file_info):
    """Process a single file and return results"""
    try:
        file_path, file_name = file_info
        file_ext = file_name.split('.')[-1].lower()
        
        # Parse or transcribe
        if file_ext in ["txt", "csv", "json"]:
            segments = parse_transcript_file(file_path)
        else:
            segments = transcribe_audio(file_path)
        
        # Ingest to ChromaDB with unique file_id
        ingest_segments_to_chroma(segments, file_id=file_name)
        
        # Create profile
        full_text = " ".join([seg['text'] for seg in segments])
        profile = create_conversation_profile(full_text)
        profile['source_file'] = file_name
        
        return {
            'file_name': file_name,
            'profile': profile,
            'segments': segments,
            'status': 'success'
        }
    except Exception as e:
        return {
            'file_name': file_name,
            'error': str(e),
            'status': 'error'
        }

# Sidebar for file management
st.sidebar.header("📁 Conversation Manager")

# Multi-file uploader
uploaded_files = st.sidebar.file_uploader(
    "Upload conversation files", 
    type=["mp3", "wav", "txt", "csv", "json"],
    accept_multiple_files=True
)

if uploaded_files:
    st.sidebar.write(f"📤 {len(uploaded_files)} files selected")
    
    if st.sidebar.button("🚀 Process All Files"):
        with st.spinner(f"Processing {len(uploaded_files)} files concurrently..."):
            # Save uploaded files
            file_paths = []
            for uploaded_file in uploaded_files:
                file_path = os.path.join("temp_uploads", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                file_paths.append((file_path, uploaded_file.name))
            
            # Process files concurrently (max 5 at a time)
            results = []
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_file = {executor.submit(process_single_file, file_info): file_info[1] 
                                for file_info in file_paths}
                
                # Show progress
                progress_bar = st.sidebar.progress(0)
                completed = 0
                
                for future in as_completed(future_to_file):
                    result = future.result()
                    results.append(result)
                    completed += 1
                    progress_bar.progress(completed / len(uploaded_files))
            
            # Update session state
            for result in results:
                if result['status'] == 'success':
                    st.session_state.conversation_files[result['file_name']] = {
                        'profile': result['profile'],
                        'segments': result['segments']
                    }
                    
                    # Save to disk
                    profile_path = os.path.join(PROFILES_DIR, f"{result['file_name']}_profile.json")
                    transcript_path = os.path.join(TRANSCRIPTS_DIR, f"{result['file_name']}_transcript.json")
                    
                    with open(profile_path, "w") as f:
                        json.dump(result['profile'], f)
                    with open(transcript_path, "w") as f:
                        json.dump(result['segments'], f)
                else:
                    st.sidebar.error(f"❌ {result['file_name']}: {result['error']}")
            
            success_count = sum(1 for r in results if r['status'] == 'success')
            st.sidebar.success(f"✅ Successfully processed {success_count}/{len(uploaded_files)} files!")
            st.rerun()

# Display loaded conversations
if st.session_state.conversation_files:
    st.sidebar.subheader("💬 Available Conversations")
    
    conversation_names = list(st.session_state.conversation_files.keys())
    selected_conversation = st.sidebar.selectbox(
        "Select active conversation:",
        options=conversation_names,
        index=conversation_names.index(st.session_state.active_conversation) 
        if st.session_state.active_conversation in conversation_names else 0
    )
    
    if selected_conversation != st.session_state.active_conversation:
        st.session_state.active_conversation = selected_conversation
        st.rerun()
    
    # Show active conversation info
    if st.session_state.active_conversation:
        active_data = st.session_state.conversation_files[st.session_state.active_conversation]
        st.sidebar.write("🎯 **Active Conversation:**")
        st.sidebar.write(f"📄 {st.session_state.active_conversation}")
        
        with st.sidebar.expander("View Profile"):
            st.json(active_data['profile'])

# Load existing conversations on startup
def load_existing_conversations():
    """Load previously processed conversations"""
    if os.path.exists(PROFILES_DIR):
        for profile_file in os.listdir(PROFILES_DIR):
            if profile_file.endswith("_profile.json"):
                file_name = profile_file.replace("_profile.json", "")
                transcript_file = os.path.join(TRANSCRIPTS_DIR, f"{file_name}_transcript.json")
                
                if os.path.exists(transcript_file):
                    try:
                        with open(os.path.join(PROFILES_DIR, profile_file), "r") as f:
                            profile = json.load(f)
                        with open(transcript_file, "r") as f:
                            segments = json.load(f)
                        
                        st.session_state.conversation_files[file_name] = {
                            'profile': profile,
                            'segments': segments
                        }
                    except Exception as e:
                        continue

# Load existing conversations on first run
if not st.session_state.conversation_files:
    load_existing_conversations()

# ...existing code for messages and display_chart...
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

# Modified chat input to work with selected conversation
if prompt := st.chat_input("Ask anything about your calls..."):
    if not st.session_state.active_conversation:
        st.warning("Please select an active conversation first.")
        st.stop()
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Get active conversation data
            active_data = st.session_state.conversation_files[st.session_state.active_conversation]
            profile = active_data['profile']
            full_transcript_segments = active_data['segments']

            chosen_tool = choose_tool(prompt, tools, profile)
            
            if chosen_tool == "create_holistic_analysis_chart":
                response = build_holistic_analysis_chart(prompt, full_transcript_segments)
                display_chart(response)
            else:
                top_k = 75 if any(k in chosen_tool for k in ["activity", "sentiment", "turn_count"]) else 20
                # Query ChromaDB with file-specific filter
                metadata_filters = {"file_id": st.session_state.active_conversation}
                retrieved = query_chroma(prompt, top_k=top_k, metadata_filters=metadata_filters)

                if "chart" in chosen_tool:
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
                        audio_file_path = f"temp_uploads/audio_{len(st.session_state.messages)}.mp3"
                        audio_response = build_audio_response(summary_text, out_mp3=audio_file_path)
                        
                        if os.path.exists(audio_response["audio_path"]):
                            with open(audio_response["audio_path"], "rb") as audio_file:
                                st.audio(audio_file.read(), format="audio/mp3")
                            
                            with open(audio_response["audio_path"], "rb") as audio_file:
                                st.download_button(
                                    "Download Audio Summary",
                                    audio_file,
                                    file_name=f"summary_{len(st.session_state.messages)}.mp3",
                                    mime="audio/mp3"
                                )
                        else:
                            st.error("Failed to generate audio file.")
                    else: 
                        st.error("Could not generate audio - no valid text to convert.")
                else: 
                    response = build_text_response(prompt, retrieved)
                    st.markdown(response.get("text_summary", "Sorry, I couldn't find an answer."))

    st.session_state.messages.append({"role": "assistant", "content": "Done."})