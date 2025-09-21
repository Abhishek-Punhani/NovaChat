from typing import List, Dict, Any
from gtts import gTTS
import os
import json
import re
from dotenv import load_dotenv
import pandas as pd
from transformers import pipeline
import google.generativeai as genai

# ==============================================================================
# AI Integration & Models
# ==============================================================================

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("WARNING: Google API key not found.")
else:
    genai.configure(api_key=GOOGLE_API_KEY)

MODEL_ID = "gemini-1.5-flash-latest"
llm_model = genai.GenerativeModel(MODEL_ID)

sentiment_analyzer = pipeline(
    "sentiment-analysis", 
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)

def _call_llm(prompt: str, context: str) -> str:
    print(f"--- Calling {MODEL_ID} via Google Gemini API ---")
    full_prompt = f"{prompt}\n\nCONTEXT:\n{context}"
    try:
        response = llm_model.generate_content(full_prompt)
        return response.text.replace("```json", "").replace("```", "").strip()
    except Exception as e:
        print(f"An error occurred with the Gemini API: {e}")
        return f"Error: Could not get a response from the AI. Details: {e}"

# ==============================================================================
# Helper Functions
# ==============================================================================
def normalize_speaker_name(speaker: str) -> str:
    """Unifies speaker names to a consistent format like 'SPEAKER_0'."""
    match = re.search(r"(\d+)", str(speaker))
    if match:
        return f"SPEAKER_{match.group(1)}"
    return str(speaker).upper().strip()

def detect_chart_type(query: str, default: str = "bar") -> str:
    """Detects the requested chart type from the user's query."""
    q = query.lower()
    if "pie chart" in q: return "pie"
    if "bar chart" in q or "bar graph" in q: return "bar"
    if "line chart" in q or "line graph" in q: return "line"
    return default

# ==============================================================================
# Core Builder Functions ("Tools")
# ==============================================================================
def build_holistic_analysis_chart(query: str, full_transcript_segments: List[Dict]) -> Dict[str, Any]:
    # ... (This function remains as it was, it's already advanced)
    context_list = [f"duration={seg.get('end', 0) - seg.get('start', 0):.1f}s, text='{seg['text']}'" for seg in full_transcript_segments]
    context_str = "\n".join(context_list)
    prompt = f"""
    You are a world-class conversation analyst. Your task is to perform a complex analysis of the entire transcript based on the user's query.
    Instructions:
    1. Read the User Query to understand the analysis required.
    2. Read the entire transcript context. Each line shows a segment's duration and text.
    3. Perform the requested analysis (e.g., topic modeling with time).
    4. You MUST return the results in the simple text format shown below. Generate the 'Title', 'x_label', and 'y_label' based on the query. Do NOT add conversation.
    EXAMPLE:
    Chart Type: pie
    Title: Time Spent per Topic
    x_label: Topics
    y_label: Total Duration (seconds)
    Data: 'Pricing'=120, 'Features'=95
    User Query: '{query}'
    """
    response_text = _call_llm(prompt, context=context_str)
    try:
        chart_details = {"type": "chart"}
        chart_type_match = re.search(r"Chart Type:\s*(\w+)", response_text, re.IGNORECASE)
        chart_details["chart_type"] = chart_type_match.group(1).lower() if chart_type_match else "bar"
        title_match = re.search(r"Title:\s*(.+)", response_text, re.IGNORECASE)
        chart_details["title"] = title_match.group(1).strip() if title_match else "Holistic Analysis Chart"
        xlabel_match = re.search(r"x_label:\s*(.+)", response_text, re.IGNORECASE)
        chart_details["x_label"] = xlabel_match.group(1).strip() if xlabel_match else ""
        ylabel_match = re.search(r"y_label:\s*(.+)", response_text, re.IGNORECASE)
        chart_details["y_label"] = ylabel_match.group(1).strip() if ylabel_match else ""
        data_pairs = re.findall(r"['\"]?([\w\s&]+)['\"]?\s*[:=]\s*(\d+)", response_text)
        if not data_pairs: raise ValueError("No valid data pairs found.")
        chart_details["data"] = {key.strip(): int(value) for key, value in data_pairs}
        return chart_details
    except Exception as e:
        print(f"Holistic Chart Parsing Error: {e}\nLLM Response was:\n{response_text}")
        return {"type": "chart", "error": "The AI failed to perform the complex analysis."}

def build_text_response(query: str, retrieved: Dict[str, Any]) -> Dict[str, Any]:
    # ... (unchanged)
    docs = retrieved.get("documents", [])
    context = "\n".join(f"- {doc}" for doc in docs)
    prompt = f"You are a helpful AI assistant. Summarize the provided context to answer the user's query concisely.\nUser Query: '{query}'"
    summary_text = _call_llm(prompt, context)
    return {"type": "text", "text_summary": summary_text, "retrieved_count": len(docs)}

def build_llm_chart_response(query: str, retrieved: Dict[str, Any]) -> Dict[str, Any]:
    # ... (unchanged)
    docs = retrieved.get("documents", [])
    context = "\n".join(docs)
    prompt = f"""
    You are a data analyst bot. Your ONLY job is to follow instructions to create chart data.
    1. Analyze the User Query to understand the data to extract and how to label it.
    2. Perform the analysis on the context (e.g., counting, categorization).
    3. IMPORTANT: If the user requests a specific chart type (e.g., "pie", "bar"), you MUST use it. Otherwise, decide the best one.
    4. You MUST return the data in the simple text format shown in the example. The 'Title', 'x_label', and 'y_label' MUST be generated based on the User Query. Do NOT add conversation.
    EXAMPLE:
    Chart Type: bar
    Title: Number of Segments by Category
    x_label: Category
    y_label: Number of Segments
    Data: 'Information'=12, 'Question'=5
    User Query: '{query}'
    """
    response_text = _call_llm(prompt, context)
    try:
        chart_details = {"type": "chart"}
        chart_type_match = re.search(r"Chart Type:\s*(\w+)", response_text, re.IGNORECASE)
        chart_details["chart_type"] = chart_type_match.group(1).lower() if chart_type_match else "bar"
        title_match = re.search(r"Title:\s*(.+)", response_text, re.IGNORECASE)
        chart_details["title"] = title_match.group(1).strip() if title_match else "AI Generated Chart"
        xlabel_match = re.search(r"x_label:\s*(.+)", response_text, re.IGNORECASE)
        chart_details["x_label"] = xlabel_match.group(1).strip() if xlabel_match else ""
        ylabel_match = re.search(r"y_label:\s*(.+)", response_text, re.IGNORECASE)
        chart_details["y_label"] = ylabel_match.group(1).strip() if ylabel_match else ""
        data_pairs = re.findall(r"['\"]?([\w\s]+)['\"]?\s*[:=]\s*(\d+)", response_text)
        if not data_pairs: raise ValueError("No valid key-value pairs found.")
        chart_details["data"] = {key.strip(): int(value) for key, value in data_pairs}
        return chart_details
    except Exception as e:
        print(f"Chart Parsing Error: {e}\nLLM Response was:\n{response_text}")
        return {"type": "chart", "error": "Failed to generate valid chart data from the AI."}

def build_speaker_activity_chart(query: str, retrieved: Dict[str, Any]) -> Dict[str, Any]:
    metas = retrieved.get("metadatas", [])
    speaker_talk_time = {}
    for md in metas:
        speaker = normalize_speaker_name(md.get("speaker", "Unknown"))
        start, end = md.get("start", 0), md.get("end", 0)
        duration = end - start
        if duration > 0: speaker_talk_time[speaker] = speaker_talk_time.get(speaker, 0) + duration
    if not speaker_talk_time: return {"type": "chart", "error": "No speaker activity data found to plot."}
    
    chart_type = detect_chart_type(query, default="bar")
    return {
        "type": "chart", "chart_type": chart_type,
        "title": "Speaker Talk Time Distribution" if chart_type == "pie" else "Total Speaker Talk Time",
        "x_label": "Speaker", "y_label": "Duration (seconds)", "data": speaker_talk_time
    }

def build_speaker_turn_count_chart(query: str, retrieved: Dict[str, Any]) -> Dict[str, Any]:
    metas = retrieved.get("metadatas", [])
    speaker_turn_counts = {}
    for md in metas:
        speaker = normalize_speaker_name(md.get("speaker", "Unknown"))
        speaker_turn_counts[speaker] = speaker_turn_counts.get(speaker, 0) + 1
    if not speaker_turn_counts: return {"type": "chart", "error": "No speaker turn data found to plot."}
    
    chart_type = detect_chart_type(query, default="bar")
    return {
        "type": "chart", "chart_type": chart_type,
        "title": "Speaker Turn Distribution" if chart_type == "pie" else "Number of Times Each Speaker Spoke",
        "x_label": "Speaker", "y_label": "Number of Turns", "data": speaker_turn_counts
    }

def build_sentiment_trend_chart(query: str, retrieved: Dict[str, Any]) -> Dict[str, Any]:
    metas = retrieved.get("metadatas", []); docs = retrieved.get("documents", [])
    speaker_filter = None
    match = re.search(r"(speaker\d+|speaker_\d+)", query, re.IGNORECASE)
    if match: speaker_filter = normalize_speaker_name(match.group(0))
    sentiment_over_time = []
    for doc, md in zip(docs, metas):
        speaker = normalize_speaker_name(md.get("speaker", "Unknown"))
        if speaker_filter and speaker != speaker_filter: continue
        if doc:
            sentiment_result = sentiment_analyzer(doc)[0]
            score = 0
            if sentiment_result['label'] == 'positive': score = 1
            elif sentiment_result['label'] == 'negative': score = -1
            sentiment_over_time.append({"time": md.get("start", 0), "sentiment_score": score})
    if not sentiment_over_time: return {"type": "chart", "error": f"No text for '{speaker_filter or 'anyone'}' to analyze."}
    df = pd.DataFrame(sentiment_over_time).sort_values(by="time")
    return {
        "type": "chart", "chart_type": "line",
        "title": f"Sentiment Trend for {speaker_filter or 'All Speakers'}",
        "x_label": "Time (seconds into call)", "y_label": "Sentiment Score (-1 to 1)",
        "data": df.set_index("time")["sentiment_score"]
    }

def build_audio_response(text: str, out_mp3: str = "narration.mp3") -> Dict[str, Any]:
    tts = gTTS(text=text, lang="en")
    tts.save(out_mp3)
    return {"type": "audio", "audio_path": out_mp3}