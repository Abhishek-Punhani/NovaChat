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

MODEL_ID = "gemini-2.5-flash"
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
    context_list = [f"duration={seg.get('end', 0) - seg.get('start', 0):.1f}s, text='{seg['text']}'" for seg in full_transcript_segments]
    context_str = "\n".join(context_list)
    prompt = f"""
    You are a data analyst bot. Analyze the conversation data and extract numerical data for creating a visual chart.
    
    Instructions:
    1. Analyze the User Query to understand what needs to be categorized/counted
    2. Look through the conversation context and extract the relevant data
    3. If no relevant data is found in the context, generate reasonable sample data based on the query (e.g., fictional sales metrics or counts)
    4. Return ONLY the numerical data in this exact format (no text explanations, no text-based charts, no conversation):
    
    Chart Type: [bar/pie/line]
    Title: [descriptive title]
    x_label: [x-axis label]
    y_label: [y-axis label]
    Data: 'Category1'=number1, 'Category2'=number2, 'Category3'=number3
    
    DO NOT include any other text. Only return the structured data format above.
    
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
    docs = retrieved.get("documents", [])
    context = "\n".join(docs)
    prompt = f"""
    You are a data analyst bot. Analyze the conversation data and extract numerical data for creating a visual chart.
    
    Instructions:
    1. Analyze the User Query to understand what needs to be categorized/counted
    2. Look through the conversation context and extract the relevant data
    3. Count/categorize the data based on the query requirements
    4. Return ONLY the numerical data in this exact format (no text explanations or text-based charts):
    
    Chart Type: [bar/pie/line]
    Title: [descriptive title]
    x_label: [x-axis label]
    y_label: [y-axis label]
    Data: 'Category1'=number1, 'Category2'=number2, 'Category3'=number3
    
    DO NOT include any text-based charts or explanations. Only return the structured data format above.
    
    User Query: '{query}'
    """
    response_text = _call_llm(prompt, context)
    try:
        chart_details = {"type": "chart"}
        chart_type_match = re.search(r"Chart Type:\s*(\w+)", response_text, re.IGNORECASE)
        if chart_type_match:
            chart_details["chart_type"] = chart_type_match.group(1).lower()
        else:
            chart_details["chart_type"] = detect_chart_type(query)
        title_match = re.search(r"Title:\s*(.+)", response_text, re.IGNORECASE)
        chart_details["title"] = title_match.group(1).strip() if title_match else "AI Generated Chart"
        xlabel_match = re.search(r"x_label:\s*(.+)", response_text, re.IGNORECASE)
        chart_details["x_label"] = xlabel_match.group(1).strip() if xlabel_match else ""
        ylabel_match = re.search(r"y_label:\s*(.+)", response_text, re.IGNORECASE)
        chart_details["y_label"] = ylabel_match.group(1).strip() if ylabel_match else ""
        data_pairs = re.findall(r"['\"]?([\w\s]+)['\"]?\s*[:=]\s*(\d+)", response_text)
        if not data_pairs:
            raise ValueError("No valid key-value pairs found.")
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