from typing import Dict, Any, Optional, List
from sentence_transformers import SentenceTransformer
import chromadb
from outputs import _call_llm 
import json
from config import EMBED_MODEL_NAME, CHROMA_DB_PATH, COLLECTION_NAME


model = SentenceTransformer(EMBED_MODEL_NAME)
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

# Context-Aware Router Function
def choose_tool(query: str, available_tools: List[Dict], conversation_profile: Dict) -> str:
    """
    Uses an LLM to act as a router with chain-of-thought reasoning,
    selecting the best tool by considering query details and conversation profile.
    """
    # Get conversation history from session state if available
    conversation_history = ""
    try:
        import streamlit as st
        if hasattr(st, 'session_state') and 'messages' in st.session_state:
            recent_messages = st.session_state.messages[-6:]  # Last 3 exchanges
            conversation_history = "\n".join([
                f"{msg['role'].upper()}: {msg['content']}" 
                for msg in recent_messages
            ])
    except:
        conversation_history = "No conversation history available."
    
    # Pre-check for explicit keywords - FIXED LOGIC
    query_lower = query.lower()
    chart_keywords = ['chart', 'graph', 'plot', 'visualize']
    pdf_keywords = ['pdf', 'download']
    report_keywords = ['report']
    
    # Check for direct keyword matches FIRST (preserves original functionality)
    if any(keyword in query_lower for keyword in pdf_keywords):
        print("Direct PDF keywords detected, routing to generate_pdf_report")
        return "generate_pdf_report"
    elif any(keyword in query_lower for keyword in report_keywords) and not any(keyword in query_lower for keyword in chart_keywords):
        print("Direct report keywords detected (not chart), routing to generate_pdf_report")
        return "generate_pdf_report"
    elif any(keyword in query_lower for keyword in chart_keywords):
        print("Direct chart keywords detected, proceeding with LLM chain-of-thought routing for chart selection")
        # Continue to LLM routing for chart type selection
    else:
        print("No direct keywords detected, using LLM chain-of-thought routing")
        # Continue to LLM routing
    
    # Chain-of-thought routing with LLM
    tool_descriptions = "\n".join([
        f"- Tool Name: {tool['name']}\n  Description: {tool['description']}" 
        for tool in available_tools
    ])
    
    profile_str = json.dumps(conversation_profile, indent=2)
    
    prompt = f"""
    You are an intelligent routing agent. Use chain-of-thought reasoning to select the best tool.

    STEP 1: Analyze the user's query intent
    STEP 2: Consider the conversation history and profile context  
    STEP 3: Look for references to previous responses or conversation elements
    STEP 4: Match the intent to the most appropriate tool
    STEP 5: Provide your final decision

    Conversation History:
    {conversation_history}

    Conversation Profile:
    {profile_str}

    Available Tools:
    {tool_descriptions}

    User Query: "{query}"

    Think step by step:
    1. What is the user asking for? (chart, text summary, PDF report, etc.)
    2. What type of analysis is needed? (simple counting, complex analysis, etc.)
    3. Does the query reference previous conversation context or responses?
    4. Are there implicit references to earlier discussion points?
    5. Which tool best matches this intent considering the conversation flow?

    Show your reasoning for each step, then provide:
    Final Decision: [exact_tool_name]
    """
    
    print("Using chain-of-thought routing...")
    response = _call_llm(prompt, context="")
    print(f"LLM routing response: {response}")
    
    # Extract the final decision
    if "Final Decision:" in response:
        chosen_tool_name = response.split("Final Decision:")[-1].strip()
    else:
        chosen_tool_name = response.strip()
    
    # Match against available tools
    for tool in available_tools:
        if tool['name'] in chosen_tool_name:
            print(f"Router selected tool: {tool['name']}")
            return tool['name']
    
    # Final fallback (shouldn't happen if LLM works correctly)
    print("Router defaulted to: summarize_text")
    return "summarize_text"

def query_chroma(query: str, top_k: int = 5, metadata_filters: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Queries the ChromaDB collection for relevant text segments.
    """
    q_emb = model.encode([query]).tolist()

    query_args = {
        'query_embeddings': q_emb,
        'n_results': top_k
    }
    if metadata_filters:
        query_args['where'] = metadata_filters

    res = collection.query(**query_args)

    docs = res.get("documents", [[]])[0]
    metadatas = res.get("metadatas", [[]])[0]
    ids = res.get("ids", [[]])[0]
    distances = res.get("distances", [[]])[0]

    return {"documents": docs, "metadatas": metadatas, "ids": ids, "distances": distances}