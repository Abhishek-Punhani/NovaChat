import os
import time
import sounddevice as sd
import numpy as np
import pyttsx3
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
import threading
import queue

# 1. Configuration and Setup 
load_dotenv()

# Whisper STT Configuration
MODEL_SIZE = "base.en"      
WHISPER_DEVICE = "cpu"      
WHISPER_COMPUTE = "int8"    

# Audio Recording Configuration
INTERVAL = 5                # Reduced to 5 seconds for better responsiveness
SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 0.01    # Threshold for detecting silence
MIN_SPEECH_LENGTH = 1.0     # Minimum speech length in seconds

print("Initializing components...")

# 2. Better TTS Initialization
def init_tts():
    """Initialize TTS with proper error handling"""
    try:
        engine = pyttsx3.init()
        
        # Set TTS properties for better performance
        voices = engine.getProperty('voices')
        if voices:
            # Try to use a female voice if available
            for voice in voices:
                if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                    engine.setProperty('voice', voice.id)
                    break
        
        # Set speech rate and volume
        engine.setProperty('rate', 180)  # Slightly faster speech
        engine.setProperty('volume', 0.9)
        
        print("✅ TTS engine initialized successfully.")
        return engine
    except Exception as e:
        print(f"❌ TTS initialization failed: {e}")
        return None

tts_engine = init_tts()

# 3. Initialize Whisper with better error handling
def init_whisper():
    """Initialize Whisper model with error handling"""
    try:
        print(f"Loading Whisper model '{MODEL_SIZE}'...")
        model = WhisperModel(MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
        print("✅ Whisper model loaded.")
        return model
    except Exception as e:
        print(f"❌ Whisper initialization failed: {e}")
        return None

whisper_model = init_whisper()

# 4. Initialize LangChain Agent with better error handling
def init_agent():
    """Initialize the LangChain agent"""
    try:
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3)
        
        @tool
        def get_current_time_and_location():
            """Returns the current time and location."""
            current_time = time.strftime("%A, %B %d, %Y %I:%M %p")
            return f"The current time in Varanasi, India is {current_time}."
        
        @tool
        def end_conversation():
            """End the conversation when user says goodbye or similar."""
            return "Goodbye! Have a great day!"
        
        tools = [get_current_time_and_location, end_conversation]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful voice assistant named Nova. Keep responses under 50 words, 
             conversational, and suitable for voice interaction. You're located in Varanasi, India.
             If user says goodbye, bye, exit, quit, or stop - use the end_conversation tool."""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        
        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, max_iterations=3)
        
        print("✅ LangChain agent initialized.")
        return agent_executor
    except Exception as e:
        print(f"❌ Agent initialization failed: {e}")
        return None

agent_executor = init_agent()

# 5. Improved Helper Functions

def detect_speech_in_audio(audio_data, threshold=SILENCE_THRESHOLD):
    """Detect if there's actual speech in the audio"""
    # Calculate RMS (Root Mean Square) to detect audio level
    rms = np.sqrt(np.mean(audio_data**2))
    return rms > threshold

def speak(text):
    """Improved text-to-speech with threading"""
    if not tts_engine or not text:
        print(f"AGENT: {text}")
        return
    
    def speak_async():
        try:
            print(f"AGENT: {text}")
            print("🔊 Speaking...")
            tts_engine.say(text)
            tts_engine.runAndWait()
            print("✅ Finished speaking.")
        except Exception as e:
            print(f"❌ TTS error: {e}")
    
    # Run TTS in a separate thread to avoid blocking
    speech_thread = threading.Thread(target=speak_async)
    speech_thread.daemon = True
    speech_thread.start()
    speech_thread.join(timeout=10)  # Max 10 seconds for speech

def transcribe_audio(audio_data):
    """Improved audio transcription with better error handling"""
    try:
        if not whisper_model:
            return ""
        
        # Check if there's actual speech
        if not detect_speech_in_audio(audio_data):
            return ""
        
        # Normalize audio
        audio_data = audio_data.flatten()
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        print("🔄 Transcribing...")
        segments, info = whisper_model.transcribe(
            audio_data, 
            beam_size=5, 
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        text = "".join(segment.text for segment in segments).strip()
        
        # Filter out common transcription artifacts
        artifacts = ["you", "thank you.", "thanks.", "okay.", "ok.", "um.", "uh.", "hmm."]
        if text.lower() in artifacts:
            return ""
        
        return text
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        return ""

def get_agent_response(user_input):
    """Get response from the agent with error handling"""
    try:
        if not agent_executor:
            return "I'm having trouble connecting to my brain right now."
        
        print("🤔 Thinking...")
        response = agent_executor.invoke({"input": user_input})
        agent_response = response.get("output", "I'm not sure how to respond to that.")
        
        # Check if it's a goodbye response
        if "goodbye" in agent_response.lower() or "have a great day" in agent_response.lower():
            return agent_response, True  # Signal to end conversation
        
        return agent_response, False
    except Exception as e:
        print(f"❌ Agent error: {e}")
        return "Sorry, I encountered an error while processing your request.", False

# 6. Main Application Loop

def main():
    """Improved main loop with better error handling"""
    if not whisper_model or not agent_executor:
        print("❌ Critical components failed to initialize. Exiting.")
        return
    
    print("\n🚀 --- Voice Agent Nova Activated ---")
    speak("Hello! I'm Nova, your voice assistant. How can I help you today?")
    
    print("⏳ Starting in 3 seconds...")
    time.sleep(3)
    
    consecutive_failures = 0
    max_failures = 5
    
    try:
        while True:
            try:
                # Record audio
                print(f"\n🎧 Listening for {INTERVAL} seconds... (Press Ctrl+C to exit)")
                audio_data = sd.rec(
                    int(INTERVAL * SAMPLE_RATE), 
                    samplerate=SAMPLE_RATE, 
                    channels=1, 
                    dtype='float32'
                )
                sd.wait()
                
                # Transcribe
                user_text = transcribe_audio(audio_data)
                
                if not user_text or len(user_text.strip()) < 2:
                    print("👂 (No clear speech detected)")
                    consecutive_failures += 1
                    
                    if consecutive_failures >= max_failures:
                        print("🔇 No speech detected for a while. Say something!")
                        speak("Are you still there? I haven't heard anything for a while.")
                        consecutive_failures = 0
                    
                    time.sleep(1)
                    continue
                
                consecutive_failures = 0  # Reset failure counter
                print(f"👤 YOU: {user_text}")
                
                # Get agent response
                agent_response, should_exit = get_agent_response(user_text)
                
                # Speak response
                speak(agent_response)
                
                # Check if we should exit
                if should_exit:
                    print("👋 Conversation ended.")
                    break
                
                # Brief pause before listening again
                time.sleep(2)
                
            except KeyboardInterrupt:
                print("\n\n👋 Shutting down agent...")
                speak("Goodbye! Have a great day!")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                consecutive_failures += 1
                
                if consecutive_failures >= 3:
                    print("💀 Too many errors. Exiting.")
                    speak("I'm experiencing technical difficulties. Goodbye!")
                    break
                
                time.sleep(2)
                
    except Exception as e:
        print(f"💀 Critical error in main loop: {e}")
    finally:
        if tts_engine:
            try:
                tts_engine.stop()
            except:
                pass
        print("🛑 Voice agent terminated.")

# 7. Audio Device Check
def check_audio_devices():
    """Check available audio devices"""
    try:
        print("\n🎤 Available Audio Devices:")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"  {i}: {device['name']} (Input)")
        print()
    except Exception as e:
        print(f"❌ Could not query audio devices: {e}")

if __name__ == "__main__":
    check_audio_devices()
    main()