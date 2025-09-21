# NovaChat 🚀

**Multimodal AI Agent for Enterprise Conversations**

NovaChat is an intelligent conversation analysis platform that transforms enterprise call transcripts into actionable insights through AI-powered analysis, interactive visualizations, and comprehensive reporting.

## 🌟 Features

### 🎯 **Intelligent Query Routing**
- **Chain-of-Thought Reasoning**: Advanced LLM router that understands context and conversation history
- **Multi-Tool Selection**: Automatically chooses the best analysis tool based on query intent
- **Context Awareness**: References previous responses and conversation flow

### 📊 **Advanced Analytics & Visualizations**
- **Speaker Activity Analysis**: Track speaking time, turn counts, and engagement patterns
- **Sentiment Trend Analysis**: Monitor emotional progression throughout conversations
- **Topic Categorization**: Automatically categorize conversation segments by themes
- **Custom Chart Generation**: Create bar charts, pie charts, and line graphs on demand

### 📄 **Comprehensive Reporting**
- **PDF Report Generation**: Professional downloadable reports with detailed analysis
- **Audio Summaries**: Text-to-speech narration of key insights
- **Interactive Dashboards**: Real-time conversation profiling and metrics

### 🔊 **Audio Processing**
- **Multi-format Support**: Process MP3, WAV, M4A audio files
- **Automatic Transcription**: Convert speech to text with speaker identification
- **Real-time Analysis**: Instant insights from uploaded conversations

## 🏗️ **Architecture**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │───▶│  Query Router    │───▶│  Analysis Tools │
│                 │    │ (Chain-of-Thought)│    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Audio Processor │    │   ChromaDB       │    │  Google Gemini  │
│                 │    │   Vector Store   │    │      LLM        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 **Quick Start**

### Prerequisites
- Python 3.8+
- Google Gemini API Key
- ffmpeg (for audio processing)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/NovaChat.git
   cd NovaChat
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   # Create .env file
   echo "GOOGLE_API_KEY=your_gemini_api_key_here" > .env
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open in browser**
   Navigate to `http://localhost:8501`

## 📁 **Project Structure**

```
NovaChat/
├── app.py                      # Main Streamlit application
├── query_engine.py            # Intelligent query routing system
├── outputs.py                 # Analysis tools and LLM integration
├── pdf_builder.py             # PDF report generation
├── normalization.py           # Audio processing and transcription
├── ingest.py                  # Vector database ingestion
├── profiler.py               # Conversation profiling
├── transcript_parser.py       # JSON transcript parsing
├── config.py                 # Configuration settings
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables
├── chroma_db/               # Vector database storage
├── temp_uploads/            # Temporary file storage
└── README.md                # This file
```

## 🛠️ **Core Components**

### **Query Router** (`query_engine.py`)
Smart routing system that uses chain-of-thought reasoning to select the best analysis tool:

```python
# Example: Intelligent routing based on query intent
query = "Show me how Michelle responded to fee discussions"
# Router analyzes: intent → context → tool selection → execution
```

### **Analysis Tools** (`outputs.py`)
Specialized functions for different types of analysis:

- `build_holistic_analysis_chart()` - Complex multi-dimensional analysis
- `build_speaker_activity_chart()` - Speaker engagement metrics
- `build_sentiment_trend_chart()` - Emotional progression tracking
- `build_pdf_response()` - Comprehensive report generation

### **Audio Processing** (`normalization.py`)
Handles audio file conversion and transcription:

```python
# Automatic transcription with speaker identification
transcribe_audio("conversation.mp3") → structured_transcript.json
```

## 💡 **Usage Examples**

### **Basic Conversation Analysis**
```
Upload audio → "Summarize the main discussion points"
```

### **Advanced Analytics**
```
"Create a breakdown chart showing time spent on fees vs. program details"
"Analyze Michelle's engagement level throughout the call"
"Generate a sentiment trend for Speaker0"
```

### **Report Generation**
```
"Generate a comprehensive report analyzing Michelle's concerns and responses"
"Create a PDF summary of this sales conversation"
```

### **Speaker Analysis**
```
"Who talked the most and for how long?"
"Show me the number of times each speaker contributed"
"Compare speaking patterns between participants"
```

## 🔧 **Configuration**

### **Environment Variables** (`.env`)
```bash
GOOGLE_API_KEY=your_gemini_api_key_here
CHROMA_DB_PATH=./chroma_db
COLLECTION_NAME=conversation_segments
EMBED_MODEL_NAME=all-MiniLM-L6-v2
```

### **Model Configuration** (`config.py`)
```python
MODEL_ID = "gemini-2.5-flash"  # Google Gemini model
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # Sentence transformer
CHROMA_DB_PATH = "./chroma_db"
```

## 📊 **Supported File Formats**

### **Audio Files**
- MP3, WAV, M4A, FLAC
- Automatic conversion to WAV for processing

### **Transcript Files**
- JSON format with speaker identification
- Structured conversation segments with timestamps

### **Output Formats**
- Interactive charts (matplotlib)
- PDF reports (comprehensive analysis)
- MP3 audio summaries (text-to-speech)

## 🤖 **AI Models & Integration**

- **Google Gemini 2.5 Flash**: Primary LLM for analysis and reasoning
- **Sentence Transformers**: Vector embeddings for semantic search
- **CardiffNLP RoBERTa**: Sentiment analysis pipeline
- **Google Text-to-Speech**: Audio summary generation

## 🎯 **Key Features in Detail**

### **Chain-of-Thought Routing**
```
Step 1: Analyze query intent (chart vs. report vs. summary)
Step 2: Consider conversation history and context
Step 3: Reference previous responses if applicable
Step 4: Select optimal tool for task
Step 5: Execute with full context awareness
```

### **Conversation Profiling**
- Automatic detection of conversation type (Sales, Support, etc.)
- Key entity extraction (names, products, companies)
- Context preservation across queries

### **Vector-Based Retrieval**
- ChromaDB integration for semantic search
- Contextual segment retrieval
- Relevant information extraction for queries

## 🚨 **Troubleshooting**

### **Common Issues**

1. **API Key Errors**
   ```bash
   # Verify .env file exists and contains valid API key
   cat .env
   ```

2. **Audio Processing Fails**
   ```bash
   # Install ffmpeg
   # Ubuntu/Debian: sudo apt install ffmpeg
   # macOS: brew install ffmpeg
   # Windows: Download from ffmpeg.org
   ```

3. **Dependencies Missing**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

## 🖥️ Presentation & Demo

- **Project Presentation (PPT)**: [View Google Slides](https://docs.google.com/presentation/d/1UahU0C1VLEXhlkRd8BtjXnSG6GI2lo-1/edit?usp=drive_link&ouid=108092071704659088422&rtpof=true&sd=true)  
- **Demo Video**: [Watch Demo](https://drive.google.com/file/d/1djk6g8MKvJQDLQVWlRQXC8SarzewL5gr/view?usp=drive_link)


## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- Google Gemini for advanced language model capabilities
- ChromaDB for vector database functionality
- Streamlit for the interactive web interface
- HuggingFace for pre-trained models and pipelines

---

**Built with ❤️ for enterprise conversation intelligence**

For questions or support, please open an issue or contact the development team.
