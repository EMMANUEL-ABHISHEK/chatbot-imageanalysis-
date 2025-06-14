# 🤖 Conversational Recognition Chatbot

A powerful **multi-modal AI chatbot** with advanced image recognition, speech processing, and **Retrieval-Augmented Generation (RAG)** capabilities. Built with FastAPI, React, and cutting-edge AI models.

## 🧹 Project Status

✅ **Recently Cleaned & Optimized** (Latest Update)

- **Removed unnecessary files**: Cache files, redundant documentation, and unused modules
- **Streamlined dependencies**: Cleaned up requirements.txt with detailed comments
- **Optimized type checking**: Comprehensive pyproject.toml configuration
- **Enhanced maintainability**: Clean, well-documented codebase ready for production

## ✨ Features

### 🎯 **Core Capabilities**

- **💬 Text Chat**: Intelligent conversations using GPT-4, Mistral 7B, or local models
- **🖼️ Image Analysis**: Advanced image captioning and analysis using BLIP and CLIP models
- **🎤 Speech Recognition**: Convert audio to text and process voice commands
- **📄 PDF Processing**: Extract and analyze content from PDF documents
- **🔍 Multi-Modal Understanding**: Process text, images, and audio together seamlessly

### 🧠 **RAG System (Retrieval-Augmented Generation)**

- **📰 Real-time News Integration**: 30+ RSS feeds updated every 15 minutes
- **🔍 Semantic Search**: Find relevant information using vector similarity
- **⏰ Fresh Content**: 24-hour filtering for the latest news and information
- **📊 Smart Citations**: Proper source attribution with links and dates
- **🎯 Context-Aware Responses**: Answers grounded in current, factual information

### 🏗️ **Architecture**

- **🚀 FastAPI Backend**: High-performance REST API with automatic documentation
- **⚛️ React Frontend**: Modern, responsive user interface
- **🧮 SentenceTransformers**: Text embeddings using `all-MiniLM-L6-v2`
- **🤗 HuggingFace Models**: BLIP for image captioning, CLIP for multimodal understanding
- **📍 Pinecone Vector Database**: Serverless vector storage with Unix timestamp filtering
- **⚡ Groq API**: Ultra-fast LLM inference with Llama-3
- **🔄 Auto-Update System**: Background RSS ingestion every 15 minutes

## 🚀 Quick Start

### 📋 **Prerequisites**

- **Python 3.9+** (3.11 recommended)
- **Node.js 16+** (for React frontend)
- **Git** for cloning the repository

### 🔑 **Required API Keys** (Free Tiers Available)

- **🔥 Groq API Key** - For ultra-fast LLM inference ([Get Free Key](https://console.groq.com/))
- **📍 Pinecone API Key** - For vector database ([Get Free Key](https://www.pinecone.io/))
- **🤗 HuggingFace API Key** - For model access ([Get Free Key](https://huggingface.co/settings/tokens))
- **🤖 OpenAI API Key** - Optional, for GPT-4 access ([Get Key](https://platform.openai.com/api-keys))

### ⚡ **Installation**

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/conversational-recognition-chatbot.git
   cd conversational-recognition-chatbot
   ```

2. **Create and activate virtual environment:**

   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install Python dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Create a `.env` file in the project root:

   ```env
   # Required API Keys
   GROQ_API_KEY=your_groq_api_key_here
   PINECONE_API_KEY=your_pinecone_api_key_here
   PINECONE_INDEX_HOST=your_pinecone_index_host_here

   # Optional API Keys
   HF_API_KEY=your_huggingface_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here

   # Optional: Custom RSS feeds (uses default 30+ feeds if not specified)
   # RSS_FEEDS=https://example.com/feed1.rss,https://example.com/feed2.rss
   ```

5. **Set up Pinecone Index:**
   - Go to [Pinecone Console](https://app.pinecone.io/)
   - Create a new index with:
     - **Name**: `news-rag` (or your preferred name)
     - **Dimensions**: `384` (for all-MiniLM-L6-v2 embeddings)
     - **Metric**: `cosine`
     - **Cloud**: `AWS` (free tier)
   - Copy the index host URL to your `.env` file

## 🏃‍♂️ **Running the Application**

### 🖥️ **Backend (FastAPI)**

```bash
# Start the FastAPI server
uvicorn main:app --host 0.0.0.0 --port 3000 --reload

# The server will start at:
# 🌐 http://localhost:3000
# 📚 API Docs: http://localhost:3000/docs
# 🔧 ReDoc: http://localhost:3000/redoc
```

### ⚛️ **Frontend (React)** - Optional

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm start

# Frontend will start at:
# 🌐 http://localhost:3000
```

### 🧪 **Testing the Setup**

```bash
# Run the comprehensive test suite
python test_unix_timestamp_filtering.py

# Test RSS ingestion
curl -X POST "http://localhost:8000/rss/update"

# Test news search
curl -X GET "http://localhost:8000/news/search?q=artificial%20intelligence&k=3"
```

## 📡 **API Endpoints**

### 💬 **Chat & Conversation**

```http
POST /chat/text          # Text-only conversation
POST /chat/image         # Image analysis and description
POST /chat/multimodal    # Combined text + image processing
POST /chat/audio         # Audio-to-text + conversation
POST /chat/pdf           # PDF analysis and Q&A
```

### 🔍 **News & RAG System**

```http
GET  /news/search        # Semantic news search with date filtering
GET  /news/chat          # Chat with news context (RAG)
POST /rss/update         # Manual RSS feed update
GET  /rss/status         # Check RSS ingestion status
```

### 📊 **Utilities**

```http
GET  /health             # System health check
GET  /stats              # Usage statistics
POST /embed              # Generate text embeddings
GET  /docs               # Interactive API documentation
```

### 🎯 **Example API Calls**

**Text Chat:**

```bash
curl -X POST "http://localhost:8000/chat/text" \
  -H "Content-Type: application/json" \
  -d '{"message": "What is artificial intelligence?", "session_id": "user123"}'
```

**News Search:**

```bash
curl -X GET "http://localhost:8000/news/search?q=climate%20change&k=5&days_filter=1"
```

**Image Analysis:**

```bash
curl -X POST "http://localhost:8000/chat/image" \
  -F "image=@path/to/your/image.jpg" \
  -F "session_id=user123"
```

## 🧠 **How RAG Works**

### 📊 **RAG Pipeline Overview**

```
User Query → Embedding → Vector Search → Context Retrieval → LLM Generation → Response
     ↓            ↓            ↓              ↓               ↓            ↓
  "AI news"  → [0.1,0.3...] → Pinecone → Recent articles → Groq/GPT-4 → Informed answer
```

### 🔄 **Step-by-Step Process**

1. **📝 Query Processing**: User sends a question about current events
2. **🧮 Embedding Generation**: Query converted to 384-dimensional vector using SentenceTransformers
3. **🔍 Semantic Search**: Vector similarity search in Pinecone with 24-hour freshness filter
4. **📰 Context Retrieval**: Top-k relevant news articles retrieved with metadata
5. **🤖 LLM Generation**: Context + query sent to Groq Llama-3 or GPT-4
6. **✨ Response**: AI generates informed response with proper citations

### ⏰ **Fresh Content Guarantee**

- **24-hour filtering** ensures only recent news
- **Unix timestamp filtering** for precise date comparisons
- **Automatic deduplication** removes similar articles
- **Source attribution** with links and publication dates

## 🎭 **Multi-Modal Processing**

### 🖼️ **Image + Text Understanding**

```
Image Upload → BLIP Captioning → Text Combination → LLM Processing → Contextual Response
     ↓              ↓                ↓                ↓               ↓
  photo.jpg → "A cat on a sofa" → "What breed is this cat?" → GPT-4 → "Based on the image..."
```

### 🎤 **Audio Processing Pipeline**

```
Audio File → Speech Recognition → Text Processing → Response Generation
     ↓              ↓                ↓               ↓
  voice.wav → "What's the weather?" → Enhanced Chatbot → "Based on current data..."
```

### 📄 **PDF Analysis**

```
PDF Upload → Text Extraction → Chunking → Embedding → Storage → Q&A Ready
     ↓            ↓            ↓         ↓         ↓         ↓
  document.pdf → Raw text → Paragraphs → Vectors → Pinecone → Ask questions
```

## 🛠️ **Advanced Configuration**

### 🔧 **Environment Variables**

```env
# Model Selection
USE_GPT4=true                    # Use GPT-4 instead of Mistral
USE_LOCAL_MODELS=false           # Use local Hugging Face models

# RAG Configuration
RAG_TOP_K=5                      # Number of articles to retrieve
RAG_DAYS_FILTER=1               # Days back to search (1-30)
AUTO_UPDATE_INTERVAL=900         # RSS update interval (seconds)

# Performance Tuning
MAX_TOKENS=1000                  # Maximum response length
TEMPERATURE=0.7                  # Response creativity (0.0-1.0)
EMBEDDING_BATCH_SIZE=32          # Batch size for embeddings
```

### 📊 **Monitoring & Logging**

```bash
# Check system status
curl http://localhost:8000/health

# View RSS ingestion logs
tail -f logs/rss_ingestion.log

# Monitor API usage
curl http://localhost:8000/stats
```

## 🚨 **Troubleshooting**

### ❌ **Common Issues**

**1. Pinecone Connection Failed**

```bash
# Check your API key and index host
echo $PINECONE_API_KEY
echo $PINECONE_INDEX_HOST
```

**2. No News Results**

```bash
# Manually trigger RSS update
curl -X POST http://localhost:8000/rss/update
```

**3. Model Loading Errors**

```bash
# Check GPU memory
nvidia-smi  # For CUDA users
# Or force CPU mode
export CUDA_VISIBLE_DEVICES=""
```

### 🔍 **Debug Mode**

```bash
# Run with debug logging
export LOG_LEVEL=DEBUG
uvicorn main:app --log-level debug
```

## 📈 **Performance & Scaling**

### ⚡ **Optimization Tips**

- **Use Groq API** for fastest LLM inference (10x faster than OpenAI)
- **Enable GPU** for local model inference (CUDA recommended)
- **Batch processing** for multiple requests
- **Caching** for frequently asked questions

### 📊 **Expected Performance**

- **Text Chat**: ~1-2 seconds response time
- **Image Analysis**: ~3-5 seconds (including BLIP processing)
- **RAG Queries**: ~2-4 seconds (including vector search)
- **RSS Ingestion**: ~30-60 seconds for all feeds

## 🤝 **Contributing**

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### 🐛 **Reporting Issues**

- Use GitHub Issues for bug reports
- Include system info, error logs, and reproduction steps
- Check existing issues before creating new ones

### 💡 **Feature Requests**

- Describe the use case and expected behavior
- Consider implementation complexity
- Discuss in GitHub Discussions first

## 📄 **License**

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- **Groq** for ultra-fast LLM inference
- **Pinecone** for vector database infrastructure
- **Hugging Face** for open-source models
- **FastAPI** for the excellent web framework
- **React** for the frontend framework

---

**⭐ If you find this project helpful, please give it a star on GitHub!**

**🔗 Links:**

- [📚 Documentation](https://github.com/yourusername/conversational-recognition-chatbot/wiki)
- [🐛 Issues](https://github.com/yourusername/conversational-recognition-chatbot/issues)
- [💬 Discussions](https://github.com/yourusername/conversational-recognition-chatbot/discussions)
- [🚀 Releases](https://github.com/yourusername/conversational-recognition-chatbot/releases)
