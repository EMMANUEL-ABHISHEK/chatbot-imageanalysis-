import os
import hashlib
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import threading

import feedparser
from dotenv import load_dotenv
import groq
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import PyPDF2
import speech_recognition as sr
from pydub import AudioSegment
import re
import html
from html.parser import HTMLParser

class HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text: List[str] = []

    def handle_data(self, data: str) -> None:
        self.text.append(data)

    def get_data(self) -> str:
        return ''.join(self.text)

class MinimalRAGService:
    def __init__(self):
        """Initialize the RAG service with all required components"""
        self.log = logging.getLogger("rag")
        warnings.simplefilter("ignore", RuntimeWarning)
        
        AudioSegment.converter = "ffmpeg"
        load_dotenv()
        
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        self.GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        self.PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        self.PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")

        if not (self.PINECONE_API_KEY and self.PINECONE_INDEX_HOST):
            self.log.error("❌ Missing required environment variables")
            raise RuntimeError("Missing required environment variables for RAG service.")

        # Initialize Gemini (Primary)
        self.gemini_client = None
        self.gemini_available = False
        if self.GEMINI_API_KEY:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.GEMINI_API_KEY)
                self.gemini_client = genai.GenerativeModel('gemini-1.5-flash')
                self.gemini_available = True
                self.log.info("✅ Gemini API initialized (Primary)")
            except Exception as e:
                self.log.warning(f"❌ Gemini initialization failed: {e}")

        # Initialize Groq (Backup)
        self.groq_client = None
        self.groq_available = False
        if self.GROQ_API_KEY:
            try:
                self.groq_client = groq.Client(api_key=self.GROQ_API_KEY)
                self.groq_available = True
                self.log.info("✅ Groq API initialized (Backup)")
            except Exception as e:
                self.log.warning(f"❌ Groq initialization failed: {e}")

        if not (self.gemini_available or self.groq_available):
            self.log.error("❌ No AI service available (neither Gemini nor Groq)")
            raise RuntimeError("No AI service available for RAG.")
        
        try:
            self.pc = Pinecone(api_key=self.PINECONE_API_KEY)
            self.index = self.pc.Index(host=self.PINECONE_INDEX_HOST)
            self.log.info("✅ Connected to Pinecone index")
        except Exception as e:
            self.log.exception("❌ Pinecone connection failed")
            raise RuntimeError(f"Failed to connect to Pinecone: {str(e)}")
        
        try:
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            self.log.info("✅ Sentence transformer loaded")
        except Exception as e:
            self.log.exception("❌ Failed to load sentence transformer")
            raise RuntimeError(f"Failed to load sentence transformer: {str(e)}")
        
        self.md5 = lambda s: hashlib.md5(s.encode()).hexdigest()
        self.auto_update_enabled = False
        self.update_thread = None
        self.last_update = None

        self.FEEDS = [
            "http://feeds.bbci.co.uk/news/rss.xml",
            "https://www.theguardian.com/world/rss",
            "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
            "https://feeds.reuters.com/reuters/topNews",
            "https://feeds.skynews.com/feeds/rss/world.xml",
            "https://www.aljazeera.com/xml/rss/all.xml",
            "https://feeds.feedburner.com/ndtvnews-top-stories",
            "https://www.hindustantimes.com/rss/topnews/rssfeed.xml",
            "https://techcrunch.com/feed/",
            "https://www.theverge.com/rss/index.xml",
            "https://www.nature.com/nature/articles?type=article&format=rss",
            "https://www.sciencedaily.com/rss/top/science.xml",
            "https://www.economist.com/international/rss.xml",
            "https://rss.nytimes.com/services/xml/rss/nyt/Politics.xml",
            "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en",
        ]
        
        self.status = "initialized"
        self.log.info("✅ MinimalRAGService initialized successfully")
        self.start_auto_update()

    def _sanitize_text(self, text: str) -> str:
        """Sanitize text by removing HTML tags and normalizing content"""
        if not text or not isinstance(text, str):
            return ""
        
        try:
            text = html.unescape(text)
            stripper = HTMLStripper()
            stripper.feed(text)
            text = stripper.get_data()
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'<!\[CDATA\[.*?\]\]>', '', text, flags=re.DOTALL)
            return text.strip()
        except Exception as e:
            self.log.warning(f"Text sanitization failed: {str(e)}")
            return str(text).strip() if text else ""

    def _validate_feed_entry(self, entry: Any) -> Optional[Dict[str, str]]:
        """Validate and sanitize a feed entry"""
        try:
            title = getattr(entry, 'title', '') or getattr(entry, 'title_detail', {}).get('value', '')
            summary = getattr(entry, 'summary', '') or getattr(entry, 'description', '')
            link = getattr(entry, 'link', '') or getattr(entry, 'id', '')
            
            title = self._sanitize_text(title)
            summary = self._sanitize_text(summary)
            link = str(link).strip() if link else ""
            
            if not title or len(title) < 5 or not summary or len(summary) < 20:
                return None
            
            if not link or not link.startswith(('http://', 'https://')):
                return None
            
            if len(title) > 500:
                title = title[:500] + "..."
            if len(summary) > 2000:
                summary = summary[:2000] + "..."
            
            return {'title': title, 'summary': summary, 'link': link}
            
        except Exception as e:
            self.log.warning(f"Feed entry validation failed: {str(e)}")
            return None

    def ingest_rss(self) -> Dict[str, Any]:
        """Ingest RSS feeds into Pinecone vector database"""
        try:
            added = 0
            errors = 0
            
            for url in self.FEEDS:
                try:
                    self.log.info(f"📰 Processing RSS feed: {url}")
                    feed = feedparser.parse(url)
                    
                    for entry in feed.entries:
                        try:
                            validated_entry = self._validate_feed_entry(entry)
                            if not validated_entry:
                                continue
                            
                            title = validated_entry['title']
                            summary = validated_entry['summary']
                            link = validated_entry['link']
                            
                            if len(summary) < 20:
                                continue
                            
                            vec = self.embedder.encode(summary).tolist()
                            current_timestamp = int(datetime.now().timestamp())
                            
                            self.index.upsert([(self.md5(link), vec, {
                                "title": title,
                                "summary": summary,
                                "link": link,
                                "source": url,
                                "published_timestamp": current_timestamp,
                                "published_date": datetime.now().isoformat()
                            })])
                            added += 1
                            
                        except Exception as e:
                            self.log.warning(f"⚠️ Failed to process entry: {str(e)}")
                            errors += 1
                            continue
                            
                except Exception as e:
                    self.log.error(f"❌ Failed to process feed {url}: {str(e)}")
                    errors += 1
                    continue
            
            stats = self.index.describe_index_stats()
            total_vectors = stats.get('total_vector_count', 0)
            
            self.log.info(f"📰 RSS ingestion complete: {added} items added, {errors} errors")
            
            return {
                "success": True,
                "articles_processed": added,
                "errors": errors,
                "total_vectors": total_vectors,
                "message": f"Successfully processed {added} articles"
            }
            
        except Exception as e:
            self.log.exception("❌ RSS ingestion failed")
            return {
                "success": False,
                "error": str(e),
                "message": "RSS ingestion failed"
            }

    def search_news(self, query: str, k: int = 5, days_filter: int = 7) -> List[Dict[str, Any]]:
        """Search for relevant news articles using semantic similarity"""
        try:
            query_vector = self.embedder.encode(query).tolist()
            cutoff_time = datetime.now() - timedelta(days=days_filter)
            cutoff_timestamp = int(cutoff_time.timestamp())
            
            results = self.index.query(
                vector=query_vector,
                top_k=k * 2,
                include_metadata=True,
                filter={"published_timestamp": {"$gte": cutoff_timestamp}}
            )
            
            if not results.matches:
                self.log.warning(f"No recent articles found for query: {query}")
                return []
            
            articles = []
            seen_titles: set[str] = set()
            
            for match in results.matches:
                title = match.metadata.get('title', 'No Title')
                title_lower = title.lower().strip()
                if title_lower not in seen_titles and len(title_lower) > 10:
                    seen_titles.add(title_lower)
                    articles.append({
                        "title": title,
                        "summary": match.metadata.get('summary', 'No Summary'),
                        "link": match.metadata.get('link', ''),
                        "source": match.metadata.get('source', ''),
                        "published_date": match.metadata.get('published_date', ''),
                        "score": float(match.score)
                    })
                    
                    if len(articles) >= k:
                        break
            
            self.log.info(f"Found {len(articles)} unique articles for query: {query}")
            return articles
            
        except Exception as e:
            self.log.exception(f"❌ News search failed for query: {query}")
            return []

    def ask_ai(self, prompt: str, max_tokens: int = 1000, query: str = "") -> str:
        """Generate response using AI (Gemini primary, Groq fallback)"""
        try:
            if not prompt or not isinstance(prompt, str):
                return "Sorry, I received an invalid prompt."

            if len(prompt) > 8000:
                prompt = prompt[:8000] + "\n\n[Content truncated for length]"

            # Try Gemini first (Primary)
            if self.gemini_available:
                try:
                    response = self.gemini_client.generate_content(
                        prompt,
                        generation_config={
                            'temperature': 0.7,
                            'max_output_tokens': max_tokens,
                            'top_p': 0.9,
                        }
                    )

                    if response and response.text:
                        self.log.info("✅ Gemini response generated successfully")
                        return response.text.strip()
                    else:
                        self.log.warning("❌ Gemini returned empty response")
                except Exception as e:
                    self.log.warning(f"❌ Gemini failed: {e}")

            # Fallback to Groq
            if self.groq_available:
                try:
                    response = self.groq_client.chat.completions.create(
                        model="llama3-8b-8192",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=max_tokens,
                        temperature=0.7,
                        top_p=0.9,
                        stream=False
                    )

                    if response and response.choices and response.choices[0].message:
                        self.log.info("✅ Groq response generated successfully (fallback)")
                        return response.choices[0].message.content.strip()
                    else:
                        self.log.warning("❌ Groq returned invalid response")
                except Exception as e:
                    self.log.warning(f"❌ Groq failed: {e}")

            return "Sorry, I'm having trouble generating a response right now."

        except Exception as e:
            self.log.exception("❌ AI API call failed")
            return f"Sorry, I encountered an error generating a response: {str(e)}"

    def ask_groq(self, prompt: str, max_tokens: int = 1000, query: str = "") -> str:
        """Legacy method for backward compatibility - redirects to ask_ai"""
        return self.ask_ai(prompt, max_tokens, query)

    def analyse_pdf(self, path: Path) -> str:
        """Analyze PDF content using Groq"""
        try:
            text_parts = []
            with open(str(path), 'rb') as file:
                reader = PyPDF2.PdfReader(file)

                for i, page in enumerate(reader.pages[:5]):
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)

            full_text = "\n".join(text_parts)

            if not full_text.strip():
                return "PDF contained no extractable text."

            prompt = f"Summarize this PDF content briefly and clearly:\n\n{full_text[:8000]}"
            return self.ask_groq(prompt)

        except Exception as e:
            self.log.exception(f"❌ PDF analysis failed for {path}")
            return f"Failed to analyze PDF: {str(e)}"

    def analyse_pdf_comprehensive(self, pdf_text: str) -> str:
        """Provide comprehensive PDF analysis with structured format"""
        try:
            if not pdf_text.strip():
                return "PDF contained no extractable text."

            # Create a comprehensive analysis prompt
            prompt = f"""Please provide a comprehensive, structured analysis of this PDF document. Format your response with clear headings, subheadings, and bullet points to cover every topic present in the document.

Structure your analysis as follows:

# 📄 DOCUMENT ANALYSIS

## 📋 Executive Summary
[Provide a brief 2-3 sentence overview of the document]

## 🎯 Main Topics Covered
[List the primary topics/themes with bullet points]

## 📖 Detailed Content Analysis

### [Topic 1 Heading]
- Key point 1
- Key point 2
- Key point 3

### [Topic 2 Heading]
- Key point 1
- Key point 2
- Key point 3

[Continue for all major topics found]

## 🔍 Key Insights & Findings
- Important insight 1
- Important insight 2
- Important insight 3

## 📊 Important Data/Statistics
[If any numbers, statistics, or data points are mentioned]

## 💡 Conclusions & Recommendations
[If the document contains conclusions or recommendations]

## 🏷️ Keywords & Concepts
[List important terms and concepts]

Please ensure you cover EVERY topic and section present in the document. Be thorough and comprehensive.

Document Content:
{pdf_text[:6000]}"""

            return self.ask_ai(prompt, max_tokens=1500)

        except Exception as e:
            self.log.exception(f"❌ Comprehensive PDF analysis failed")
            return f"Failed to analyze PDF comprehensively: {str(e)}"

    def google_stt(self, wav_path: Path) -> Optional[str]:
        """Convert speech to text using Google Speech Recognition"""
        try:
            recognizer = sr.Recognizer()

            with sr.AudioFile(str(wav_path)) as source:
                audio = recognizer.record(source)
                text = recognizer.recognize_google(audio)
                self.log.info(f"🗣️ STT successful: {text[:50]}...")
                return text

        except Exception as e:
            self.log.warning(f"🗣️ STT failed: {str(e)}")
            return None

    def chat_with_news(self, query: str) -> str:
        """Chat with news context using RAG"""
        try:
            articles = self.search_news(query, k=5, days_filter=1)

            if not articles:
                articles = self.search_news(query, k=5, days_filter=7)

            if not articles:
                return self.ask_ai(f"Please answer this question: {query}")

            context = ""
            for i, article in enumerate(articles[:3], 1):
                context += f"{i}. {article['title']}\n   {article['summary'][:200]}...\n   Source: {article['source']}\n   Date: {article['published_date']}\n\n"

            prompt = f"""Based on the following recent news articles, please answer the user's question: "{query}"

Recent News Context:
{context}

Please provide a comprehensive answer based on the news context above. Include relevant details and cite the sources when appropriate."""

            return self.ask_ai(prompt, max_tokens=800, query=query)

        except Exception as e:
            self.log.exception(f"❌ Chat with news failed for query: {query}")
            return f"Sorry, I encountered an error while searching for news: {str(e)}"

    def get_status(self) -> Dict[str, Any]:
        """Get service status and statistics"""
        try:
            stats = self.index.describe_index_stats()
            return {
                "status": self.status,
                "total_vectors": stats.get('total_vector_count', 0),
                "last_update": self.last_update.isoformat() if self.last_update else None,
                "auto_update_enabled": self.auto_update_enabled,
                "groq_available": bool(self.GROQ_API_KEY),
                "pinecone_available": True,
                "embedder_available": True
            }
        except Exception:
            return {
                "status": "error",
                "groq_available": bool(self.GROQ_API_KEY),
                "pinecone_available": False,
                "embedder_available": True
            }

    def start_auto_update(self):
        """Start automatic RSS feed updates every 15 minutes"""
        try:
            def update_worker():
                while self.auto_update_enabled:
                    try:
                        self.log.info("🔄 Starting automatic RSS feed update...")
                        result = self.ingest_rss()
                        self.last_update = datetime.now()
                        self.log.info(f"✅ Auto-update completed: {result.get('articles_processed', 0)} articles")
                    except Exception as e:
                        self.log.error(f"❌ Auto-update failed: {str(e)}")

                    import time
                    time.sleep(900)  # 15 minutes

            self.auto_update_enabled = True
            self.update_thread = threading.Thread(target=update_worker, daemon=True)
            self.update_thread.start()
            self.log.info("🚀 Auto-update thread started (15-minute intervals)")

        except Exception as e:
            self.log.error(f"❌ Failed to start auto-update: {str(e)}")

    def stop_auto_update(self):
        """Stop automatic RSS feed updates"""
        self.auto_update_enabled = False
        if self.update_thread:
            self.update_thread.join(timeout=5)
        self.log.info("🛑 Auto-update stopped")

try:
    rag_service = MinimalRAGService()
except Exception as e:
    logging.getLogger("rag").critical(f"Failed to initialize RAG service: {e}")

    class FallbackRAGService:
        def ingest_rss(self) -> Dict[str, Any]:
            return {"success": False, "error": "RAG service unavailable"}

        def search_news(self, query: str, k: int = 5, days_filter: int = 7) -> List[Dict[str, Any]]:
            return []

        def ask_groq(self, prompt: str, max_tokens: int = 1000, query: str = "") -> str:
            return "RAG service is currently unavailable."

        def analyse_pdf(self, path: Path) -> str:
            return "PDF analysis is currently unavailable."

        def analyse_pdf_comprehensive(self, pdf_text: str) -> str:
            return "Comprehensive PDF analysis is currently unavailable."

        def google_stt(self, wav_path: Path) -> Optional[str]:
            return None

        def chat_with_news(self, query: str) -> str:
            return "News chat is currently unavailable."

        def get_status(self) -> Dict[str, Any]:
            return {"status": "unavailable"}

    rag_service = FallbackRAGService()
