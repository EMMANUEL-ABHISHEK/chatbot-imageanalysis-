from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
from PIL import Image
import io
import os
import logging
import uuid
import PyPDF2
from datetime import datetime

# Direct imports - simplified approach
from model_service import model_service
from minimal_rag_service import rag_service

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("direct_chatbot")

# FastAPI app with direct configuration
app = FastAPI(
    title="Direct Image Recognition Chatbot",
    description="Streamlined chatbot with Gemini Vision + BLIP for image analysis",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files for images
os.makedirs("./generated_images", exist_ok=True)
app.mount("/images", StaticFiles(directory="./generated_images"), name="images")

# Simple session storage
sessions = {}

# Direct response models
class TabContext(BaseModel):
    mode: str = "general"
    hasUploadedFile: bool = False
    fileType: Optional[str] = None
    fileName: Optional[str] = None
    uploadedAt: Optional[str] = None

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    mode: str = "general"  # "image", "pdf", "rag", "general"
    tab_context: Optional[TabContext] = None

class ChatResponse(BaseModel):
    response: str
    confidence: float
    session_id: str
    error: bool = False

class ImageResponse(BaseModel):
    caption: str
    detected_elements: List[str]
    session_id: str
    confidence: float
    error: bool = False

class PDFResponse(BaseModel):
    summary: str
    session_id: str
    pages_processed: int
    error: bool = False

# Direct session management
def get_session_id(session_id: Optional[str] = None) -> str:
    """Get or create session ID"""
    if not session_id:
        session_id = str(uuid.uuid4())
    
    if session_id not in sessions:
        sessions[session_id] = {
            "pdf_content": None,
            "last_image_caption": None,
            "last_image_attributes": []
        }
    
    return session_id

# Direct processing functions
def process_text_direct(message: str, session_id: str, mode: str = "general") -> tuple:
    """Direct text processing with explicit mode handling"""
    session = sessions.get(session_id, {})

    # Check what content is available
    has_pdf = bool(session.get("pdf_content"))
    has_image = bool(session.get("last_image_caption"))

    logger.info(f"Processing text - Mode: {mode}, Has PDF: {has_pdf}, Has Image: {has_image}")

    # EXPLICIT MODE HANDLING - Frontend determines the mode
    # 1. If mode is explicitly "pdf", use PDF processing
    if mode == "pdf" and has_pdf:
        logger.info("Using explicit PDF mode")
        # PDF processing handled below

    # 2. If mode is explicitly "image", use image processing
    elif mode == "image" and has_image:
        logger.info("Using explicit image mode")
        # Image processing handled below

    # 3. If mode is explicitly "rag", use RAG processing
    elif mode == "rag":
        logger.info("Using explicit RAG mode")
        # RAG processing handled below

    # 4. Auto-detection for "general" mode only
    elif mode == "general":
        logger.info("Auto-detecting mode for general request")
        # Auto-detect based on content and keywords
        pdf_keywords = ["document", "pdf", "page", "text", "content", "summary", "report", "analysis", "findings", "recommendations", "conclusion"]
        if has_pdf and any(word in message.lower() for word in pdf_keywords):
            mode = "pdf"
            logger.info("Auto-detected PDF mode based on keywords")
        elif has_pdf:
            mode = "pdf"  # Default to PDF if available
            logger.info("Auto-detected PDF mode (default with PDF content)")
        elif has_image and any(word in message.lower() for word in ["image", "picture", "color", "see", "visual", "photo"]):
            mode = "image"
            logger.info("Auto-detected image mode based on keywords")
        elif any(word in message.lower() for word in ["news", "current", "today", "recent", "happening"]):
            mode = "rag"
            logger.info("Auto-detected RAG mode based on keywords")

    if mode == "pdf" and has_pdf:
        # Enhanced PDF processing with better context
        pdf_content = session['pdf_content']
        pdf_summary = session.get('pdf_summary', '')

        # Create comprehensive prompt for PDF questions
        prompt = f"""You are analyzing a PDF document. Here is the content and summary:

PDF SUMMARY:
{pdf_summary}

PDF CONTENT (First 3000 characters):
{pdf_content[:3000]}

USER QUESTION: {message}

Please provide a detailed, helpful answer based on the PDF content above. If the question asks for specific information, quote relevant parts from the document. If asking for analysis or interpretation, provide thoughtful insights based on the content."""

        try:
            response = rag_service.ask_ai(prompt, max_tokens=800)
            return response, 0.95
        except:
            return "I couldn't process your PDF question. Please try again.", 0.5

    elif mode == "rag":
        try:
            response = rag_service.chat_with_news(message)
            return response, 0.9
        except:
            return "I couldn't access news data. Please try again.", 0.5

    elif mode == "image" and has_image:
        caption = session["last_image_caption"]
        attributes = session.get("last_image_attributes", [])

        # Enhanced image context processing
        colors = [attr.get('label', '') for attr in attributes if attr.get('type') == 'color']
        objects = [attr.get('label', '') for attr in attributes if attr.get('type') == 'object']

        if "color" in message.lower():
            return f"Colors in the image: {', '.join(colors) if colors else 'various colors'}. {caption}", 0.9
        elif any(word in message.lower() for word in ["what", "describe", "see", "elements"]):
            all_elements = colors + objects
            return f"I can see: {', '.join(all_elements[:5]) if all_elements else 'various elements'}. {caption}", 0.9
        else:
            return f"About the image: {caption}", 0.8

    else:
        # Fallback to general processing
        try:
            result = model_service.process_text(message, session_id)
            if isinstance(result, tuple):
                return result
            elif isinstance(result, dict):
                return result.get("response", "I'm not sure how to respond."), result.get("confidence", 0.7)
            else:
                return str(result), 0.7
        except:
            return "I'm having trouble processing your message. Please try again.", 0.5

@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "Direct Image Recognition Chatbot API",
        "version": "2.0.0",
        "active_sessions": len(sessions)
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Direct chat endpoint with enhanced mode detection"""
    try:
        received_session_id = request.session_id
        session_id = get_session_id(request.session_id)

        # Determine the correct mode from tab_context or request.mode
        mode = request.mode
        if request.tab_context and request.tab_context.mode:
            mode = request.tab_context.mode
            logger.info(f"Using tab_context mode: {mode}")

        logger.info(f"Chat request: {request.message} (mode: {mode}, received_session: {received_session_id}, using_session: {session_id})")

        # For PDF mode, ensure we have PDF content in session
        if mode == "pdf":
            session = sessions.get(session_id, {})
            if not session.get("pdf_content"):
                logger.warning(f"PDF mode requested but no PDF content in session {session_id}")
                available_pdf_sessions = [sid for sid, data in sessions.items() if data.get('pdf_content')]
                logger.info(f"Available sessions with PDF content: {available_pdf_sessions}")

                # Provide helpful error message with debugging info
                error_msg = f"❌ **No PDF Found in Current Session**\n\n"
                error_msg += f"Current session: `{session_id}`\n"
                error_msg += f"Sessions with PDF content: {len(available_pdf_sessions)}\n\n"
                error_msg += "**Solutions:**\n"
                error_msg += "1. Upload a PDF document first\n"
                error_msg += "2. Make sure you're in PDF mode when asking questions\n"
                error_msg += "3. Try refreshing the page if the issue persists\n\n"
                if available_pdf_sessions:
                    error_msg += f"**Debug Info:** PDF content found in sessions: {available_pdf_sessions[:3]}"

                return ChatResponse(
                    response=error_msg,
                    confidence=0.8,
                    session_id=session_id
                )

        response, confidence = process_text_direct(request.message, session_id, mode)

        return ChatResponse(
            response=response,
            confidence=confidence,
            session_id=session_id
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        session_id = get_session_id(request.session_id)
        return ChatResponse(
            response="I encountered an error processing your message.",
            confidence=0.0,
            session_id=session_id,
            error=True
        )

@app.post("/api/image", response_model=ImageResponse)
async def analyze_image(file: UploadFile = File(...), session_id: Optional[str] = None):
    """Direct image analysis endpoint"""
    try:
        session_id = get_session_id(session_id)
        logger.info(f"Image analysis for session {session_id}")
        
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Use model service directly
        result = model_service.process_image(image, session_id)
        
        if result.get("error"):
            return ImageResponse(
                caption="Failed to analyze image",
                detected_elements=[],
                session_id=session_id,
                confidence=0.0,
                error=True
            )
        
        # Extract elements
        attributes = result.get("attributes", [])
        detected_elements = [attr.get("label", "") for attr in attributes if attr.get("label")]
        
        # Enhanced context storage for better follow-up questions
        caption = result.get("caption", "")
        sessions[session_id]["last_image_caption"] = caption
        sessions[session_id]["last_image_attributes"] = attributes
        sessions[session_id]["image_context"] = f"Image shows: {caption}. Detected: {', '.join(detected_elements[:5])}"
        
        return ImageResponse(
            caption=result.get("caption", "Image processed"),
            detected_elements=detected_elements,
            session_id=session_id,
            confidence=result.get("confidence", 0.9)
        )
        
    except Exception as e:
        logger.error(f"Image analysis error: {e}")
        session_id = get_session_id(session_id)
        return ImageResponse(
            caption="Failed to analyze image",
            detected_elements=[],
            session_id=session_id,
            confidence=0.0,
            error=True
        )

@app.post("/api/pdf", response_model=PDFResponse)
async def analyze_pdf(file: UploadFile = File(...), session_id: Optional[str] = None):
    """Direct PDF analysis endpoint"""
    try:
        received_session_id = session_id
        session_id = get_session_id(session_id)
        logger.info(f"PDF analysis - Received session_id: {received_session_id}, Using session_id: {session_id}")
        
        if not file.content_type or file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        # Extract PDF text
        pdf_data = await file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_data))
        text_content = ""
        
        max_pages = min(10, len(pdf_reader.pages))
        for page_num in range(max_pages):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            if page_text.strip():
                text_content += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
        
        if not text_content.strip():
            return PDFResponse(
                summary="No text could be extracted from the PDF",
                session_id=session_id,
                pages_processed=0,
                error=True
            )
        
        # Analyze PDF
        summary = rag_service.analyse_pdf_comprehensive(text_content)
        
        # Enhanced PDF context storage for better follow-up questions
        sessions[session_id]["pdf_content"] = text_content
        sessions[session_id]["pdf_summary"] = summary
        sessions[session_id]["pdf_context"] = f"PDF Summary: {summary[:200]}..."
        sessions[session_id]["pdf_filename"] = file.filename
        sessions[session_id]["pdf_uploaded_at"] = datetime.now().isoformat()

        # Clear any previous image context to avoid conflicts
        sessions[session_id].pop("last_image_caption", None)
        sessions[session_id].pop("last_image_attributes", None)
        sessions[session_id].pop("image_context", None)
        
        return PDFResponse(
            summary=summary,
            session_id=session_id,
            pages_processed=max_pages
        )
        
    except Exception as e:
        logger.error(f"PDF analysis error: {e}")
        session_id = get_session_id(session_id)
        return PDFResponse(
            summary=f"Failed to process PDF: {str(e)}",
            session_id=session_id,
            pages_processed=0,
            error=True
        )

@app.post("/api/vision", response_model=ImageResponse)
async def analyze_vision(file: UploadFile = File(...), session_id: Optional[str] = None):
    """Vision analysis endpoint (alias for /api/image for frontend compatibility)"""
    return await analyze_image(file, session_id)

@app.get("/rag/chat")
async def rag_chat(q: str):
    """RAG chat endpoint"""
    try:
        response = rag_service.chat_with_news(q)
        return {"response": response, "success": True}
    except Exception as e:
        logger.error(f"RAG chat error: {e}")
        return {"response": "Error processing news question.", "success": False}

@app.get("/api/status")
async def get_status():
    return {
        "status": "online",
        "active_sessions": len(sessions),
        "services": {
            "model_service": "available",
            "rag_service": "available"
        }
    }

@app.get("/api/debug/sessions")
async def debug_sessions():
    """Debug endpoint to view all sessions and their content"""
    session_info = {}
    for session_id, session_data in sessions.items():
        session_info[session_id] = {
            "has_pdf_content": bool(session_data.get("pdf_content")),
            "pdf_filename": session_data.get("pdf_filename", "None"),
            "pdf_uploaded_at": session_data.get("pdf_uploaded_at", "None"),
            "has_image_context": bool(session_data.get("last_image_caption")),
            "content_length": len(session_data.get("pdf_content", "")) if session_data.get("pdf_content") else 0
        }
    return {
        "total_sessions": len(sessions),
        "sessions": session_info
    }

@app.get("/api/session/{session_id}")
async def get_session_info(session_id: str):
    """Get session information for debugging"""
    if session_id in sessions:
        session_data = sessions[session_id]
        return {
            "session_id": session_id,
            "has_pdf_content": bool(session_data.get("pdf_content")),
            "has_image_content": bool(session_data.get("last_image_caption")),
            "pdf_filename": session_data.get("pdf_filename"),
            "pdf_uploaded_at": session_data.get("pdf_uploaded_at"),
            "pdf_summary_length": len(session_data.get("pdf_summary", "")),
            "pdf_content_length": len(session_data.get("pdf_content", ""))
        }
    return {"error": f"Session {session_id} not found", "active_sessions": list(sessions.keys())}

@app.delete("/api/session/{session_id}")
async def clear_session(session_id: str):
    """Clear session data"""
    if session_id in sessions:
        del sessions[session_id]
        return {"message": f"Session {session_id} cleared"}
    return {"message": f"Session {session_id} not found"}

@app.post("/api/session/transfer-pdf")
async def transfer_pdf_content(from_session: str, to_session: str):
    """Transfer PDF content from one session to another (debugging helper)"""
    if from_session not in sessions:
        return {"error": f"Source session {from_session} not found"}

    if to_session not in sessions:
        sessions[to_session] = {
            "pdf_content": None,
            "last_image_caption": None,
            "last_image_attributes": []
        }

    source_session = sessions[from_session]
    target_session = sessions[to_session]

    if source_session.get("pdf_content"):
        # Transfer all PDF-related data
        target_session["pdf_content"] = source_session["pdf_content"]
        target_session["pdf_summary"] = source_session.get("pdf_summary", "")
        target_session["pdf_context"] = source_session.get("pdf_context", "")
        target_session["pdf_filename"] = source_session.get("pdf_filename", "")
        target_session["pdf_uploaded_at"] = source_session.get("pdf_uploaded_at", "")

        return {
            "message": f"PDF content transferred from {from_session} to {to_session}",
            "pdf_filename": target_session["pdf_filename"],
            "content_length": len(target_session["pdf_content"])
        }
    else:
        return {"error": f"No PDF content found in source session {from_session}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)
