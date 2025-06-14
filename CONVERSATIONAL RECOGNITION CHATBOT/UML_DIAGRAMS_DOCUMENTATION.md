# 🎨 UML Diagrams Documentation
## Conversational Image Recognition Chatbot

### 📋 **Overview**
This document contains comprehensive UML diagrams for the multi-modal AI chatbot system including Use Case, Class, Activity, Deployment, and Sequence diagrams using PlantUML syntax.

**System Components:**
- **Frontend**: React Application (Port 8000)
- **Backend**: FastAPI Server (Port 3000)
- **AI Services**: Gemini API, Groq API, Google Vision API
- **Storage**: Pinecone Vector Database, Session Storage
- **Processing**: Image Analysis, PDF Processing, RAG System

---

## 🎯 **1. Use Case Diagram**

```plantuml
@startuml UseCase_Chatbot
!theme plain
title Conversational Image Recognition Chatbot - Use Cases

left to right direction

actor "User" as user
actor "System Administrator" as admin
actor "Gemini API" as gemini
actor "Groq API" as groq
actor "Google Vision API" as vision
actor "Pinecone DB" as pinecone

rectangle "Chatbot System" {
  usecase "Upload Image" as UC1
  usecase "Analyze Image" as UC2
  usecase "Ask Questions about Image" as UC3
  usecase "Upload PDF" as UC4
  usecase "Analyze PDF Content" as UC5
  usecase "Ask Questions about PDF" as UC6
  usecase "Search News/RAG" as UC7
  usecase "Generate Text Response" as UC8
  usecase "Manage Sessions" as UC9
  usecase "Switch Chat Modes" as UC10
  usecase "Transfer Sessions" as UC11
  usecase "Monitor System Health" as UC12
  usecase "Configure APIs" as UC13
}

' User interactions
user --> UC1
user --> UC3
user --> UC4
user --> UC6
user --> UC7
user --> UC10

' System processes
UC1 --> UC2
UC4 --> UC5
UC3 --> UC8
UC6 --> UC8
UC7 --> UC8

' Admin functions
admin --> UC11
admin --> UC12
admin --> UC13

' External API dependencies
UC2 --> vision
UC8 --> gemini
UC8 --> groq
UC7 --> pinecone

' Session management
UC3 --> UC9
UC6 --> UC9
UC7 --> UC9

@enduml
```

---

## 🏗️ **2. Class Diagram**

```plantuml
@startuml Class_Chatbot
!theme plain
title Conversational Image Recognition Chatbot - Class Diagram

package "Frontend" {
  class App {
    +state: AppState
    +sessionId: string
    +chatMode: string
    +messages: Message[]
    +handleSendMessage()
    +handleFileUpload()
    +switchChatMode()
  }
  
  class ChatMessage {
    +id: string
    +text: string
    +sender: string
    +timestamp: Date
    +render()
  }
  
  class ImageUpload {
    +file: File
    +preview: string
    +onFileSelect()
    +validateFile()
  }
  
  class TabContext {
    +mode: string
    +hasUploadedFile: boolean
    +fileType: string
    +fileName: string
  }
}

package "Backend API" {
  class FastAPIApp {
    +app: FastAPI
    +sessions: dict
    +cors_middleware()
    +static_files()
  }
  
  class ChatRequest {
    +message: string
    +session_id: string
    +tab_context: TabContext
    +force_image_context: boolean
  }
  
  class ImageResponse {
    +caption: string
    +detected_elements: list
    +session_id: string
    +confidence: float
    +error: boolean
  }
  
  class PDFResponse {
    +summary: string
    +session_id: string
    +pages_processed: int
  }
}

package "Model Service" {
  class ModelService {
    +device: torch.device
    +conversation_history: dict
    +image_context: dict
    +gemini_client: GenerativeModel
    +groq_client: Groq
    +blip_model: BlipModel
    +clip_model: CLIPModel
    
    +generate_text_response()
    +extract_image_features()
    +process_image()
    +process_multimodal()
    +clear_session()
    -_generate_with_gemini()
    -_generate_with_groq()
    -_extract_with_google_vision()
    -_extract_with_blip()
    -_enhance_caption_with_ai()
  }
}

package "RAG Service" {
  class RAGService {
    +pinecone_client: Pinecone
    +embedding_model: SentenceTransformer
    +groq_client: Groq
    +gemini_client: GenerativeModel
    
    +search_news()
    +chat_with_news()
    +analyse_pdf_comprehensive()
    +ingest_rss()
    +ask_ai()
    -_create_embeddings()
    -_filter_by_date()
  }
}

package "External APIs" {
  class GeminiAPI {
    +api_key: string
    +model: string
    +generate_content()
  }
  
  class GroqAPI {
    +api_key: string
    +model: string
    +chat_completions_create()
  }
  
  class GoogleVisionAPI {
    +api_key: string
    +analyze_image()
  }
  
  class PineconeDB {
    +api_key: string
    +index_host: string
    +query()
    +upsert()
  }
}

' Relationships
App --> ChatMessage
App --> ImageUpload
App --> TabContext
App --> FastAPIApp : HTTP requests

FastAPIApp --> ModelService
FastAPIApp --> RAGService
FastAPIApp --> ChatRequest
FastAPIApp --> ImageResponse
FastAPIApp --> PDFResponse

ModelService --> GeminiAPI
ModelService --> GroqAPI
ModelService --> GoogleVisionAPI

RAGService --> GeminiAPI
RAGService --> GroqAPI
RAGService --> PineconeDB

@enduml
```

---

## 🔄 **3. Activity Diagram**

```plantuml
@startuml Activity_Chatbot
!theme plain
title Conversational Image Recognition Chatbot - Activity Flow

start

:User opens application;
:Select chat mode;

if (Mode selected?) then (Image Analysis)
  :Upload image file;
  :Validate file format;
  if (Valid image?) then (yes)
    :Process with Google Vision API;
    if (Vision API success?) then (yes)
      :Generate enhanced caption;
    else (no)
      :Fallback to BLIP model;
      :Generate basic caption;
    endif
    :Store image context in session;
    :Display image analysis;
    :Wait for user questions;
    :Generate context-aware responses;
  else (no)
    :Show error message;
    stop
  endif

elseif (Mode selected?) then (PDF Analysis)
  :Upload PDF file;
  :Validate PDF format;
  if (Valid PDF?) then (yes)
    :Extract text content;
    :Generate comprehensive summary;
    :Store PDF content in session;
    :Display PDF summary;
    :Wait for user questions;
    :Answer based on PDF content;
  else (no)
    :Show error message;
    stop
  endif

elseif (Mode selected?) then (News & RAG)
  :Enter text query;
  :Search vector database;
  if (Results found?) then (yes)
    :Retrieve relevant articles;
    :Generate contextual response;
  else (no)
    :Use general AI response;
  endif
  :Display response with sources;

endif

:Continue conversation;
if (More questions?) then (yes)
  :Process follow-up questions;
  :Maintain session context;
else (no)
  :End conversation;
  stop
endif

@enduml
```

---

## 🧩 **4. Component Diagram**

```plantuml
@startuml Component_Chatbot
!theme plain
title Conversational Image Recognition Chatbot - Component Architecture

package "Frontend Layer" {
  component [React App] as ReactApp {
    component [Chat Interface] as ChatUI
    component [Image Upload] as ImageUpload
    component [PDF Upload] as PDFUpload
    component [Tab Manager] as TabManager
    component [Message Display] as MessageDisplay
    component [Loading Indicators] as LoadingUI
  }

  component [HTTP Client] as HTTPClient
  component [Session Manager] as SessionMgr
  component [File Validator] as FileValidator
}

package "API Gateway Layer" {
  component [FastAPI Server] as FastAPI {
    component [CORS Middleware] as CORS
    component [Static Files] as StaticFiles
    component [Request Router] as Router
    component [Response Handler] as ResponseHandler
  }

  component [Session Storage] as SessionStore
  component [File Handler] as FileHandler
}

package "Business Logic Layer" {
  component [Model Service] as ModelService {
    component [Text Generator] as TextGen
    component [Image Processor] as ImageProc
    component [Context Manager] as ContextMgr
    component [API Fallback] as APIFallback
  }

  component [RAG Service] as RAGService {
    component [Vector Search] as VectorSearch
    component [PDF Analyzer] as PDFAnalyzer
    component [News Aggregator] as NewsAgg
    component [Embedding Generator] as EmbedGen
  }
}

package "AI Processing Layer" {
  component [Gemini Client] as GeminiClient
  component [Groq Client] as GroqClient
  component [Google Vision Client] as VisionClient

  component [Local Models] as LocalModels {
    component [BLIP Model] as BLIPModel
    component [CLIP Model] as CLIPModel
    component [ViT Model] as ViTModel
    component [SentenceTransformer] as STModel
  }
}

package "Data Layer" {
  component [Pinecone Vector DB] as PineconeDB
  component [RSS Feed Reader] as RSSReader
  component [File System] as FileSystem
  component [Memory Cache] as MemCache
}

package "External Services" {
  component [Gemini API] as GeminiAPI
  component [Groq API] as GroqAPI
  component [Google Vision API] as VisionAPI
  component [RSS Sources] as RSSSources
}

' Frontend connections
ReactApp --> HTTPClient : HTTP Requests
ReactApp --> SessionMgr : Session Management
ReactApp --> FileValidator : File Validation

HTTPClient --> FastAPI : REST API Calls

' API Gateway connections
FastAPI --> ModelService : Service Calls
FastAPI --> RAGService : Service Calls
FastAPI --> SessionStore : Session Data
FastAPI --> FileHandler : File Operations

' Business Logic connections
ModelService --> GeminiClient : Text Generation
ModelService --> GroqClient : Fallback Generation
ModelService --> VisionClient : Image Analysis
ModelService --> LocalModels : Local Processing

RAGService --> PineconeDB : Vector Operations
RAGService --> GeminiClient : AI Generation
RAGService --> GroqClient : Fallback AI
RAGService --> LocalModels : Embeddings

' AI Processing connections
GeminiClient --> GeminiAPI : API Calls
GroqClient --> GroqAPI : API Calls
VisionClient --> VisionAPI : API Calls

' Data Layer connections
PineconeDB --> MemCache : Caching
RSSReader --> RSSSources : Data Ingestion
FileHandler --> FileSystem : File Storage

' Internal component relationships
TextGen --> APIFallback : Error Handling
ImageProc --> ContextMgr : Context Storage
VectorSearch --> EmbedGen : Embedding Creation
PDFAnalyzer --> NewsAgg : Content Processing

@enduml
```

---

## 🚀 **5. Deployment Diagram**

```plantuml
@startuml Deployment_Chatbot
!theme plain
title Conversational Image Recognition Chatbot - Deployment Architecture

node "Client Browser" {
  component "React Frontend" as frontend {
    port "Port 8000" as p8000
  }
}

node "Application Server" {
  component "FastAPI Backend" as backend {
    port "Port 3000" as p3000
  }

  component "Model Service" as models {
    artifact "BLIP Model"
    artifact "CLIP Model"
    artifact "ViT Model"
  }

  component "RAG Service" as rag {
    artifact "SentenceTransformer"
    artifact "PDF Parser"
  }
}

cloud "External APIs" {
  component "Gemini API" as gemini
  component "Groq API" as groq
  component "Google Vision API" as vision
}

cloud "Vector Database" {
  database "Pinecone" as pinecone {
    artifact "News Embeddings"
    artifact "RSS Data"
  }
}

cloud "RSS Feeds" {
  component "News Sources" as rss
}

node "Local Storage" {
  database "Session Store" as sessions
  folder "Generated Images" as images
  folder "Uploaded Files" as uploads
}

' Connections
frontend -down-> backend : HTTP/REST API
backend -down-> models : Local Processing
backend -down-> rag : Document Analysis
backend -right-> gemini : AI Generation
backend -right-> groq : Fallback AI
backend -right-> vision : Image Analysis
backend -down-> sessions : Session Management
backend -down-> images : File Storage
backend -down-> uploads : File Storage
rag -right-> pinecone : Vector Search
rag -up-> rss : Data Ingestion

' Protocols
frontend : HTTPS
backend : REST API
gemini : HTTPS/API
groq : HTTPS/API
vision : HTTPS/API
pinecone : HTTPS/gRPC

@enduml
```

---

## 📋 **6. Sequence Diagram - Image Analysis Flow**

```plantuml
@startuml Sequence_ImageAnalysis
!theme plain
title Image Analysis Sequence Diagram

actor User
participant "React Frontend" as Frontend
participant "FastAPI Backend" as Backend
participant "Model Service" as ModelService
participant "Google Vision API" as Vision
participant "BLIP Model" as BLIP
participant "Gemini API" as Gemini

User -> Frontend: Upload image file
Frontend -> Frontend: Validate file format
Frontend -> Backend: POST /api/image (multipart/form-data)

Backend -> Backend: Create/get session
Backend -> ModelService: process_image(image, session_id)

ModelService -> ModelService: Resize image if needed
ModelService -> Vision: Extract features with Google Vision

alt Vision API Success
    Vision -> ModelService: Return detailed analysis
    ModelService -> ModelService: Parse vision response
    ModelService -> ModelService: Extract colors and objects
else Vision API Fails
    ModelService -> BLIP: Generate caption with BLIP model
    BLIP -> ModelService: Return basic caption
    ModelService -> Gemini: Enhance caption with AI
    Gemini -> ModelService: Return enhanced caption
end

ModelService -> ModelService: Store image context in session
ModelService -> Backend: Return image analysis result

Backend -> Frontend: Return ImageResponse (caption, attributes)
Frontend -> Frontend: Display image analysis
Frontend -> User: Show caption and detected elements

User -> Frontend: Ask follow-up question about image
Frontend -> Backend: POST /api/chat (with image context)

Backend -> ModelService: generate_text_response(text, session_id, with_image_context=true)
ModelService -> ModelService: Get image context from session
ModelService -> Gemini: Generate response with image context
Gemini -> ModelService: Return contextual response
ModelService -> Backend: Return response with confidence

Backend -> Frontend: Return chat response
Frontend -> User: Display contextual answer

@enduml
```

---

## 📄 **7. Sequence Diagram - PDF Analysis Flow**

```plantuml
@startuml Sequence_PDFAnalysis
!theme plain
title PDF Analysis Sequence Diagram

actor User
participant "React Frontend" as Frontend
participant "FastAPI Backend" as Backend
participant "RAG Service" as RAGService
participant "Gemini API" as Gemini
participant "Session Store" as Sessions

User -> Frontend: Upload PDF file
Frontend -> Frontend: Validate PDF format
Frontend -> Backend: POST /api/pdf (multipart/form-data)

Backend -> Backend: Create/get session
Backend -> Backend: Extract text from PDF using PyPDF2

loop For each page (max 50 pages)
    Backend -> Backend: Extract text content
end

Backend -> RAGService: analyse_pdf_comprehensive(pdf_text)
RAGService -> Gemini: Generate comprehensive analysis
Gemini -> RAGService: Return detailed summary
RAGService -> Backend: Return analysis result

Backend -> Sessions: Store PDF content and summary
Backend -> Sessions: Clear any previous image context
Backend -> Frontend: Return PDFResponse (summary, session_id)

Frontend -> Frontend: Display PDF summary
Frontend -> User: Show analysis results

User -> Frontend: Ask question about PDF content
Frontend -> Backend: POST /api/chat (with PDF context)

Backend -> Backend: Check session for PDF content
Backend -> RAGService: Generate response based on PDF content
RAGService -> Gemini: Answer question using PDF context
Gemini -> RAGService: Return contextual answer
RAGService -> Backend: Return response

Backend -> Frontend: Return chat response
Frontend -> User: Display PDF-specific answer

@enduml
```

---

## 🔍 **8. Sequence Diagram - RAG News Search Flow**

```plantuml
@startuml Sequence_RAGSearch
!theme plain
title RAG News Search Sequence Diagram

actor User
participant "React Frontend" as Frontend
participant "FastAPI Backend" as Backend
participant "RAG Service" as RAGService
participant "Pinecone DB" as Pinecone
participant "Gemini API" as Gemini

User -> Frontend: Enter news query in RAG mode
Frontend -> Backend: GET /rag/chat?q=query

Backend -> RAGService: chat_with_news(query)
RAGService -> RAGService: search_news(query, k=5, days_filter=1)

RAGService -> RAGService: Create query embeddings
RAGService -> Pinecone: Query vector database

alt Recent articles found
    Pinecone -> RAGService: Return relevant articles (1 day)
else No recent articles
    RAGService -> RAGService: Expand search to 7 days
    RAGService -> Pinecone: Query with broader date range
    Pinecone -> RAGService: Return articles (7 days)
end

alt Articles found
    RAGService -> RAGService: Format context from top 3 articles
    RAGService -> Gemini: Generate response with news context
    Gemini -> RAGService: Return contextual response
else No articles found
    RAGService -> Gemini: Generate general response
    Gemini -> RAGService: Return general answer
end

RAGService -> Backend: Return final response
Backend -> Frontend: Return news-based answer
Frontend -> User: Display response with sources

@enduml
```

---

## 📊 **UML Diagrams Summary**

### **Diagram Overview**

| Diagram Type | Purpose | Key Components | Complexity |
|--------------|---------|----------------|------------|
| **Use Case** | System functionality from user perspective | 13 use cases, 6 actors | Medium |
| **Class** | System structure and relationships | 15+ classes across 5 packages | High |
| **Activity** | Business process flow | 3 main workflows, decision points | Medium |
| **Component** | System architecture and component relationships | 6 layers, 25+ components | High |
| **Deployment** | Physical architecture and infrastructure | 7 nodes, multiple protocols | High |
| **Sequence** | Interaction flows over time | 3 detailed scenarios | High |

### **Key Architectural Patterns**

1. **Microservices Architecture**: Separate Model Service and RAG Service
2. **API Gateway Pattern**: FastAPI as central routing hub
3. **Fallback Pattern**: Multiple AI APIs with graceful degradation
4. **Session Management**: Stateful conversation handling
5. **Multi-modal Processing**: Unified interface for different input types

### **Technology Stack Visualization**

- **Frontend**: React (Port 8000) → User Interface
- **Backend**: FastAPI (Port 3000) → API Gateway
- **AI Services**: Gemini (Primary), Groq (Fallback) → Text Generation
- **Vision**: Google Vision API, BLIP Model → Image Analysis
- **Storage**: Pinecone Vector DB → RAG System
- **Processing**: Local ML Models → Offline Capabilities

### **Data Flow Patterns**

1. **Image Flow**: Upload → Vision API → Context Storage → Q&A
2. **PDF Flow**: Upload → Text Extraction → Analysis → Q&A
3. **RAG Flow**: Query → Vector Search → Context Generation → Response

---

*Generated for Conversational Image Recognition Chatbot v2.0.0*
*UML Documentation Version: 1.0*
