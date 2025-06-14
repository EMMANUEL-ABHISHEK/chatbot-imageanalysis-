# 🧪 Test Cases Documentation
## Conversational Image Recognition Chatbot

### 📋 **Project Overview**
This document provides comprehensive test cases for the multi-modal AI chatbot with image recognition, PDF analysis, and RAG capabilities.

**System Components:**
- **Backend**: FastAPI (Port 3000)
- **Frontend**: React (Port 8000)
- **AI Services**: Gemini API (Primary), Groq API (Fallback)
- **Image Processing**: Google Vision API, BLIP Model
- **RAG System**: Pinecone Vector Database, RSS Feeds

---

## 🎯 **Test Categories Overview**

This document focuses on three essential testing types:
1. **Unit Testing** - Individual component functionality
2. **Integration Testing** - Component interaction and data flow
3. **Acceptance Testing** - End-user scenarios and business requirements

---

## 🔧 **1. Unit Testing**

### **1.1 Model Service Unit Tests**

| Test ID | Function/Method | Input | Expected Output | Test Type | Priority |
|---------|-----------------|-------|-----------------|-----------|----------|
| UT-MS-001 | `generate_text_response()` | "Hello world" | Text response + confidence | Function Test | High |
| UT-MS-002 | `_generate_with_gemini()` | Valid text + session | Gemini API response | API Integration | High |
| UT-MS-003 | `_generate_with_groq()` | Valid text + session | Groq API response | API Integration | High |
| UT-MS-004 | `extract_image_features()` | PIL Image object | Feature dictionary | Image Processing | High |
| UT-MS-005 | `_extract_with_google_vision()` | Valid image | Google Vision result | Vision API | High |
| UT-MS-006 | `_extract_with_blip()` | Valid image | BLIP caption + attributes | Local Model | Medium |
| UT-MS-007 | `_enhance_caption_with_ai()` | Basic caption | Enhanced caption | AI Enhancement | Medium |
| UT-MS-008 | `_update_conversation_history()` | Text + session_id | Updated history | Session Management | High |
| UT-MS-009 | `_get_image_context()` | session_id | Image context string | Context Retrieval | Medium |
| UT-MS-010 | `clear_session()` | session_id | Boolean success | Session Cleanup | Medium |

### **1.2 RAG Service Unit Tests**

| Test ID | Function/Method | Input | Expected Output | Test Type | Priority |
|---------|-----------------|-------|-----------------|-----------|----------|
| UT-RAG-001 | `search_news()` | Query string + k=5 | List of articles | Search Function | High |
| UT-RAG-002 | `ask_ai()` | Prompt text | AI response | AI Generation | High |
| UT-RAG-003 | `analyse_pdf_comprehensive()` | PDF text | Comprehensive analysis | PDF Processing | High |
| UT-RAG-004 | `chat_with_news()` | News query | Contextual response | RAG Pipeline | High |
| UT-RAG-005 | `ingest_rss()` | RSS feeds | Ingestion status | Data Ingestion | Medium |
| UT-RAG-006 | `_create_embeddings()` | Text list | Vector embeddings | Embedding Creation | Medium |
| UT-RAG-007 | `_filter_by_date()` | Articles + days | Filtered articles | Date Filtering | Medium |

### **1.3 API Endpoint Unit Tests**

| Test ID | Endpoint | HTTP Method | Input | Expected Response | Status Code | Priority |
|---------|----------|-------------|-------|-------------------|-------------|----------|
| UT-API-001 | `/api/status` | GET | None | System status JSON | 200 | High |
| UT-API-002 | `/api/chat` | POST | Message + session_id | Chat response | 200 | High |
| UT-API-003 | `/api/image` | POST | Image file | Image analysis | 200 | High |
| UT-API-004 | `/api/pdf` | POST | PDF file | PDF summary | 200 | High |
| UT-API-005 | `/api/multimodal` | POST | Image + text | Combined response | 200 | Medium |
| UT-API-006 | `/rag/chat` | GET | Query parameter | RAG response | 200 | High |
| UT-API-007 | `/api/vision` | POST | Image file | Vision analysis | 200 | Medium |
| UT-API-008 | `/api/transfer-session` | POST | Session IDs | Transfer status | 200 | Low |

### **1.4 Frontend Component Unit Tests**

| Test ID | Component | Test Case | Input/Action | Expected Behavior | Priority |
|---------|-----------|-----------|--------------|-------------------|----------|
| UT-FE-001 | ChatMessage | Render user message | Message props | Displays message correctly | High |
| UT-FE-002 | ChatMessage | Render bot message | Bot response props | Shows bot styling | High |
| UT-FE-003 | ChatInput | Text input handling | User typing | Updates input state | High |
| UT-FE-004 | ChatInput | Send message | Click send button | Triggers onSend callback | High |
| UT-FE-005 | ImageUpload | File selection | Select image file | Updates file state | High |
| UT-FE-006 | ImageUpload | File validation | Invalid file type | Shows error message | Medium |
| UT-FE-007 | LoadingIndicator | Show loading | Loading state true | Displays spinner | Medium |
| UT-FE-008 | TabContext | Tab switching | Click tab | Updates active tab | High |

---

## � **2. Integration Testing**

### **2.1 API Integration Tests**

| Test ID | Integration Scenario | Components | Test Steps | Expected Result | Priority |
|---------|---------------------|------------|------------|-----------------|----------|
| IT-API-001 | Frontend ↔ Backend Chat | React + FastAPI | 1. Send message from UI<br>2. Process in backend<br>3. Return response | Message flow complete | High |
| IT-API-002 | Image Upload Pipeline | Frontend + Model Service | 1. Upload image<br>2. Process with Vision API<br>3. Return analysis | Image analyzed successfully | High |
| IT-API-003 | PDF Analysis Flow | Frontend + RAG Service | 1. Upload PDF<br>2. Extract text<br>3. Generate summary | PDF processed and stored | High |
| IT-API-004 | Session Management | Frontend + Backend | 1. Create session<br>2. Store context<br>3. Retrieve context | Session data persists | High |
| IT-API-005 | RAG Query Pipeline | Frontend + RAG + Pinecone | 1. Submit query<br>2. Search vectors<br>3. Generate response | Relevant news response | High |

### **2.2 Service Integration Tests**

| Test ID | Service Integration | Services | Test Scenario | Expected Outcome | Priority |
|---------|-------------------|----------|---------------|------------------|----------|
| IT-SVC-001 | Model Service + APIs | Gemini + Groq | Gemini fails → Groq fallback | Seamless API switching | High |
| IT-SVC-002 | Image Processing Chain | Google Vision + BLIP | Vision API fails → BLIP fallback | Alternative processing | High |
| IT-SVC-003 | RAG + AI Generation | RAG Service + Gemini | Search articles → Generate response | Contextual AI response | High |
| IT-SVC-004 | Session + Context | Model Service + Session Store | Store image context → Use in chat | Context-aware responses | Medium |
| IT-SVC-005 | PDF + RAG Integration | PDF Parser + Vector DB | Parse PDF → Store embeddings | PDF content searchable | Medium |

### **2.3 Data Flow Integration Tests**

| Test ID | Data Flow | Path | Input | Output | Validation | Priority |
|---------|-----------|------|-------|--------|------------|----------|
| IT-DATA-001 | Image → Caption → Context | Upload → Process → Store | Image file | Stored caption + attributes | Context retrievable | High |
| IT-DATA-002 | PDF → Text → Embeddings | Upload → Parse → Vectorize | PDF file | Vector embeddings in DB | Searchable content | High |
| IT-DATA-003 | Query → Search → Response | Input → Vector Search → AI | Text query | Relevant response | Accurate information | High |
| IT-DATA-004 | Session → Transfer → Restore | Store → Transfer → Load | Session data | Transferred context | Data integrity maintained | Medium |

---

## ✅ **3. Acceptance Testing**

### **3.1 User Story Acceptance Tests**

| Test ID | User Story | As a... | I want to... | So that... | Acceptance Criteria | Priority |
|---------|------------|---------|--------------|------------|-------------------|----------|
| AT-US-001 | Image Analysis | User | Upload an image and ask questions | I can understand image content | ✓ Image uploads successfully<br>✓ Accurate caption generated<br>✓ Follow-up questions answered | High |
| AT-US-002 | Document Q&A | Researcher | Upload PDF and ask specific questions | I can quickly find information | ✓ PDF uploads and processes<br>✓ Content-specific answers<br>✓ Multiple questions supported | High |
| AT-US-003 | News Research | Journalist | Ask about recent events | I get current information | ✓ Recent news retrieved<br>✓ Relevant sources cited<br>✓ Date filtering works | High |
| AT-US-004 | Multi-format Analysis | Analyst | Work with images and documents | I can cross-reference content | ✓ Multiple file types supported<br>✓ Context switching works<br>✓ Integrated insights provided | Medium |

### **3.2 Business Requirement Tests**

| Test ID | Business Requirement | Requirement | Test Scenario | Success Criteria | Priority |
|---------|---------------------|-------------|---------------|------------------|----------|
| AT-BR-001 | Multi-modal Processing | Support text, image, PDF inputs | Upload each file type + interact | All formats processed correctly | High |
| AT-BR-002 | Real-time Responses | Provide timely responses | Submit various queries | Response time < 10 seconds | High |
| AT-BR-003 | Context Awareness | Maintain conversation context | Multi-turn conversations | Context preserved across turns | High |
| AT-BR-004 | Accurate Information | Provide reliable information | Fact-checking queries | Information accuracy > 90% | High |
| AT-BR-005 | User-friendly Interface | Intuitive user experience | New user interaction | User completes tasks without help | Medium |

### **3.3 End-to-End Acceptance Tests**

| Test ID | E2E Scenario | User Journey | Steps | Expected Experience | Priority |
|---------|--------------|--------------|-------|-------------------|----------|
| AT-E2E-001 | Complete Image Workflow | Photo analysis + questions | 1. Upload family photo<br>2. Get automatic description<br>3. Ask "How many people?"<br>4. Ask "What are they wearing?" | Smooth, accurate interaction | High |
| AT-E2E-002 | Document Research Flow | PDF analysis + Q&A | 1. Upload research paper<br>2. Get summary<br>3. Ask about methodology<br>4. Ask about conclusions | Comprehensive document understanding | High |
| AT-E2E-003 | News Investigation | Current events research | 1. Switch to RAG mode<br>2. Ask about recent tech news<br>3. Follow up with specific questions<br>4. Get source citations | Up-to-date, sourced information | High |
| AT-E2E-004 | Mixed Content Analysis | Multi-format workflow | 1. Upload infographic image<br>2. Upload related PDF report<br>3. Ask comparative questions<br>4. Get integrated insights | Cross-format understanding | Medium |

### **3.4 Performance Acceptance Tests**

| Test ID | Performance Requirement | Metric | Test Condition | Acceptance Threshold | Priority |
|---------|------------------------|--------|----------------|---------------------|----------|
| AT-PERF-001 | Response Time - Text | Latency | Simple text query | < 3 seconds | High |
| AT-PERF-002 | Response Time - Image | Processing Time | Standard image (1MB) | < 8 seconds | High |
| AT-PERF-003 | Response Time - PDF | Analysis Time | 10-page document | < 15 seconds | High |
| AT-PERF-004 | Concurrent Users | Throughput | 5 simultaneous users | No performance degradation | Medium |
| AT-PERF-005 | File Size Handling | Capacity | Various file sizes | Up to 10MB files supported | Medium |

### **3.5 Usability Acceptance Tests**

| Test ID | Usability Aspect | Test Method | User Task | Success Criteria | Priority |
|---------|------------------|-------------|-----------|------------------|----------|
| AT-UX-001 | Ease of Use | User observation | First-time image upload | Completes without assistance | High |
| AT-UX-002 | Error Recovery | Error simulation | Handle upload failure | Clear error message + retry option | High |
| AT-UX-003 | Navigation | User flow | Switch between modes | Intuitive tab switching | Medium |
| AT-UX-004 | Mobile Experience | Device testing | Use on mobile device | Responsive design works | Medium |
| AT-UX-005 | Accessibility | Screen reader | Navigate with assistive tech | All features accessible | Low |

---

## � **Test Summary & Execution Plan**

### **Test Case Summary**

| Test Category | Total Tests | High Priority | Medium Priority | Low Priority |
|---------------|-------------|---------------|-----------------|--------------|
| **Unit Testing** | **33** | **20** | **10** | **3** |
| - Model Service | 10 | 6 | 3 | 1 |
| - RAG Service | 7 | 4 | 3 | 0 |
| - API Endpoints | 8 | 6 | 1 | 1 |
| - Frontend Components | 8 | 6 | 2 | 0 |
| **Integration Testing** | **12** | **8** | **4** | **0** |
| - API Integration | 5 | 5 | 0 | 0 |
| - Service Integration | 5 | 3 | 2 | 0 |
| - Data Flow | 4 | 3 | 1 | 0 |
| **Acceptance Testing** | **25** | **15** | **8** | **2** |
| - User Stories | 4 | 3 | 1 | 0 |
| - Business Requirements | 5 | 4 | 1 | 0 |
| - End-to-End | 4 | 4 | 0 | 0 |
| - Performance | 5 | 3 | 2 | 0 |
| - Usability | 5 | 2 | 2 | 1 |
| **TOTAL** | **70** | **43** | **22** | **5** |

### **Test Execution Priority**

#### **Phase 1: Critical Path (High Priority - 43 tests)**
1. **Unit Tests** - Core functionality validation
   - Model Service API integration (Gemini/Groq)
   - Image processing pipeline (Google Vision/BLIP)
   - RAG system components
   - Essential API endpoints

2. **Integration Tests** - Component interaction
   - Frontend ↔ Backend communication
   - API fallback mechanisms
   - Data flow validation

3. **Acceptance Tests** - Business requirements
   - Core user stories
   - Performance thresholds
   - End-to-end workflows

#### **Phase 2: Standard Features (Medium Priority - 22 tests)**
- Enhanced functionality testing
- Secondary integration scenarios
- Usability and accessibility features

#### **Phase 3: Edge Cases (Low Priority - 5 tests)**
- Error handling edge cases
- Performance optimization
- Advanced accessibility features

### **Test Environment Requirements**

#### **Development Environment**
- **Backend**: Python 3.9+, FastAPI, uvicorn main:app --port 3000
- **Frontend**: Node.js 16+, React development server, npm start (port 8000)
- **Required Services**: Gemini API (Primary), Groq API (Fallback), Google Vision API, Pinecone Vector Database

#### **Test Data Requirements**
- **Images**: 10 test images (various formats, sizes)
- **PDFs**: 5 test documents (different content types)
- **API Keys**: Valid keys for all services
- **Test Queries**: Prepared question sets for each domain

### **Success Criteria**

#### **Minimum Viable Product (MVP)**
- ✅ **90%** of High Priority tests pass
- ✅ **Core functionality** works end-to-end
- ✅ **API fallbacks** function correctly
- ✅ **Performance targets** met for critical operations

#### **Production Ready**
- ✅ **95%** of all tests pass
- ✅ **Error handling** graceful and informative
- ✅ **Security** requirements satisfied
- ✅ **Accessibility** standards met

### **Test Reporting Template**

#### **Executive Summary**
| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Overall Pass Rate | 95% | __%  | ⚠️/✅ |
| High Priority Pass Rate | 100% | __%  | ⚠️/✅ |
| Critical Bugs | 0 | __ | ⚠️/✅ |
| Performance Issues | 0 | __ | ⚠️/✅ |

#### **Detailed Results**
| Test Category | Passed | Failed | Skipped | Pass Rate | Issues |
|---------------|--------|--------|---------|-----------|--------|
| Unit Testing | __/33 | __ | __ | __% | __ |
| Integration Testing | __/12 | __ | __ | __% | __ |
| Acceptance Testing | __/25 | __ | __ | __% | __ |
| **TOTAL** | **__/70** | **__** | **__** | **__%** | **__** |

#### **Critical Issues**
- [ ] **Issue 1**: [Description] - Priority: High/Medium/Low
- [ ] **Issue 2**: [Description] - Priority: High/Medium/Low
- [ ] **Issue 3**: [Description] - Priority: High/Medium/Low

#### **Recommendations**
1. **Immediate Actions**: Critical fixes required before release
2. **Short-term**: Improvements for next iteration
3. **Long-term**: Enhancement opportunities

---

## 🚀 **Getting Started with Testing**

### **Quick Test Execution**
1. **Start the application**: uvicorn main:app --port 3000 --reload, cd frontend && npm start
2. **Run basic health checks**: curl http://localhost:3000/api/status
3. **Test core functionality**: Upload test image via UI, Upload test PDF via UI, Try RAG query via UI
4. **Verify all three modes work**: Image Analysis tab, PDF Analysis tab, News & RAG tab

### **Automated Testing Setup**
- **Install testing dependencies**: pip install pytest requests pytest-asyncio
- **Run unit tests**: pytest tests/unit/
- **Run integration tests**: pytest tests/integration/
- **Generate test report**: pytest --html=report.html

---

*Generated for Conversational Image Recognition Chatbot v2.0.0*
*Test Documentation Version: 1.0*
