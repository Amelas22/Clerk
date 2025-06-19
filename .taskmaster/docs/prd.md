# Product Requirements Document: Legal AI Knowledge Management & Motion Drafting System
**Project Code Name: Clerk**

## 1. Introduction/Overview

This feature creates an AI-powered system for our law firm to automatically organize case documents and draft legal motions. Currently, attorneys spend hours searching through case files and drafting responses to motions from scratch. This system will solve these problems by:

- Automatically reading and organizing all PDF documents from our Box folders
- Creating searchable databases for each legal case
- Using AI to draft responses to opposing counsel's motions
- Tracking deadlines and assigning daily tasks to team members

**Goal:** Build a system that saves attorney time on document research and motion drafting while ensuring all case information remains separate and secure.

## 2. Goals

1. **Reduce motion drafting time** from 8+ hours to under 2 hours per motion
2. **Enable instant access** to any case information through natural language questions
3. **Achieve 80% daily usage** by all attorneys within 6 months of launch
4. **Maintain 100% data isolation** between different legal cases
5. **Process 100% of existing PDF documents** from Box into searchable format (approximately 4,000-6,000 documents across all cases)

## 3. User Stories

### Attorney Stories
1. **As an attorney**, I want to upload a motion filed by opposing counsel and receive a complete outline for our response, so that I can focus on legal strategy rather than research.

2. **As an attorney**, I want to ask questions like "What did the expert witness say about the accident timeline?" or "What were the patient's vitals on the day of the incident?" and get immediate answers with source documents, so that I can quickly prepare for depositions.

3. **As an attorney**, I want to provide basic case facts and have the system draft an initial complaint using our firm's style and successful examples, so that I can file cases faster.

4. **As an attorney**, I want clear explanations when the AI cannot answer my question (like "I don't have enough information about that topic in this case"), so that I know what additional research is needed.

### Support Staff Stories
5. **As a paralegal**, I want to receive a prioritized task list each morning showing what needs to be done for my assigned cases, so that I never miss important deadlines.

6. **As a legal assistant**, I want to upload court deadlines once and have the system automatically remind the right people, so that we never miss filing dates.

### Management Stories
7. **As a managing partner**, I want to see which cases have upcoming deadlines and which attorneys might need help, so that I can allocate resources effectively.

## 4. Functional Requirements

1. **The system must connect to Box API** and access all matter folders
2. **The system must extract text from PDF documents** (OCR already completed by firm) including:
   - Motion practice documents
   - Medical records
   - Expert reports
   - Discovery documents
   - Correspondence
3. **The system must detect duplicate documents** using file hashes and avoid processing the same content twice
4. **The system must handle large PDF files** (medical records can be 200+ pages) by:
   - Splitting into manageable sections
   - Summarizing each section
   - Creating contextual chunks with summaries
5. **The system must store document chunks in a vector database** with metadata including:
   - Matter ID
   - Original document location
   - Document type (pleading, expert report, etc.)
   - Date added
7. **The system must maintain a separate firm-wide knowledge base** containing:
   - Successful motion examples
   - Firm writing style guides
   - Legal argument templates
7. **The system must allow users to upload a defense motion** and generate:
   - Research from the case database
   - Research from external sources (Perplexity, Jina APIs)
   - A complete outline with legal citations
   - Export to spreadsheet format
8. **The system must allow users to edit outlines in a spreadsheet** and mark them ready for drafting
9. **The system must generate complete motion drafts** in .docx format, drafting section by section
10. **The system must save all generated documents back to the correct Box folder**
11. **The system must track deadlines from:**
    - Manual entry by staff
    - Trial orders uploaded to Box
    - Filing timestamps on documents
12. **The system must generate daily task lists** for each team member based on their role and assigned cases
13. **The system must provide a chat interface** where any user can ask questions about any case
14. **The system must ensure case data isolation** - facts from Case A must never appear in responses about Case B
15. **The system must format legal citations in Bluebook style** but must NEVER create fictional citations
16. **The system must handle document versioning** by deleting old vectors when updated documents are uploaded
17. **The system must maintain an error log** that tracks:
    - Failed document processing attempts
    - API connection failures
    - Cases where AI cannot provide answers
18. **The system must chunk documents at ~1400 characters** and add contextual summaries to each chunk before storing

## 5. Non-Goals (Out of Scope)

1. This feature will **NOT** integrate with Westlaw or LexisNexis for citation checking (APIs not available)
2. This feature will **NOT** handle non-PDF documents (Word docs, emails, etc.)
3. This feature will **NOT** automatically file documents with the court
4. This feature will **NOT** manage billing or timekeeping
5. This feature will **NOT** provide legal advice or replace attorney judgment
6. This feature will **NOT** handle documents that aren't already OCR'd

## 6. Design Considerations

### User Interface
- Primary interface: Open WebUI (ChatGPT-like interface)
- Users interact through natural language chat
- Spreadsheet interface for outline editing (Google Sheets or similar)
- Simple web forms for deadline entry

### User Flow Example
1. Attorney opens Open WebUI
2. Types: "Draft a response to the motion to dismiss in Smith v. Jones"
3. Uploads the opposing counsel's motion
4. Reviews generated outline in spreadsheet
5. Makes edits and marks as "ready"
6. Receives draft motion in Box folder

## 7. Technical Considerations

### Infrastructure Setup
- **Python 3.11+** for document processing scripts
- **Qdrant** for vector database (single collection with metadata filtering)
- **Box API** for document access
- **Ollama** for running local AI models
- **Open WebUI** for chat interface
- **n8n** for workflow automation
- **Flowise** for building AI agents
- **Storage:** Minimum 2GB for vector storage, 10GB+ for document caching

### Key Technical Decisions
- Use **one vector database with metadata** instead of separate databases per case
- Store approximately 240,000 vectors initially (120 cases × 40 docs × 50 chunks each)
- All infrastructure runs locally except LLM API calls
- Use existing Box folder structure to organize documents
- Plan for significant storage needs due to medical records and extensive motion practice
- Deploy to VPS for production use with caching layer
- Use file hashes for duplicate detection
- Implement contextual chunking with ~1400 character chunks plus summaries

### Integration Points
- Box API (must maintain authentication)
- Perplexity Deep Research API
- Jina Deep Search API
- Future: Potential Westlaw/LexisNexis integration

## 8. Success Metrics

1. **Primary Metric:** 80% of attorneys use the system daily after 6 months
2. **Time Savings:** Average motion outline generated in under 15 minutes
3. **Accuracy:** Zero instances of case data mixing between matters
4. **Adoption:** 10+ queries per day per attorney after first month
5. **Quality:** 75% of generated outlines require only minor edits

## 9. Resolved Decisions (Previously Open Questions)

1. **Duplicate Detection:** Use file hashes to identify duplicate documents.

2. **Chunk Size:** Document chunks will be ~1400 characters with contextual summaries added before vector storage.

3. **LLM Selection:** To be determined based on testing and performance requirements.

4. **Permissions:** All users have access to all cases (no restrictions needed).

5. **Version Control:** Keep only the most recent version of documents. Delete old vectors when updated versions are uploaded.

6. **Training Data:** Final filed motions will be used to improve the system while maintaining client confidentiality.

7. **Error Handling:** 
   - AI provides clear explanations when it cannot complete tasks (e.g., "I don't have enough information to answer that")
   - Maintain error log for debugging and system improvement

8. **Citation Format:** System will format legal citations in proper Bluebook style but MUST NOT create fictional citations.

9. **Performance:** 
   - Deploy to VPS for production use
   - Implement caching for frequently accessed data
   - Response time is not critical (quality over speed)

10. **Medical Record Processing:** 
    - Split large documents into sections
    - Summarize each section
    - Combine summaries for contextual chunking
    - Different context window than standard legal documents

## 10. Remaining Open Questions

1. **Contextual Chunking Method:** What specific approach should we use to generate the contextual summary that gets added to each 1400-character chunk?

2. **VPS Requirements:** What are the minimum server specifications needed to handle 240,000+ vectors and concurrent users?

3. **Caching Strategy:** Should we cache at the query level, document level, or chunk level? What cache invalidation strategy?

4. **Citation Verification:** How do we ensure the AI never creates fictional citations? Should we maintain a citation database for validation?

## 11. Project File Structure

```
clerk/
├── README.md
├── requirements.txt
├── .env.example
├── config/
│   ├── __init__.py
│   ├── settings.py              # All configuration variables
│   └── prompts.py               # AI prompt templates
│
├── src/
│   ├── __init__.py
│   ├── document_processing/
│   │   ├── __init__.py
│   │   ├── box_client.py        # Box API connection and file access
│   │   ├── pdf_extractor.py     # PDF text extraction
│   │   ├── chunker.py           # Document chunking with context
│   │   ├── deduplicator.py      # Hash-based duplicate detection
│   │   └── medical_processor.py # Special handling for medical records
│   │
│   ├── vector_storage/
│   │   ├── __init__.py
│   │   ├── qdrant_client.py     # Qdrant connection
│   │   ├── embeddings.py        # Generate vector embeddings
│   │   ├── indexer.py           # Store and update vectors
│   │   └── searcher.py          # Query vectors with metadata
│   │
│   ├── ai_agents/
│   │   ├── __init__.py
│   │   ├── motion_drafter.py    # Motion drafting agent
│   │   ├── case_researcher.py   # Case-specific Q&A agent
│   │   ├── task_manager.py      # Deadline and task assignment
│   │   └── citation_formatter.py # Bluebook citation formatting
│   │
│   ├── integrations/
│   │   ├── __init__.py
│   │   ├── perplexity.py        # Perplexity API integration
│   │   ├── jina.py              # Jina search integration
│   │   ├── spreadsheet.py       # Export/import outlines
│   │   └── docx_generator.py    # Generate Word documents
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py            # Error logging system
│       ├── cache.py             # Caching layer
│       └── validators.py        # Input validation
│
├── scripts/
│   ├── initial_import.py        # One-time import of all existing docs
│   ├── daily_sync.py            # Daily Box sync script
│   └── cleanup_vectors.py       # Remove old version vectors
│
├── tests/
│   ├── __init__.py
│   ├── test_document_processing/
│   ├── test_vector_storage/
│   ├── test_ai_agents/
│   └── test_integrations/
│
├── n8n/
│   ├── workflows/
│   │   ├── motion_drafting_workflow.json
│   │   ├── daily_task_workflow.json
│   │   └── deadline_monitor_workflow.json
│   └── README.md
│
├── flowise/
│   ├── flows/
│   │   ├── case_researcher_flow.json
│   │   ├── motion_drafter_flow.json
│   │   └── manager_agent_flow.json
│   └── README.md
│
├── deployment/
│   ├── docker-compose.yml       # For local development
│   ├── production.yml           # VPS deployment config
│   └── setup_vps.sh            # VPS setup script
│
└── docs/
    ├── API.md                   # API documentation
    ├── SETUP.md                 # Setup instructions
    └── USER_GUIDE.md            # End-user documentation
```

## 12. Development Roadmap

### Phase 1: Foundation (Weeks 1-3)
**Goal:** Set up infrastructure and basic document processing

**Week 1: Environment Setup**
- [ ] Set up development environment with all tools
- [ ] Configure Qdrant database with vector schema
- [ ] Create Box API connection and test authentication
- [ ] Set up basic project structure and git repository

**Week 2: Document Processing Pipeline**
- [ ] Build PDF text extraction module
- [ ] Implement 1400-character chunking algorithm
- [ ] Create hash-based duplicate detection
- [ ] Test with sample documents from 2-3 cases

**Week 3: Vector Storage**
- [ ] Implement vector embedding generation
- [ ] Build Qdrant storage with metadata
- [ ] Create basic search functionality
- [ ] Run initial import on 10 test cases

**Deliverable:** Can import PDFs from Box and search them

### Phase 2: Core AI Features (Weeks 4-7)
**Goal:** Build the motion drafting workflow

**Week 4: Basic AI Agents**
- [ ] Set up Ollama with chosen LLM
- [ ] Create case researcher agent in Flowise
- [ ] Build simple Q&A interface in Open WebUI
- [ ] Test accuracy on known case facts

**Week 5: Motion Research Pipeline**
- [ ] Integrate Perplexity API for external research
- [ ] Integrate Jina API for deep search
- [ ] Build outline generation logic
- [ ] Create spreadsheet export functionality

**Week 6: Motion Drafting**
- [ ] Build section-by-section drafting agent
- [ ] Implement citation formatting (Bluebook)
- [ ] Create Word document generator
- [ ] Add citation validation checks

**Week 7: Integration & Testing**
- [ ] Connect all components in n8n workflow
- [ ] Test full pipeline: upload motion → get draft
- [ ] Fix bugs and optimize prompts
- [ ] Get feedback from 2-3 attorneys

**Deliverable:** Working motion drafting system

### Phase 3: Advanced Features (Weeks 8-10)
**Goal:** Add document versioning and special handling

**Week 8: Document Management**
- [ ] Implement version control with vector deletion
- [ ] Build medical record processor
- [ ] Add contextual summary generation
- [ ] Create firm-wide knowledge base

**Week 9: Enhanced Processing**
- [ ] Optimize for large medical records
- [ ] Implement caching layer
- [ ] Add comprehensive error logging
- [ ] Build document update detection

**Week 10: Performance & Scale**
- [ ] Import all 120 cases
- [ ] Performance test with 240,000+ vectors
- [ ] Optimize slow queries
- [ ] Prepare for VPS deployment

**Deliverable:** Full document processing at scale

### Phase 4: Task Management (Weeks 11-12)
**Goal:** Build deadline tracking and task assignment

**Week 11: Deadline System**
- [ ] Create deadline extraction from documents
- [ ] Build manual deadline entry interface
- [ ] Implement deadline calculation logic
- [ ] Set up notification system

**Week 12: Task Assignment**
- [ ] Build management oversight agent
- [ ] Create daily task list generator
- [ ] Implement role-based task routing
- [ ] Create management dashboard

**Deliverable:** Complete task management system

### Phase 5: Deployment & Training (Weeks 13-14)
**Goal:** Deploy to production and train users

**Week 13: Production Deployment**
- [ ] Set up VPS environment
- [ ] Deploy all services with Docker
- [ ] Configure Caddy for HTTPS
- [ ] Run full system tests

**Week 14: User Training**
- [ ] Create user documentation
- [ ] Conduct training sessions
- [ ] Set up support channel
- [ ] Monitor initial usage

**Deliverable:** Live system with trained users

### Phase 6: Optimization (Weeks 15-16)
**Goal:** Refine based on real usage

**Week 15-16: Feedback & Iteration**
- [ ] Gather user feedback
- [ ] Fix reported issues
- [ ] Optimize slow operations
- [ ] Add requested features
- [ ] Document lessons learned

**Deliverable:** Production-ready, optimized system

### Success Milestones
- **Month 1:** Basic document search working
- **Month 2:** First AI-drafted motion completed
- **Month 3:** Full system in daily use
- **Month 4:** 80% user adoption achieved