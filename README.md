# RAG-Based Phishing Guideline Generation System

## Overview

This repository contains the implementation and evaluation of a Retrieval-Augmented Generation (RAG) system for generating actionable phishing response guidelines for cybersecurity practitioners.

**Research Focus**: Evaluating RAG system trustworthiness across three dimensions:
- **Context Relevance**: Quality of retrieved knowledge base chunks
- **Answer Relevance**: Alignment between generated guidelines and practitioner queries
- **Faithfulness**: Source grounding vs. hallucination risk


## Key Findings

| Metric | Average Score | Interpretation |
|--------|---------------|----------------|
| **Context Relevance** | 0.677 | Good retrieval quality for definitional queries; weaker for procedural content |
| **Answer Relevance** | 0.703 | Guidelines generally on-topic; degraded for post-incident scenarios |
| **Faithfulness** | 0.328 |  **Critical concern**: LLM frequently adds details beyond retrieved context |

**Main Insight**: RAG systems show promise for knowledge synthesis tasks but require strict safeguards (source citation, confidence thresholds, human oversight) before high-stakes cybersecurity deployment.


## Knowledge Base

**17 authoritative sources** across four categories:

### Category A: Official Standards
### Category B: Government & Regulatory
### Category C: Industry Playbooks
### Category D: Technical Research 


## Quick Start

### Prerequisites

- Python 3.12+
- Google Colab or Jupyter Notebook
- Google Gemini API key

### Installation

1. **Install dependencies**
```bash
pip install -r requirements.txt
```

2. **Set up API key**
```python
import google.generativeai as genai
genai.configure(api_key='API_KEY')
```

3. **Run the notebook**
```bash
jupyter notebook RAG.ipynb
```


## Technical Implementation Details

### Embedding Model: all-MiniLM-L6-v2
- **Architecture**: Sentence-BERT (384-dim embeddings)

### Vector Index: FAISS (Flat L2)
- **Configuration**: Exact search with L2 distance metric

### LLM: Google Gemini 2.5 Flash
- **Temperature**: 0.7

### Chunking Strategy
- **Method**: RecursiveCharacterTextSplitter
- **Size**: 400 characters (≈60-80 words)
- **Overlap**: 50 characters

---

## Disclaimer

**This is a research prototype, not a production system.**  
