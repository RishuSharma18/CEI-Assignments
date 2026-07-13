# 📚 Document Question Answering System using RAG


## Overview

This project implements a Retrieval-Augmented Generation (RAG) system that answers questions from custom PDF documents.


The system:

- Loads documents
- Splits text into chunks
- Generates embeddings
- Stores vectors
- Retrieves relevant information
- Generates grounded answers



## Architecture


PDF

↓

Text Extraction

↓

Chunking

↓

Embedding Generation

↓

FAISS Vector Database

↓

Similarity Search

↓

LLM Generation

↓

Answer



## Tech Stack


Python

LangChain

FAISS

Sentence Transformers

HuggingFace Transformers

Streamlit



## Installation


Clone repository

