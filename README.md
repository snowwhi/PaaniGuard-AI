# 🌊 PaaniGuard - Pakistan Water Intelligence System

## 📋 Project Overview

**PaaniGuard** is an AI-powered RAG (Retrieval-Augmented Generation) system that makes Pakistan's water crisis research accessible through natural conversation. Built in 48 hours during the HEC GenAI Hackathon, it transforms static PDF research into an interactive knowledge base.

---

## 🎯 Problem Statement

Pakistan faces a severe water crisis:
- **40%** of children suffer from stunting due to contaminated water
- **33 million** people displaced by 2022 floods
- **$30+ billion** in economic losses from floods
- Expert research trapped in PDFs, inaccessible to citizens
- No centralized system for water intelligence

---

## 💡 Solution: PaaniGuard

An intelligent chatbot that:
- 🔍 **Searches** through 4 expert PDFs on Pakistan water crisis
- 🤖 **Uses Gemini AI** to generate accurate, contextual answers
- 💬 **Responds** in natural language (Urdu/English)
- 📊 **Visualizes** water data through interactive charts

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| **Frontend** | Gradio 4.44.0 |
| **Vector Database** | FAISS + sentence-transformers |
| **Embeddings** | all-MiniLM-L6-v2 |
| **LLM** | Google Gemini 1.5 Flash |
| **PDF Processing** | PyPDF |
| **Deployment** | Hugging Face Spaces |
| **Language** | Python 3.10+ |

---

## 📂 Data Sources
   WaterwayDataset of All the rivers and dams of pakistan 
   https://data.humdata.org/m/dataset/hotosm_pak_waterways
   
4 Expert Research PDFs on:
- 📘 Indus River System & Hydrology
- 📗 Pakistan Flood Assessment 2022
- 📕 Groundwater Depletion in Punjab/Balochistan
- 📙 Climate Change Impact on Water Resourcers

