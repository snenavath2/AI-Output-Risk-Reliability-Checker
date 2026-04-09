# AI Output Risk & Reliability Checker
## 📄 Publication
This project is published on Zenodo and available as a citable research output.
🔗 https://doi.org/10.5281/zenodo.19478025


## Overview
This project is an AI-based system that compares outputs from two language models (GPT and Grok) to identify epistemic divergence in biomedical documents.

## Features
- PDF upload and text extraction
- Dual model analysis (OpenAI GPT + xAI Grok)
- Structured JSON output
- Epistemic divergence detection
- Streamlit dashboard
- Local storage of results

## Technologies Used
- Python
- Streamlit
- PyMuPDF (fitz)
- OpenAI API
- xAI API

## How It Works
1. Upload a biomedical PDF
2. Text is extracted and cleaned
3. Same input is sent to GPT and Grok
4. Outputs are structured into JSON
5. Differences (divergence) are calculated
6. Results are displayed in dashboard
7. Results are saved locally

## How to Run
```bash
pip install streamlit pymupdf openai
streamlit run app.py
