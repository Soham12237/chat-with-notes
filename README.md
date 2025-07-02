# Chat with Your Notes (PDF Chat Assistant)

This is a Streamlit web application that allows users to upload a PDF document and ask questions about its contents. The system extracts the text from the PDF, generates embeddings using a sentence-transformer, retrieves relevant chunks using FAISS, and uses a generative language model from Hugging Face to answer user queries.

---

## Features

- Upload and extract text from PDF files
- Split text into chunks for better semantic search
- Encode chunks using Sentence Transformers
- Store and retrieve vectors with FAISS
- Use a Hugging Face model (FLAN-T5) to answer questions based on retrieved content
- Simple and efficient Streamlit interface

---

## Requirements

Create a directory .streamlit/secrets.toml and put the content in the PDF uploaded inside the file. 
Host locally using streamlit run app.py in bash or command prompt. There is automatic redirection to browser for web app deployment. 


Install all dependencies using:

```bash
pip install -r requirements.txt

