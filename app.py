import streamlit as st
from utils import extract_text_from_pdf, create_vector_store
from transformers import pipeline
import torch

st.set_page_config(page_title="Free Chat with Your Notes", layout="wide")
st.title("📚 Chat with Your Notes (Open Source)")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    st.success("PDF uploaded successfully.")
    with st.spinner("Extracting text and building knowledge base..."):
        raw_text = extract_text_from_pdf(uploaded_file)
        index, embed_model, documents = create_vector_store(raw_text)

    
    qa_pipeline = pipeline(
    "text2text-generation",
    model="google/gemma-3n-E4B-it"
)



    st.success("You can now ask questions!")

    question = st.text_input("Ask a question based on the uploaded PDF:")
    if question:
        
        question_vec = embed_model.encode([question])
        top_k = 5
        D, I = index.search(question_vec, top_k)
        context = "\n".join([documents[i].page_content for i in I[0]])

    
        prompt = f"Answer the question based on the context below:\n\nContext:\n{context}\n\nQuestion: {question}"
        with st.spinner("Thinking..."):
            response = qa_pipeline(prompt, max_new_tokens=256)[0]['generated_text']
            st.write("### 🤖 Answer")
            st.write(response)
