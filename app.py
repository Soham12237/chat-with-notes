import streamlit as st
from utils import extract_text_from_pdf, create_vector_store
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Chat with Your Notes (Offline)", layout="wide")
st.title("📚 Chat with Your Notes (No Hugging Face)")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    st.success("PDF uploaded successfully.")
    with st.spinner("Extracting text and building knowledge base..."):
        raw_text = extract_text_from_pdf(uploaded_file)
        vectorizer, vectors, documents = create_vector_store(raw_text)

    st.success("You can now ask questions!")

    question = st.text_input("Ask a question based on the uploaded PDF:")
    if question:
        question_vec = vectorizer.transform([question])
        similarities = cosine_similarity(question_vec, vectors)[0]
        top_indices = similarities.argsort()[-3:][::-1]
        context = "\n\n".join([documents[i].page_content for i in top_indices])

        st.write("### 🤖 Best Matching Sections")
        st.write(context)
