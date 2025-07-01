from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.docstore.document import Document

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())

def create_vector_store(text):
    chunk_size = 500
    chunk_overlap = 100
    chunks = []
    for i in range(0, len(text), chunk_size - chunk_overlap):
        chunks.append(text[i:i+chunk_size])

    vectorizer = TfidfVectorizer().fit(chunks)
    vectors = vectorizer.transform(chunks)

    documents = [Document(page_content=c) for c in chunks]
    return vectorizer, vectors, documents
