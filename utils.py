from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from langchain.docstore.document import Document
import faiss

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())

def create_vector_store(text):
    # Split into chunks
    chunk_size = 500
    chunk_overlap = 100
    chunks = []
    for i in range(0, len(text), chunk_size - chunk_overlap):
        chunks.append(text[i:i+chunk_size])
    
    # Use sentence-transformers for embedding
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = model.encode(chunks)

    # Store in FAISS
    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)

    documents = [Document(page_content=c) for c in chunks]
    return index, model, documents
