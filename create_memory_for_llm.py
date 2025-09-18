from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# Step 1: Load textbook PDFs
DATA_PATH = "data/"

def load_pdf_files(folder_path):
    loader = DirectoryLoader(
        folder_path,
        glob='*.pdf',
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} pages from PDFs.")
    return documents

# Step 2: Chunk the documents
def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} text chunks.")
    return chunks

# Step 3: Get embedding model
def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("✅ Embedding model loaded.")
    return embedding_model

# Step 4: Store in FAISS vector DB
def store_embeddings(chunks, embed_model, save_path):
    db = FAISS.from_documents(chunks, embed_model)
    db.save_local(save_path)
    print(f"✅ FAISS index and texts saved to '{save_path}'")

# Main driver
if __name__ == "__main__":
    documents = load_pdf_files(DATA_PATH)
    chunks = create_chunks(documents)
    embed_model = get_embedding_model()
    store_embeddings(chunks, embed_model, "vectorstore/db_faiss")
