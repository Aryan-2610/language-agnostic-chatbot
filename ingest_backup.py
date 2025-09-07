import os
import glob
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import chromadb
from utils import extract_text

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize Chroma
CHROMA_PATH = "db"
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection("campus_docs")

def ingest():
    pdf_files = glob.glob("data/*.pdf")

    for pdf in pdf_files:
        doc_id = os.path.basename(pdf)

        # Skip if already ingested
        existing = collection.get(ids=[doc_id], include=[])
        if existing.get("ids"):
            continue

        text = extract_text(pdf)
        if not text:
            print(f"[!] Could not extract text from {pdf}")
            continue

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(text)

        # Embed chunks
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=API_KEY)
        vectors = embeddings.embed_documents(chunks)

        # Store in Chroma
        for i, chunk in enumerate(chunks):
            collection.add(
                ids=[f"{doc_id}_{i}"],
                documents=[chunk],
                embeddings=[vectors[i]],
                metadatas=[{"source": doc_id}]
            )
        print(f"[+] Ingested {pdf}")

if __name__ == "__main__":
    ingest()
