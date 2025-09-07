import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import chromadb

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=API_KEY)

# Initialize DB
CHROMA_PATH = "db"
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection("campus_docs")

# Small talk responses
SMALL_TALK = {
    "hi": "Hello! How can I help you today?",
    "hello": "Hi there 👋 How can I assist?",
    "bye": "Goodbye! Have a great day 🎓",
}

# Search query in DB
def retrieve_context(query, k=3):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=API_KEY)
    query_vector = embeddings.embed_query(query)
    results = collection.query(query_embeddings=[query_vector], n_results=k)
    return results

def chat(query):
    # Handle small talk
    if query.lower().strip() in SMALL_TALK:
        return SMALL_TALK[query.lower().strip()]

    # Retrieve context
    results = retrieve_context(query)
    docs = results.get("documents", [[]])[0]

    if not docs:  # No relevant docs
        return llm.invoke(query).content

    context = "\n".join(docs)
    prompt = f"""
    You are a helpful campus assistant.
    Use the context below to answer the question.
    If the context is not relevant, answer normally.

    Context:
    {context}

    Question: {query}
    """

    response = llm.invoke(prompt)
    return response.content

if __name__ == "__main__":
    while True:
        query = input("User: ")
        if query.lower() in ["quit", "exit"]:
            break
        print("Bot:", chat(query))
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import chromadb

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=API_KEY)

# Initialize DB
CHROMA_PATH = "db"
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection("campus_docs")

# Small talk responses
SMALL_TALK = {
    "hi": "Hello! How can I help you today?",
    "hello": "Hi there 👋 How can I assist?",
    "bye": "Goodbye! Have a great day 🎓",
}

# Store conversation history
conversation_history = []

# Search query in DB
def retrieve_context(query, k=3):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=API_KEY)
    query_vector = embeddings.embed_query(query)
    results = collection.query(query_embeddings=[query_vector], n_results=k)
    return results

def chat(query):
    global conversation_history

    # Handle small talk
    if query.lower().strip() in SMALL_TALK:
        response = SMALL_TALK[query.lower().strip()]
        conversation_history.append(("user", query))
        conversation_history.append(("assistant", response))
        return response

    # Retrieve context
    results = retrieve_context(query)
    docs = results.get("documents", [[]])[0]

    # Build conversation history text
    history_text = ""
    for role, msg in conversation_history[-6:]:  # keep last 3 exchanges
        history_text += f"{role.capitalize()}: {msg}\n"

    if docs:
        context = "\n".join(docs)
        prompt = f"""
        You are a helpful campus assistant.
        Use the context below to answer the question.
        If the context is not relevant, answer normally.

        Context:
        {context}

        Conversation so far:
        {history_text}

        User: {query}
        Assistant:
        """
    else:
        prompt = f"""
        You are a helpful campus assistant.

        Conversation so far:
        {history_text}

        User: {query}
        Assistant:
        """

    response = llm.invoke(prompt).content

    # Update history
    conversation_history.append(("user", query))
    conversation_history.append(("assistant", response))

    return response

if __name__ == "__main__":
    while True:
        query = input("User: ")
        if query.lower() in ["quit", "exit"]:
            break
        print("Bot:", chat(query))
