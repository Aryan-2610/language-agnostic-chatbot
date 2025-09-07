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

from langchain.prompts import ChatPromptTemplate

# System-level instructions (enforced every turn)
SYSTEM_PROMPT = """
You are a campus administration assistant.
Respond in a clear, professional, and polite tone,
as if you are a staff member helping students.

Your response MUST always be in valid JSON with the following keys:
- intent: one of ["greeting", "ending", "query", "inap_language"]
- fallback: a float between 0.0 and 1.0 (higher if you are unsure or if query needs admin)
- msg: your actual assistant reply (string, natural language)
- doc: the main source PDF filename if used, else null
- page: the page number if available, else null

Guidelines:
- For greetings (hi, hello), intent = "greeting"
- For goodbyes (bye), intent = "ending"
- For abusive or inappropriate queries, intent = "inap_language"
- Otherwise, intent = "query"
- fallback ≥ 0.7 if you cannot answer confidently (so ticket can be raised)
- msg should sound like you already know the policy, never mention "context" or "documents".
- doc/page should be filled ONLY if you used context from a PDF.
"""


# Create reusable prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{history}\nUser: {query}\nAssistant:")
])



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
    context = "\n".join(docs) if docs else ""

    # Build conversation history text
    history_text = ""
    for role, msg in conversation_history[-6:]:
        history_text += f"{role.capitalize()}: {msg}\n"

    # Fill in template
    formatted_prompt = prompt_template.format(
        history=history_text + ("\n" + context if context else ""),
        query=query
    )

    response = llm.invoke(formatted_prompt).content

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
        return {
            "intent": "greeting" if "hi" in query.lower() or "hello" in query.lower() else "ending",
            "fallback": 0.0,
            "msg": SMALL_TALK[query.lower().strip()],
            "doc": None,
            "page": None
        }

    # Retrieve context
    results = retrieve_context(query)
    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []

    # Extract source + page safely
    doc_name = None
    page_num = None
    if isinstance(metadatas, dict):   # case: single dict
        doc_name = metadatas.get("source")
        page_num = metadatas.get("page")
    elif isinstance(metadatas, list) and metadatas:  # case: list of dicts
        doc_name = metadatas[0].get("source")
        page_num = metadatas[0].get("page")

    context = "\n".join(docs) if docs else ""

    # Build conversation history
    history_text = ""
    for role, msg in conversation_history[-6:]:
        history_text += f"{role.capitalize()}: {msg}\n"

    # JSON-enforcing prompt
    prompt = prompt_template.format(
        history=history_text + ("\n" + context if context else ""),
        query=query
    )

    response = llm.invoke(prompt).content

    # Ensure valid JSON (sometimes LLM may add extra text)
    import json
    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        data = {
            "intent": "query",
            "fallback": 1.0,
            "msg": "Sorry, I could not process this request correctly.",
            "doc": None,
            "page": None
        }

    # Attach doc + page from retrieval if LLM didn’t include them
    if not data.get("doc") and doc_name:
        data["doc"] = doc_name
    if not data.get("page") and page_num:
        data["page"] = page_num

    # Update history
    conversation_history.append(("user", query))
    conversation_history.append(("assistant", data["msg"]))

    return data


if __name__ == "__main__":
    while True:
        query = input("User: ")
        if query.lower() in ["quit", "exit"]:
            break
        print("Bot:", chat(query))
