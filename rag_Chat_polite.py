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
ensure your answers are helpful and clearly respond to the query dont leave ambiguity
act like a admin officer , help the students have a balance of politeness and admin authority remember you are admin

Guidelines:
- Be direct and confident, but also courteous.
- Do NOT use phrases like "based on the context", "based on the information",
  or "according to the documents".
- Do NOT mention sources or reference material.
- Speak as if you already know the policies and procedures.
- Formatting rule:
  * Only use **bold** for dates (e.g., **12th March 2025**) and money amounts
    (e.g., **₹5000**, **$100**). No other formatting.
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
        response = SMALL_TALK[query.lower().strip()]
        conversation_history.append(("user", query))
        conversation_history.append(("assistant", response))
        return response

    # Retrieve context
    results = retrieve_context(query)
    docs = results.get("documents", [[]])[0]

    # Build conversation history text
    history_text = ""
    for role, msg in conversation_history[-6:]:  # last 3 exchanges
        history_text += f"{role.capitalize()}: {msg}\n"

    if docs:
        context = "\n".join(docs)
        prompt = f"""
        You are a campus administration assistant.
        Answer queries in a clear, professional, and authoritative tone,
        as if you are part of the administration.

        Here is some reference material you may use:
        {context}

        Important rules:
        - Do NOT say phrases like "based on the context", "according to the information", 
          "from the documents", or anything similar.
        - Just answer directly, as if you already know the policy.
        - Formatting rule: Only use **bold** for dates (e.g., **12th March 2025**) 
          and money amounts (e.g., **₹5000**, **$100**). No other formatting.

        Conversation so far:
        {history_text}

        User: {query}
        Assistant:
        """
    else:
        prompt = f"""
        You are a campus administration assistant.
        Answer queries in a clear, professional, and authoritative tone,
        as if you are part of the administration.

        Important rules:
        - Do NOT say phrases like "based on the context", "according to the information", 
          or "from the documents".
        - Just answer directly, as if you already know the policy.
        - Formatting rule: Only use **bold** for dates (e.g., **12th March 2025**) 
          and money amounts (e.g., **₹5000**, **$100**). No other formatting.

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
