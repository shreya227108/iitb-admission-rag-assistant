import streamlit as st
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Load API key
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ_API_KEY not found.")
    st.stop()

# Optimize chunking for better performance
splitter = SentenceSplitter(chunk_size=400, chunk_overlap=50)
Settings.text_splitter = splitter

#Cache for time optimization
@st.cache_resource
def load_rag_system():

    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    llm = Groq(
        model="llama-3.1-8b-instant",
        api_key=groq_api_key,
        temperature=0.0,
        max_tokens=1000  # context limit declaration
    )

    Settings.embed_model = embed_model
    Settings.llm = llm

    documents = SimpleDirectoryReader(input_dir="data").load_data()

    index = VectorStoreIndex.from_documents(documents)

    retriever = index.as_retriever(similarity_top_k=8)

    return retriever, llm

# Load once
retriever, llm = load_rag_system()

#college name
def is_identity_query(query):
    query = query.lower()

    identity_phrases = [
        "which colleg is this",
        "what is the college name",
        "which college this is",
        "which university is this",
        "what is the name of university",
        "which institute is this",
        "what is the name of institute",
        "which college is this",
        "what is name of college?",
        "where am i"
    ]

    return any(phrase in query for phrase in identity_phrases)

#Greetings detection
def is_greeting(query):
    greetings = [
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "how are you",
        "hii",
        "yo",
    ]

    query = query.lower().strip()

    # Only treat as greeting if short message (<= 3 words)
    if len(query.split()) <= 3:
        return query in greetings

    return False

#small gratitudes
def is_small_talk(query):
    small_talk_phrases = [
        "ok",
        "okay",
        "ok that's great",
        "great",
        "nice",
        "cool",
        "thanks",
        "thank you",
        "alright",
        "got it"
    ]

    query = query.lower().strip()

    return any(phrase in query for phrase in small_talk_phrases)

#Exit greetings
def is_exit(query):
    exit_words = [
        "bye",
        "goodbye",
        "exit",
        "see you",
        "thanks bye",
        "quit",
        "ok thanks",
        "Thanks",
        "That's Great"
    ]

    query = query.lower().strip()

    return any(word in query for word in exit_words)

#Context memory
def get_conversation_context():

    if "messages" not in st.session_state:
        return ""

    # Get last 5 exchanges
    last_messages = st.session_state.messages[-5:]

    conversation_text = ""

    for msg in last_messages:
        role = msg["role"]
        content = msg["content"]
        conversation_text += f"{role.upper()}: {content}\n"

    return conversation_text


# Query function
def admission_assistant(user_query):


    # -------------------------------
    # IDENTITY HANDLER
    # -------------------------------
    if is_identity_query(user_query):
        return """
    This assistant provides admission-related information for:

    🎓 Indian Institute of Technology (IIT)

    You can ask about:
    - Eligibility
    - Fee structure
    - Admission procedure
    - Important dates
    """

    # -------------------------------
    # GREETING HANDLER
    # -------------------------------
    if is_greeting(user_query):
        return """
    👋 Hello! Welcome to the IIT Admission Q&A Assistant.

    I can assist you with:

    - Eligibility criteria  
    - Fee structure  
    - Required documents  
    - Important dates  
    - Admission procedure  

    Please ask your admission-related question.
    """

    # -------------------------------
    # SMALL TALK HANDLER
    # -------------------------------
    if is_small_talk(user_query):
        return """
    😊 You're welcome! 

    If you have any questions about IIT admissions, feel free to ask.
    """

    # -------------------------------
    # EXIT HANDLER
    # -------------------------------
    if is_exit(user_query):
        return """
    👋 Thank you for using the IIT Admission Q&A Assistant.

    If you need assistance again, feel free to return.

    Have a great day!
    """

    # -------------------------------
    # RAG RETRIEVAL
    # -------------------------------

    retrieved_nodes = retriever.retrieve(user_query)

    if not retrieved_nodes:
        return "❌ The requested information is not available in official IIT admission documents."

    top_3_nodes = sorted(
        retrieved_nodes,
        key=lambda x: x.score if x.score else 0,
        reverse=True
    )[:3]

    # 🔐 Strong similarity safety
    if top_3_nodes[0].score is not None and top_3_nodes[0].score < 0.25:
        return "❌ The requested information is not available in official IIT admission documents."

    refined_context = "\n\n".join(
        [node.node.text for node in top_3_nodes]
    )

    conversation_history = get_conversation_context()

    prompt = f"""
    You are an official IIT Admission Assistant.

    STRICT RULES:
    - Use only the provided retrieved context.
    - Use conversation history only for understanding follow-up references.
    - Do NOT hallucinate.
    - If answer not available in retrieved context, say:
    "The requested information is not available in official IIT admission documents."

    Conversation History:
    {conversation_history}

    Retrieved Context:
    {refined_context}

    Current Question:
    {user_query}

    Answer:
    """


    response = llm.complete(prompt)

    # 🔐 Extra safety: prevent LLM from leaking external info
    if "WIT" in response.text or "MIT" in response.text:
        return "❌ The requested information is not available in official IIT admission documents."

    return response.text


#Cache Query Results
@st.cache_data(show_spinner=False)
def cached_query(user_query):
    return admission_assistant(user_query)

# Streamlit UI
st.set_page_config(page_title="Admission Assistant")
st.title("🎓 IIT Admission Q&A Assistant")

# ------------------------------
# SIDEBAR
# ------------------------------
with st.sidebar:
    st.header("📌 About This App")
    st.write("""
    🎓 RAG-powered Admission Assistant
    
    Built using:
    - LlamaIndex
    - HuggingFace Embeddings
    - Groq LLaMA 3
    - Top-3 Retrieval Refinement
    - Similarity Threshold Protection

    🎓 Please Tell me what can I help you with:
    - Eligibility criteria
    - Fee structure(PhD, UG)
    - scholarship documents
    - Rules of college
    - Important dates
    - Admission procedure   
    """)

    if st.button("🔄 Clear Chat"):
        st.session_state.messages = []
        st.rerun()
        
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask your question..."):

    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    # Save user message
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    # Generate assistant response
    with st.chat_message("assistant"):
        response = cached_query(prompt)
        st.markdown(response)

    # Save assistant message
    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

#--------------questions you can ask the AI-----------------------
#what is the eligibility criteria?
#What is the fee structure ?
#What documents are required for the scholarship?
#what are the important dates?
#Hello or any other greetings
