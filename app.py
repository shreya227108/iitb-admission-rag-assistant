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
        "Bye",
        "By,"
        "See you"
    ]

    query = query.lower().strip()

    # Only treat as greeting if short message (<= 3 words)
    if len(query.split()) <= 3:
        return query in greetings

    return False

# Query function
def admission_assistant(user_query):

    # -------------------------------
    # GREETING HANDLER (No RAG)
    # -------------------------------
    if is_greeting(user_query):
        return """
    👋 Hello! Welcome to the IIT Admission Q&A Assistant.

    I can help you with:

    • Eligibility criteria  
    • Fee structure
    • scholarship documents
    • Rules of college
    • Important dates
    • Admission procedure  

    Please ask your admission-related question.
    """

    # -------------------------------
    # RAG FLOW FOR QUESTIONS
    # -------------------------------

    retrieved_nodes = retriever.retrieve(user_query)

    if not retrieved_nodes:
        return "❌ No relevant information found in official admission documents."

    top_3_nodes = sorted(
        retrieved_nodes,
        key=lambda x: x.score if x.score else 0,
        reverse=True
    )[:3]

    # Similarity safety check
    if top_3_nodes[0].score is None or top_3_nodes[0].score < 0.3:
        return "❌ Information not available in official admission documents."

    refined_context = "\n\n".join(
        [node.node.text for node in top_3_nodes]
    )

    prompt = f"""
        You are an official IIT Admission Assistant.

        Use ONLY the provided context.
        Answer in professional bullet points.

        Context:
        {refined_context}

        Question:
        {user_query}

        Answer:
        """
    
    #testing only
    print("Top score:", top_3_nodes[0].score)

    response = llm.complete(prompt)

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

    • Eligibility criteria  
    • Fee structure(PhD, UG)
    • scholarship documents
    • Rules of college
    • Important dates  
    • Admission procedure  
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
