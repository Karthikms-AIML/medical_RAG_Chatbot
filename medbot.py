import os
import streamlit as st
import random
from dotenv import load_dotenv, find_dotenv

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import ChatHuggingFace


from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
HF_TOKEN = os.getenv("HF_TOKEN")
print("HF_TOKEN:", HF_TOKEN)


DB_FAISS_PATH = "vectorstore/db_faiss"

# --- UI Setup ---
st.set_page_config(page_title="🧠 MediBot", layout="centered")
st.title("🩺 Chat with MedicalBot")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'suggestions' not in st.session_state:
    st.session_state.suggestions = []

if st.button("🆕 New Chat"):
    st.session_state.chat_history = []
    st.session_state.suggestions = []

# --- Prompt & Sentiment ---
SAFE_PROMPT = """
Use the context below to answer the user's question.
If the symptoms are vague or match multiple conditions, gently list possible causes.

Avoid alarming language. Add: "This is not a diagnosis. Please consult a doctor."

Context: {context}
Question: {question}
Answer:
"""

def detect_sentiment(text):
    lower = text.lower()
    if any(x in lower for x in ['worried', 'scared', 'anxious', 'afraid', 'pain', 'serious', 'weak', 'die', 'fatal']):
        return 'anxious'
    return 'neutral'

def get_suggestions(query):
    q = query.lower()
    if "fever" in q:
        return ["Should I take paracetamol?", "Can dengue cause fever?", "What if fever lasts more than 3 days?"]
    if "headache" in q:
        return ["What are types of headaches?", "Is headache a sign of migraine?", "How to cure headache naturally?"]
    if "diabetes" in q:
        return ["Foods to avoid in diabetes?", "Is diabetes curable?", "How to manage sugar levels?"]
    return ["What are common medical symptoms?", "Should I see a doctor?", "Can AI help with diagnosis?"]

def load_llm():
    endpoint = HuggingFaceEndpoint(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        task="conversational",
        huggingfacehub_api_token=HF_TOKEN,
        temperature=0.7  # ✅ Pass directly
    )
    return ChatHuggingFace(llm=endpoint)

@st.cache_resource
def get_vectorstore():
    embed_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return FAISS.load_local(DB_FAISS_PATH, embed_model, allow_dangerous_deserialization=True)

def create_chain():
    retriever = get_vectorstore().as_retriever(search_kwargs={
        'k': 5,
        'search_type': 'mmr'  # diversity instead of only relevance
    })

    prompt = PromptTemplate(template=SAFE_PROMPT, input_variables=["context", "question"])
    llm = load_llm()

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

# --- Chat UI ---
user_input = st.chat_input("Ask your question...")

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append(("user", user_input))

    with st.spinner("Thinking..."):
        sentiment = detect_sentiment(user_input)

        chain = create_chain()
        response = chain.invoke({"query": user_input})
        result = response["result"]

        if sentiment == 'anxious':
            result += "\n\n💡 Don't worry — many conditions share symptoms. Always consult a doctor for diagnosis."

        st.chat_message("assistant").markdown(result)
        st.session_state.chat_history.append(("assistant", result))

        # Suggestions
        st.session_state.suggestions = get_suggestions(user_input)

# --- Display chat history ---
for role, content in st.session_state.chat_history:
    st.chat_message(role).markdown(content)

# --- Follow-Up Suggestions ---
if st.session_state.suggestions:
    st.markdown("💡 You can also ask:")
    for q in st.session_state.suggestions:
        if st.button(q):
            st.session_state.chat_history.append(("user", q))
            chain = create_chain()
            response = chain.invoke({'query': q})
            answer = response["result"]
            st.chat_message("user").markdown(q)
            st.chat_message("assistant").markdown(answer)
            st.session_state.chat_history.append(("assistant", answer))
