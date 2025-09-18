import os
from dotenv import load_dotenv, find_dotenv

from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import  HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace

# Load environment variable for HF_TOKEN
load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HF_TOKEN")

# Constants
HUGGINGFACE_REPO_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"
DB_FAISS_PATH = "vectorstore/db_faiss"

# Updated safe prompt template
SAFE_PROMPT_TEMPLATE = """
Use the context below to answer the user's question clearly.

If the symptoms are vague or match multiple conditions, gently list possible causes.
Avoid alarming language. Be factual, calm, and informative.
Add: "This is not a diagnosis. Please consult a doctor."

Context: {context}
Question: {question}
Answer:
"""

# Load the LLM from HuggingFace
def load_llm():
    endpoint = HuggingFaceEndpoint(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        task="conversational",
        huggingfacehub_api_token=HF_TOKEN,
        temperature=0.7  # ✅ Pass directly
    )

    return ChatHuggingFace(llm=endpoint)

# Set custom LangChain prompt
def set_custom_prompt():
    return PromptTemplate(template=SAFE_PROMPT_TEMPLATE, input_variables=["context", "question"])

# Load your stored FAISS vector index (built in Phase 1)
def load_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create the RAG chain (LLM + FAISS)
def get_qa_chain():
    retriever = load_vectorstore().as_retriever(search_kwargs={
        "k": 5,
        "search_type": "mmr"  # more diverse answers
    })

    chain = RetrievalQA.from_chain_type(
        llm=load_llm(),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": set_custom_prompt()
        }
    )

    return chain

# Main interactive loop
if __name__ == "__main__":
    qa_chain = get_qa_chain()

    while True:
        user_query = input("\nAsk your medical question (or type 'exit' to quit):\n> ")

        if user_query.strip().lower() in ["exit", "quit"]:
            print("👋 Exiting the assistant. Stay healthy!")
            break

        try:
            response = qa_chain.invoke({'query': user_query})

            print("\n💬 RESULT:\n", response["result"])

            # Optional: Show source documents
            # for i, doc in enumerate(response.get("source_documents", []), start=1):
            #     print(f"\n📄 Source Document {i}:\n{doc.page_content[:500]}...")  # show first 500 chars

        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
