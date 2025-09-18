# appserver.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from connect_memory_with_llm import get_qa_chain

app = FastAPI()
qa_chain = get_qa_chain()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/ask")
async def ask(req: Request):
    data = await req.json()
    query = data.get("query", "")
    response = qa_chain.invoke({"query": query})
    return {"response": response["result"]}
