from flask import Flask, render_template, request
from dotenv import load_dotenv
import os

from src.helper import download_hugging_face_embeddings
from src.prompt import *

from langchain_pinecone import PineconeVectorStore
from langchain_community.llms import Ollama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)


'''
# Load environment variables
load_dotenv()

# Pinecone key only
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


# ===============================
# Embeddings
# ===============================
embeddings = download_hugging_face_embeddings()

# Pinecone index
index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# ===============================
# Ollama LLM (LOCAL)
# ===============================
llm = Ollama(
    model="llama3",   # or mistral / phi3 / gemma
    temperature=0.2
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


# ===============================
# Routes
# ===============================
@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    print("User:", msg)

    response = rag_chain.invoke({"input": msg})
    answer = response["answer"]

    print("Bot:", answer)
    return answer


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)'''
