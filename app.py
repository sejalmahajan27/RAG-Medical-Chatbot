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
    model="mistral",   # or mistral / phi3 / gemma
    temperature=0
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
    msg = request.form["msg"].strip()
    print("User:", msg)

    # Handle casual greetings
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    thanks = ["thank you", "thanks"]

    msg_lower = msg.lower()

    if msg_lower in greetings:
        answer = "Hello! I’m your medical assistant bot. How can I help you today?"
    elif msg_lower in thanks:
        answer = "You’re welcome! 😊"
    else:
        # Send to RAG chain only if it's a meaningful query
        response = rag_chain.invoke({"input": msg})
        answer = response.get("answer", "Sorry, I couldn't find an answer for that.")

    print("Bot:", answer)
    return answer





if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)


