# RAG-Medical-Chatbot

A **medical chatbot** using **RAG (Retrieval-Augmented Generation)** with **LangChain**, **Pinecone**, and **Ollama (Mistral)**.

---

## How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/sejalmahajan27/RAG-Medical-Chatbot.git
cd RAG-Medical-Chatbot/Medical-Chatbot

### STEP 01- Create and activate a conda environment

conda create -n medibot python=3.10 -y
conda activate medibot



### STEP 02- install the requirements

pip install -r requirements.txt



### STEP 03 - Create a .env file in the root directory

PINECONE_API_KEY="your_pinecone_api_key_here"
OLLAMA_MODEL="mistral"   # Mistral local LLM


# STEP 04 - Store embeddings in Pinecone
python store_index.py


# STEP 05 - Run the chatbot
python app.py

#Open your browser: 
http://localhost:8080



### Techstack Used:

- Python
- LangChain
- Flask
- Ollama(Mistral)
- Pinecone


