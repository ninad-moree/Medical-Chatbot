from flask import Flask, render_template, jsonify, request
from langchain_pinecone import PineconeVectorStore
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.helper import download_embedding
from src.prompt import *
import os
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

app = Flask(__name__)

MEDICALBOT_PINECONE_API_KEY = os.getenv("MEDICALBOT_PINECONE_API_KEY")
GOOGLE_GEN_AI = os.getenv("GOOGLE_GEN_AI")

embeddings = download_embedding()

index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

chatModel = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=GOOGLE_GEN_AI,
    temperature=0
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template('chat_v2.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response: ", response["answer"])
    return str(response["answer"])


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)