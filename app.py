# import os
# import json
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain_community.vectorstores import FAISS
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# # Flask app setup
# app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": ["http://localhost:5173", "http://localhost:5000"]}})

# # Configure Google Generative AI
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# if not GOOGLE_API_KEY:
#     raise ValueError("GOOGLE_API_KEY is missing in the .env file")

# # Helper function to extract text from a PDF
# def get_pdf_text(pdf_file):
#     text = ""
#     pdf_reader = PdfReader(pdf_file)
#     for page in pdf_reader.pages:
#         text += page.extract_text()
#     return text

# # Helper function to split text into chunks
# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     return text_splitter.split_text(text)

# # Helper function to create and save a vector store
# def create_vector_store(user_id, text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local(f"faiss_indices/{user_id}_faiss_index")

# # Helper function to load the question-answering chain
# def get_conversational_chain():
#     prompt_template = """
#     Use the following context to answer the question as thoroughly as possible. 
#     If the answer is not found in the context, reply with "The answer is not available in the context."

#     Context:
#     {context}

#     Question:
#     {question}

#     Answer:
#     """
#     model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# # Helper function to handle user input
# def process_user_question(user_id, question):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     try:
#         vector_store = FAISS.load_local(
#             f"faiss_indices/{user_id}_faiss_index", 
#             embeddings, 
#             allow_dangerous_deserialization=True
#         )
#         docs = vector_store.similarity_search(question)
#         chain = get_conversational_chain()
#         response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
#         return response["output_text"]
#     except Exception as e:
#         return f"Error: {str(e)}"


# # Endpoint to upload and process PDFs
# @app.route('/upload_pdfs', methods=['POST'])
# def upload_pdfs():
#     if 'pdfs' not in request.files or 'user_id' not in request.form:
#         return jsonify({"error": "PDFs or user ID missing"}), 400

#     pdf_files = request.files.getlist('pdfs')
#     user_id = request.form['user_id']

#     all_chunks = []
#     for pdf_file in pdf_files:
#         text = get_pdf_text(pdf_file)
#         chunks = get_text_chunks(text)
#         all_chunks.extend(chunks)

#     create_vector_store(user_id, all_chunks)
#     return jsonify({"message": "PDFs processed successfully"}), 200

# # Endpoint to handle user questions
# @app.route('/ask_question', methods=['POST'])
# def ask_question():
#     data = request.json
#     if 'user_id' not in data or 'question' not in data:
#         return jsonify({"error": "User ID or question missing"}), 400

#     user_id = data['user_id']
#     question = data['question']
#     answer = process_user_question(user_id, question)
#     return jsonify({"answer": answer}), 200

# if __name__ == '__main__':
#     # Ensure faiss_indices directory exists
#     os.makedirs('faiss_indices', exist_ok=True)
#     app.run(host="0.0.0.0", port=3000)




import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Flask app setup
app = Flask(__name__)
CORS(app)

# Configure Google Generative AI
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing in the .env file")

# Helper function to extract text from a PDF
def get_pdf_text(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Helper function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

# Helper function to create and save a vector store
def create_vector_store(user_id, text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(f"faiss_indices/{user_id}_faiss_index")

# Helper function to load the question-answering chain
def get_conversational_chain():
    prompt_template = """
    Use the following context to answer the question as thoroughly as possible. 
    If the answer is not found in the context, reply with "The answer is not available in the context."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Helper function to handle user input
def process_user_question(user_id, question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        vector_store = FAISS.load_local(
            f"faiss_indices/{user_id}_faiss_index", 
            embeddings, 
            # allow_dangerous_deserialization=True
        )
        docs = vector_store.similarity_search(question)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
        
        # Clean the output
        output_text = response["output_text"].strip()
        
        # Remove unwanted symbols
        cleaned_output = output_text.replace("**", "").replace("\n*", "").replace("\n", "").replace("*", "").strip()

        return cleaned_output
    except Exception as e:
        return f"Error: {str(e)}"

# Endpoint to upload and process PDFs
@app.route('/upload_pdfs', methods=['POST'])
def upload_pdfs():
    if 'pdfs' not in request.files or 'user_id' not in request.form:
        return jsonify({"error": "PDFs or user ID missing"}), 400

    pdf_files = request.files.getlist('pdfs')
    user_id = request.form['user_id']

    all_chunks = []
    for pdf_file in pdf_files:
        text = get_pdf_text(pdf_file)
        chunks = get_text_chunks(text)
        all_chunks.extend(chunks)

    create_vector_store(user_id, all_chunks)
    return jsonify({"message": "PDFs processed successfully"}), 200

# Endpoint to handle user questions
@app.route('/ask_question', methods=['POST'])
def ask_question():
    data = request.json
    if 'user_id' not in data or 'question' not in data:
        return jsonify({"error": "User ID or question missing"}), 400

    user_id = data['user_id']
    question = data['question']
    answer = process_user_question(user_id, question)

    # Return clean JSON response
    return jsonify({"answer": answer}), 200

if __name__ == '__main__':
    # Ensure faiss_indices directory exists
    os.makedirs('faiss_indices', exist_ok=True)
    app.run(host="0.0.0.0", port=3000)
