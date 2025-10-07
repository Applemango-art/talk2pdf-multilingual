import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import re

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_conversational_chain(target_language_code):
    
    language_instruction = {
        "en": "Answer the question in English as detailed as possible from the provided context.",
        "hi": "Answer in Hindi as detailed as possible from the provided context.",
        "bn": "Answer in Bengali as detailed as possible from the provided context.",
        "mr": "Answer in Marathi as detailed as possible from the provided context.",
        "ta": "Answer in Tamil as detailed as possible from the provided context.",
        "te": "Answer in Telugu as detailed as possible from the provided context.",
        "es": "Answer in Spanish as detailed as possible from the provided context.",
        "fr": "Answer in French as detailed as possible from the provided context."
    }.get(target_language_code, "Answer the question in English as detailed as possible from the provided context.")

    prompt_template = f"""
{language_instruction}
If the answer is not in provided context say, "answer is not available in the context", don't guess.

Context:\n{{context}}?\n
Question:\n{{question}}\n

Answer:
"""

    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def answer_question(user_question, target_language_code):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(user_question)

    chain = get_conversational_chain(target_language_code)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

def main():
    st.set_page_config(page_title="Chat PDF Multilingual with Gemini")

    st.sidebar.title("Menu:")
    pdf_docs = st.sidebar.file_uploader(
        "Upload your PDF Files and Click on Submit & Process", accept_multiple_files=True
    )

    lang_options = {
        "English": "en",
        "Hindi": "hi",
        "Bengali": "bn",
        "Marathi": "mr",
        "Tamil": "ta",
        "Telugu": "te",
        "Spanish": "es",
        "French": "fr"
    }
    chosen_language = st.sidebar.selectbox("Choose your preferred chat language", options=list(lang_options.keys()), index=0)
    selected_lang_code = lang_options[chosen_language]

    process_button = st.sidebar.button("✨ Submit & Process")

    st.header("Chat with PDF in Your Language 💁")

    if process_button:
        if pdf_docs:
            with st.spinner("Processing PDFs and creating vector store..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                _ = get_vector_store(text_chunks)
                st.session_state.vector_store_ready = True
                st.session_state.messages = []
                st.session_state.chat_language = selected_lang_code
            st.success("Processing completed. You can ask questions now.")
        else:
            st.sidebar.error("Please upload at least one PDF file before processing.")

    if "vector_store_ready" not in st.session_state or not st.session_state.vector_store_ready:
        st.info("Upload PDFs and click 'Submit & Process' to start chatting.")
        return

    st.divider()
    st.subheader(f"Chat with PDF in {chosen_language}")

    for msg in st.session_state.get("messages", []):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input(f"Ask a question (answers will be in {chosen_language})..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Generating answer..."):
                answer = answer_question(prompt, st.session_state.chat_language)
            st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()


