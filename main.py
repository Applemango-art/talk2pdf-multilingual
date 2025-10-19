import os
import time
import io
import base64
import wave
import requests
from PyPDF2 import PdfReader
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

from google import genai
from google.genai import types

# -----------------------------
# Env & clients
# -----------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)

# -----------------------------
# PDF ingestion
# -----------------------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    # ---- NEW: manual batching to avoid Gemini API errors ----
    def embed_in_batches(chunks, batch_size=10):
        all_vectors = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            try:
                batch_vectors = embeddings.embed_documents(batch)
                all_vectors.extend(batch_vectors)
            except Exception as e:
                print(f"Batch {i//batch_size + 1} failed: {e}")
        return all_vectors

    # Optional cleanup (filter very small garbage chunks from repeated headers, etc.)
    text_chunks = [c for c in text_chunks if len(c.strip()) > 20]

    vectors = embed_in_batches(text_chunks, batch_size=10)

    
    store = FAISS.from_embeddings(
        embeddings=list(zip(vectors, text_chunks)),
        embedding=embeddings
    )

    store.save_local("faiss_index")
    return store


# -----------------------------
# RAG: Gemini QA chain
# -----------------------------
def get_conversational_chain(lang_code):
    lang_instructions = {
        "en": "Answer in English as detailed as possible from the provided context.",
        "hi": "Answer in Hindi as detailed as possible from the provided context.",
        "bn": "Answer in Bengali as detailed as possible from the provided context.",
        "mr": "Answer in Marathi as detailed as possible from the provided context.",
        "ta": "Answer in Tamil as detailed as possible from the provided context.",
        "te": "Answer in Telugu as detailed as possible from the provided context.",
        "es": "Answer in Spanish as detailed as possible from the provided context.",
        "fr": "Answer in French as detailed as possible from the provided context."
    }.get(lang_code, "Answer in English as detailed as possible from the provided context.")

    prompt_template = f"""{lang_instructions}
If the answer is not in provided context say, "answer is not available in the context", don't guess.

Context:
{{context}}

Question:
{{question}}

Answer:
"""
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def answer_question(user_question, target_language_code):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = store.similarity_search(user_question)
    chain = get_conversational_chain(target_language_code)
    resp = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return resp["output_text"]

# -----------------------------
# Gemini TTS
# -----------------------------
VOICE_BY_LANG = {
    "en": "kore",
    "hi": "kore",
    "bn": "puck",
    "mr": "puck",
    "ta": "puck",
    "te": "puck",
    "es": "zephyr",
    "fr": "charon",
}

def pcm_to_wav_bytes(pcm_bytes, sample_rate=24000, num_channels=1, sample_width=2):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(num_channels)
        w.setsampwidth(sample_width)
        w.setframerate(sample_rate)
        w.writeframes(pcm_bytes)
    buf.seek(0)
    return buf

def gemini_tts(text, voice_name="kore"):
    resp = client.models.generate_content(
        model="gemini-2.5-flash-preview-tts",
        contents=text,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
                )
            )
        )
    )
    data = resp.candidates[0].content.parts[0].inline_data.data
    audio_bytes = base64.b64decode(data) if isinstance(data, str) else data
    return io.BytesIO(audio_bytes) if audio_bytes[:4] == b"RIFF" else pcm_to_wav_bytes(audio_bytes)

# -----------------------------
# AssemblyAI STT (chosen language)
# -----------------------------
def assemblyai_transcribe_bytes(file_bytes, app_lang_code):
    headers = {"authorization": ASSEMBLYAI_API_KEY}
    assemblyai_languages = {
        "en": "en", "hi": "hi", "bn": "bn", "mr": "mr",
        "ta": "ta", "te": "te", "es": "es", "fr": "fr"
    }
    chosen_lang = assemblyai_languages.get(app_lang_code, "en")

    up = requests.post(
        "https://api.assemblyai.com/v2/upload",
        headers=headers,
        data=file_bytes
    )
    up.raise_for_status()
    audio_url = up.json()["upload_url"]

    job = requests.post(
        "https://api.assemblyai.com/v2/transcript",
        headers=headers,
        json={"audio_url": audio_url, "language_code": chosen_lang, "speaker_labels": False}
    )
    job.raise_for_status()
    job_id = job.json()["id"]

    while True:
        res = requests.get(
            f"https://api.assemblyai.com/v2/transcript/{job_id}",
            headers=headers
        )
        res.raise_for_status()
        j = res.json()
        status = j["status"]
        if status == "completed":
            return j["text"]
        if status == "error":
            raise RuntimeError(j.get("error", "Transcription failed"))
        time.sleep(2)




