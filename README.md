# talk2pdf-multilingual (🌐 [Live Demo](https://sannidhya-das-talk2pdf-multilingual.streamlit.app/) )

![App Interface 1](https://github.com/SannidhyaDas/talk2pdf-multilingual/blob/main/page_images/appInterface_1.png)    

![App Interface 2](https://github.com/SannidhyaDas/talk2pdf-multilingual/blob/main/page_images/appInterface_2.png)

## 🧠 Chat & Talk with PDF (Gemini + AssemblyAI)

Interact with your PDFs using **chat or voice** — in **multiple languages**!  
This app combines **Google Gemini (LLMs + TTS)**, **AssemblyAI (Speech-to-Text)**, and **LangChain + FAISS** for a full multimodal RAG experience.

---

## 🚀 Features

✅ **PDF Chatting (RAG)** — Upload one or more PDFs, and ask questions about their content.  
✅ **Voice Chat Mode** — Speak your question, get spoken answers back.  
✅ **Multilingual Support** — Works in 8 languages: English, Hindi, Bengali, Marathi, Tamil, Telugu, Spanish, and French.  
✅ **Speech-to-Text (AssemblyAI)** — Converts voice input to text in your chosen language.  
✅ **Text-to-Speech (Gemini TTS)** — Speaks the AI’s answers naturally.  
✅ **LangChain + FAISS Vector Store** — Efficient retrieval of answers from large PDFs.  
✅ **Streamlit UI** — Modern and minimal web interface.

---

## 🧩 Tech Stack

| Component | Technology Used |
|------------|----------------|
| **LLM & Embeddings** | Google Gemini 2.5 Flash + Gemini Embeddings |
| **Text Split & QA Chain** | LangChain (Recursive Splitter, FAISS, QA Chain) |
| **Speech-to-Text** | [AssemblyAI API](https://www.assemblyai.com) |
| **Text-to-Speech** | Gemini 2.5 Flash Preview (Audio Modality) |
| **Frontend** | Streamlit |
| **Vector Store** | FAISS |
| **Environment** | Python, `.env` for API Keys |

---

## 🧠 Architecture Overview

Below is the dual-mode pipeline of Chat & Talk with PDF, showing how text and voice inputs flow through RAG and TTS/STT modules.

![Model Pipeline](https://github.com/SannidhyaDas/talk2pdf-multilingual/blob/main/page_images/chatRAG_pipeline.drawio.png)

### 🧾 Key Pipeline Summary

| Stage                    | Component                               | Function                        |
| ------------------------ | --------------------------------------- | ------------------------------- |
| **1. PDF Processing**    | `PyPDF2`, `LangChain`                   | Extracts and chunks PDF text    |
| **2. Embedding & Store** | `GoogleGenerativeAIEmbeddings`, `FAISS` | Creates searchable vector DB    |
| **3. Query Retrieval**   | `FAISS.similarity_search()`             | Finds relevant chunks           |
| **4. QA Generation**     | `Gemini (ChatGoogleGenerativeAI)`       | Generates language-aware answer |
| **5. Voice Input**       | `AssemblyAI API`                        | Converts speech → text          |
| **6. Voice Output**      | `Gemini TTS`                            | Converts text → speech          |
| **7. Frontend**          | `Streamlit`                             | Interactive UI                  |


---
## 📦 Installation

1. **Clone this repository**

```bash
git clone https://github.com/SannidhyaDas/talk2pdf-multilingual.git
cd talk2pdf-multilingual
```
2. Create a virtual environment and install dependencies

```bash
python -m venv venv
venv\Scripts\activate       # on Windows
# or source venv/bin/activate  # on Mac/Linux

pip install -r requirements.txt

```
3. Set up environment variables (.env file)

```ini
GOOGLE_API_KEY=your_google_gemini_api_key
ASSEMBLYAI_API_KEY=your_assemblyai_api_key
```
4. Run the Streamlit app

```bash    
streamlit run app.py
```
---
## 🧾 Usage
### 💬 Chat with PDF

1. Upload one or more PDF files in the sidebar.
2. Click Submit & Process.
3. Ask questions in your selected language.

### 🎤 Talk with PDF

1. Switch to Talk with PDF mode from the sidebar.
2. Upload or record an audio file (wav/mp3/m4a).
3. The app transcribes your question, searches your PDFs, and answers — both in text and audio.

## 🌍 Supported Languages

Although Gemini supports 30+ languages and AssemblyAI offers over 90 languages, this application currently provides the user with a choice among 8 select languages.

| Language | Code | TTS Voice |
| -------- | ---- | --------- |
| English  | `en` | kore      |
| Hindi    | `hi` | kore      |
| Bengali  | `bn` | puck      |
| Marathi  | `mr` | puck      |
| Tamil    | `ta` | puck      |
| Telugu   | `te` | puck      |
| Spanish  | `es` | zephyr    |
| French   | `fr` | charon    |

## 🧰 Key Files
```bash
talk2pdf-multilingual/
│
├── main.py              # Core logic for PDF processing, embeddings, RAG QA chain, Speech-to-Text, and Text-to-Speech.
├── app.py               # Streamlit front-end — handles user interaction and integrates chat & talk modes.
├── v1app.py             # same logics and functions but without voice chat feature. (version 1) 
├── test.ipynb           # testing script with examples and explanations. 
├── requirements.txt            # Python dependencies
├── app_images/             # working pipeline and app interface .png files
│   ├── appInterface_1.png
│   ├── appInterface_2.png
│   └── chatRAGpipeline.drawio.png        # Main working pipeline
├── README.md                   # Project documentation
└── .env                  # API keys for Gemini and AssemblyAI. (Non-shareable/hidden)
```
---
## 📚 Example Use Cases

This project demonstrates how Generative AI can move beyond experimentation and deliver *real business and societal impact*.  
Below are some practical applications that show how such an AI assistant can enhance productivity, accessibility, and decision-making across domains:

- 📄 **Interactive Research Companion:**  
  Seamlessly study or summarize complex academic PDFs, extracting key insights and simplifying technical language for faster understanding.  
  Ideal for students, researchers, and data analysts who deal with dense, information-heavy documents.

- 🎧 **Voice-Based Q&A for Accessibility:**  
  Enables hands-free, voice-driven interaction with documents — making AI assistance more inclusive for users with visual impairments or those multitasking.  
  This feature bridges technology with accessibility, turning information into a truly universal resource.

- 🌐 **Multilingual Knowledge Assistant:**  
  Supports multiple languages for document understanding and interaction, allowing teams across geographies to collaborate effortlessly.  
  This promotes global reach and knowledge democratization within enterprises.

- 🧾 **Enterprise Knowledge Base Querying:**  
  Acts as a smart interface to corporate documentation, product manuals, or client data — helping employees instantly retrieve critical information.  
  Reduces search time, improves onboarding, and supports better business decisions through natural language queries.

