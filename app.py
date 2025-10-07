
import streamlit as st
from main import (
    get_pdf_text, get_text_chunks, get_vector_store,
    answer_question, gemini_tts, assemblyai_transcribe_bytes, VOICE_BY_LANG
)

def main():
    st.set_page_config(page_title="Chat & Talk with PDF (Gemini + AssemblyAI)")

    mode_option = st.sidebar.radio("Select Mode:", ("Chat with PDF", "Talk with PDF"))
    st.sidebar.title("Menu:")
    pdf_docs = st.sidebar.file_uploader(
        "Upload your PDF Files and Click on Submit & Process",
        accept_multiple_files=True
    )

    lang_options = {
        "English": "en", "Hindi": "hi", "Bengali": "bn", "Marathi": "mr",
        "Tamil": "ta", "Telugu": "te", "Spanish": "es", "French": "fr"
    }
    chosen_language = st.sidebar.selectbox(
        "Choose your preferred language",
        options=list(lang_options.keys()),
        index=0
    )
    selected_lang_code = lang_options[chosen_language]
    process_button = st.sidebar.button("✨ Submit & Process")

    if process_button:
        if pdf_docs:
            with st.spinner("Processing PDFs..."):
                raw_text = get_pdf_text(pdf_docs)
                chunks = get_text_chunks(raw_text)
                _ = get_vector_store(chunks)
                st.session_state.vector_store_ready = True
                st.session_state.messages = []
                st.session_state.chat_language = selected_lang_code
            st.success("Ready to interact.")
        else:
            st.sidebar.error("Upload PDFs first!")

    if not st.session_state.get("vector_store_ready"):
        st.info("Upload and process PDFs to start.")
        return

    if mode_option == "Chat with PDF":
        st.subheader(f"Chat with PDF in {chosen_language}")
        for msg in st.session_state.get("messages", []):
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
        if prompt := st.chat_input(f"Ask questions (in {chosen_language})"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Generating response..."):
                    answer = answer_question(prompt, st.session_state.chat_language)
                st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

    elif mode_option == "Talk with PDF":
        st.subheader(f"Talk with PDF in {chosen_language}")
        st.markdown("###### Record audio on your device or use an [online recorder](https://online-voice-recorder.com/), then upload below:")
        audio_file = st.file_uploader("Record/Upload voice (wav/mp3/m4a)", type=["wav", "mp3", "m4a"])
        typed_fallback = st.text_area("Or type your message...", height=120)
        query_text = None

        if audio_file is not None:
            with st.spinner("Transcribing your audio..."):
                try:
                    query_text = assemblyai_transcribe_bytes(audio_file.read(), selected_lang_code)
                    st.write("You said:", query_text)
                except Exception as e:
                    st.error(f"Transcription failed: {e}")

        if not query_text and typed_fallback:
            query_text = typed_fallback

        if query_text:
            st.session_state.messages.append({"role": "user", "content": query_text})
            with st.chat_message("user"):
                st.write(query_text)

            with st.spinner("Searching and answering from your PDFs..."):
                answer_text = answer_question(query_text, st.session_state.chat_language)

            st.write("Answer:", answer_text)

            voice_name = VOICE_BY_LANG.get(st.session_state.chat_language, "kore")
            with st.spinner("Generating voice reply..."):
                try:
                    audio_stream = gemini_tts(answer_text, voice_name=voice_name)
                    st.audio(audio_stream, format="audio/wav")
                except Exception as e:
                    st.error(f"TTS failed: {e}")

            st.session_state.messages.append({"role": "assistant", "content": answer_text})

        for msg in st.session_state.get("messages", []):
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

if __name__ == "__main__":
    main()
