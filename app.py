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
import pytesseract
import cv2
import tempfile
import speech_recognition as sr
from gtts import gTTS
import io
from datetime import datetime

# Load environment variables
load_dotenv()

# Configure Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize speech recognizer
recognizer = sr.Recognizer()

# Initialize session state for chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def cleanup_old_audio():
    """Clean up old audio files from session state"""
    if 'audio_file_path' in st.session_state:
        try:
            os.remove(st.session_state.audio_file_path)
        except (OSError, FileNotFoundError):
            pass

def text_to_speech(text):
    """Convert text to speech and return the audio file as a temporary file"""
    try:
        
        cleanup_old_audio()
        
        # Creating a temp file to store the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
            tts = gTTS(text=text, lang='en')
            tts.save(temp_audio_file.name)
            return temp_audio_file.name
    except Exception as e:
        st.error(f"Error in text-to-speech conversion: {str(e)}")
        return None

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() if page.extract_text() else ""
        images = extract_images_from_pdf(pdf)
        text += " ".join([extract_text_from_image(image) for image in images])
    return text

def extract_images_from_pdf(pdf_file):
    images = []
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(pdf_file.read())
        pdf_path = temp_pdf.name
    pdf_images = cv2.imreadmulti(pdf_path)
    for img in pdf_images[1]:
        images.append(img)
    return images

def extract_text_from_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the context, say, "Answer not available in the context." Avoid wrong answers.\n\n
    Context:\n {context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    answer = response["output_text"]
    st.write("Reply:", answer)

    # Adding each qstn answr to chat history
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.chat_history.append({
        "timestamp": timestamp,
        "question": user_question,
        "answer": answer
    })

    # Generating audio for each response
    audio_file_path = text_to_speech(answer)
    if audio_file_path:
        st.session_state.audio_file_path = audio_file_path

        st.audio(audio_file_path)

def voice_query():
    with sr.Microphone() as source:
        st.info("Listening for your question...")
        audio_data = recognizer.listen(source)
    try:
        user_question = recognizer.recognize_google(audio_data)
        st.write("You asked:", user_question)
        user_input(user_question)
    except sr.UnknownValueError:
        st.warning("Could not understand the audio.")
    except sr.RequestError as e:
        st.error(f"Could not request results; {e}")

def display_chat_history():
    """Display the chat history in a collapsible section"""
    with st.expander("View Chat History", expanded=False):
        if len(st.session_state.chat_history) == 0:
            st.info("No questions asked yet.")
        else:
            for i, chat in enumerate(reversed(st.session_state.chat_history), 1):
                st.markdown(f"### Question {len(st.session_state.chat_history) - i + 1}")
                st.markdown(f"**Time:** {chat['timestamp']}")
                st.markdown(f"**Question:** {chat['question']}")
                st.markdown(f"**Answer:** {chat['answer']}")
                st.markdown("---")

def clear_history():
    """Clear the chat history"""
    st.session_state.chat_history = []
    st.success("Chat history cleared!")

def main():
    st.set_page_config("Chat PDF with Voice & Visual Extraction")
    st.header("Gemini Document Chatbot with OCR & Voice Querying")

    if st.button("Ask a Voice Question"):
        voice_query()

    user_question = st.text_input("Ask Your PDF a Question")
    if user_question:
        user_input(user_question)

    col1, col2 = st.columns([6, 1])
    with col1:
        display_chat_history()
    with col2:
        if st.button("Clear History"):
            clear_history()

    with st.sidebar:
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Document processing complete!")

if __name__ == "__main__":
    main()