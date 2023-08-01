import pysqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import os
import PyPDF2
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from PyPDF2 import PdfReader
import openai
import re
import spacy


# Set up your OpenAI API key from Streamlit secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]


def read_pdf_text(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""

    for page in pdf_reader.pages:
        text += page.extract_text()

    return text

# Function to extract GPA using regular expression
def extract_gpa(text):
    gpa_pattern = r"\bGPA\b\s*:\s*([\d.]+)"
    gpa_match = re.search(gpa_pattern, text, re.IGNORECASE)
    return gpa_match.group(1) if gpa_match else None

# Function to extract email using regular expression
def extract_email(text):
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    email_match = re.search(email_pattern, text)
    return email_match.group() if email_match else None

# Initialize conversation history in session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Function to extract candidate name using spaCy NER
def extract_candidate_name(resume_text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(resume_text)
    candidate_name = None
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            candidate_name = ent.text
            break
    return candidate_name

def extract_previous_companies_ner(resume_text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(resume_text)
    previous_companies = []
    for ent in doc.ents:
        if ent.label_ == "ORG":
            previous_companies.append(ent.text)
    return previous_companies

# Function to extract schools using spaCy NER
def extract_schools_ner(resume_text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(resume_text)
    schools = []
    for ent in doc.ents:
        if ent.label_ == "ORG" and "school" in ent.text.lower():
            schools.append(ent.text)
    return schools

# Page title and styling
st.set_page_config(page_title='GForce Resume Reader', layout='wide')
st.title('GForce Resume Reader')

# List to store uploaded resume contents and extracted information
uploaded_resumes = []
candidates_info = []

# File upload
uploaded_files = st.file_uploader('Please upload your resume', type='pdf', accept_multiple_files=True)

# Process uploaded resumes
if uploaded_files:
    for idx, uploaded_file in enumerate(uploaded_files):
        if uploaded_file is not None:
            resume_text = read_pdf_text(uploaded_file)
            uploaded_resumes.append(resume_text)
            # Extract GPA, email, and past experience
            gpa = extract_gpa(resume_text)
            email = extract_email(resume_text)
            # Extract candidate name using spaCy NER
            candidate_name = extract_candidate_name(resume_text)
            # Store the information for each candidate
            candidate_info = {
                'name': candidate_name,
                'gpa': gpa,
                'email': email,
            }
            candidates_info.append(candidate_info)

            # Add context for each candidate using the candidate's name
            st.session_state.conversation_history.append({'role': 'system', 'content': f"Context for {candidate_name}:"})
            st.session_state.conversation_history.append({'role': 'user', 'content': resume_text})

# User query
user_query = st.text_area('You (Type your message here):', value='', help='Ask away!', height=100, key="user_input")

# Form input and query
send_user_query = st.button('Send', help='Click to submit the query', key="send_user_query")
if send_user_query:
    if user_query.strip() != '':
        with st.spinner('Chatbot is typing...'):
            # Get the updated conversation history
            conversation_history = st.session_state.conversation_history.copy()
            # Append the user query to the conversation history
            conversation_history.append({'role': 'user', 'content': user_query})
            # Generate the response using the updated conversation history
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=conversation_history,
                api_key=openai_api_key
            )
            # Get the assistant's response
            assistant_response = response['choices'][0]['message']['content']
            # Append the assistant's response to the conversation history
            st.session_state.conversation_history.append({'role': 'assistant', 'content': assistant_response})


# Chat UI with sticky headers and input prompt
st.markdown("""
<style>
    .chat-container {
        height: 25px;
        overflow-y: scroll;
    }
    .user-bubble {
        display: flex;
        justify-content: flex-start;
    }
    .user-bubble > div {
        padding: 15px;
        background-color: #e0e0e0;
        border-radius: 10px;
        width: 50%;
        margin-left: 50%;
    }
    .assistant-bubble {
        display: flex;
        justify-content: flex-end;
    }
    .assistant-bubble > div {
        padding: 15px;
        background-color: #0078d4;
        color: white;
        border-radius: 10px;
        width: 50%;
        margin-right: 50%;
    }
    .chat-input-prompt {
        position: sticky;
        bottom: 0;
        background-color: white;
        padding: 10px;
        width: 100%;
    }
    .chat-header {
        position: sticky;
        top: 0;
        background-color: #f2f2f2;
        padding: 10px;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Display the entire conversation history in chat format
if st.session_state.conversation_history:
    for i, message in enumerate(st.session_state.conversation_history):
        if message['role'] == 'user':
            st.markdown(f'<div class="user-bubble"><div>{message["content"]}</div></div>', unsafe_allow_html=True)
        elif message['role'] == 'assistant':
            st.markdown(f'<div class="assistant-bubble"><div>{message["content"]}</div></div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Add a clear conversation button
clear_conversation = st.button('Clear Conversation', key="clear_conversation")
if clear_conversation:
    st.session_state.conversation_history.clear()
