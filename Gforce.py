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

# Function to extract past experience using GPT-3's prompt
def extract_experience(resume_text):
    prompt = f"Please provide your past experience:"
    response = openai.Completion.create(
        model="gpt-3.5-turbo",
        prompt=prompt,
        temperature=0,
        max_tokens=100,
        stop=["\n"],
        api_key=openai_api_key
    )
    experience = response['choices'][0]['text'].replace(prompt, "").strip()
    return experience

# Function to extract candidate name using GPT-3.5-turbo model
def extract_candidate_name(resume_text):
    prompt = f"What is candidate's name?"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": resume_text},
        ],
        api_key=openai_api_key
    )
    candidate_name = response['choices'][0]['message']['content'].strip()
    return candidate_name

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
    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            resume_text = read_pdf_text(uploaded_file)
            uploaded_resumes.append(resume_text)
            # Extract GPA, email, and past experience
            gpa = extract_gpa(resume_text)
            email = extract_email(resume_text)
            experience = extract_experience(resume_text)
            # Extract candidate name using GPT-3's prompt
            candidate_name = extract_candidate_name(resume_text)
            # Store the information for each candidate
            candidate_info = {
                'name': candidate_name,
                'gpa': gpa,
                'email': email,
                'experience': experience,
            }
            candidates_info.append(candidate_info)

# Display extracted information for each candidate
if candidates_info:
    st.subheader('Extracted Information for Each Candidate:')
    for candidate_info in candidates_info:
        st.markdown(f'**{candidate_info["name"]}:**')
        st.markdown(f'- GPA: {candidate_info["gpa"]}')
        st.markdown(f'- Email: {candidate_info["email"]}')
        st.markdown(f'- Past Experience:')
        st.text(candidate_info["experience"])
# Retrieve or initialize conversation history using SessionState
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# User query
user_query = st.text_area('You (Type your message here):', value='', help='Ask away!', height=100, key="user_input")

# Form input and query
send_user_query = st.button('Send', help='Click to submit the query', key="send_user_query")
if send_user_query:
    if user_query.strip() != '':
        with st.spinner('Chatbot is typing...'):
            # Add the user query to the conversation history
            st.session_state.conversation_history.append({'role': 'user', 'content': user_query})
            # Get the updated conversation history
            conversation_history = st.session_state.conversation_history.copy()
            # Append the uploaded resumes' content to the conversation history
            conversation_history.extend([{'role': 'system', 'content': resume_text} for resume_text in uploaded_resumes])
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
        padding: 5px;
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
