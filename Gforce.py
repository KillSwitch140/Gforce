import streamlit as st
import os
import PyPDF2
import re
import spacy
import openai

# Set up your OpenAI API key from Streamlit secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Function to read PDF text
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
    # Assume the candidate name is in the first line of the resume text
    first_line = resume_text.strip().split('\n')[0]
    
    # Initialize spaCy NER model
    nlp = spacy.load("en_core_web_sm")
    
    # Process the first line with spaCy NER
    doc = nlp(first_line)
    candidate_name = None
    
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            candidate_name = ent.text
            break

    # If spaCy NER did not find a PERSON entity in the first line, use the entire first line as the candidate name
    if not candidate_name:
        candidate_name = first_line.strip()
        
    return candidate_name

# Page title and styling
st.set_page_config(page_title='GForce Resume Reader', layout='wide')
st.title('GForce Resume Reader')

# List to store uploaded resume contents and extracted information
uploaded_resumes = []
candidates_info = []

# File upload
uploaded_files = st.file_uploader('Please upload your resume', type='pdf', accept_multiple_files=True)

# Add a separate section in the sidebar to get job details
st.sidebar.header('Job Details')
job_title = st.sidebar.text_input("Enter the job title:")
qualifications = st.sidebar.text_area("Enter the qualifications for the job (separated by commas):")

# Display job details in the sidebar
st.sidebar.write(f'Job Title: {job_title}')
st.sidebar.write(f'Qualifications: {", ".join(qualifications)}')

# Add a button to trigger candidate selection
select_candidates = st.sidebar.button('Select Candidates', key="select_candidates")

# Function to generate assistant response
def generate_response(api_key, query_text, job_title, qualifications, candidates_info):
    # Prepare the conversation history with system message introducing the bot's role
    conversation_history = [
        {'role': 'system', 'content': 'Hello! I am your recruiter assistant. My role is to go through resumes and help recruiters make informed decisions.'},
        {'role': 'user', 'content': query_text}
    ]

    # Process resumes and store the summaries in candidates_info
    for idx, candidate_info in enumerate(candidates_info):
        resume_text = candidate_info["resume_text"]
        conversation_history.append({'role': 'system', 'content': f'Resume {idx + 1}: {resume_text}'})
    
    # Check if the user query is related to selecting candidates based on qualifications
    if "select_candidates" in query_text:
        # Add a prompt for selecting candidates based on qualifications
        prompt = "Based on the qualifications you provided, please recommend the top candidates."
        conversation_history.append({'role': 'user', 'content': prompt})

    # Use GPT-3.5-turbo for recruiter assistant tasks based on prompts
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation_history,
        api_key=api_key
    )

    # Get the assistant's response
    assistant_response = response['choices'][0]['message']['content']
    return assistant_response

# User query
user_query = st.text_area('You (Type your message here):', value='', help='Ask away!', height=100, key="user_input")

# Form input and query
send_user_query = st.button('Send', help='Click to submit the query', key="send_user_query")
if send_user_query:
    if user_query.strip() != '':
        with st.spinner('Chatbot is typing...'):
            # Add the user query to the conversation history
            st.session_state.conversation_history.append({'role': 'user', 'content': user_query})
            
            # Check if the bot needs to ask the qualification question
            if len(candidates_info) > 0 and not any("Based on the qualifications" in message["content"] for message in st.session_state.conversation_history):
                # Add the qualification question to the conversation history
                if job_title and qualifications:
                    qualifications_str = ", ".join(qualifications)
                    st.session_state.conversation_history.append({'role': 'system', 'content': f'Great! You are looking for candidates for the position of {job_title} with qualifications in {qualifications_str}. How can I assist you?'})
                else:
                    st.session_state.conversation_history.append({'role': 'system', 'content': 'What qualifications are you looking for in a candidate?'})
            
            # Get the updated conversation history
            conversation_history = st.session_state.conversation_history.copy()
            # Append the uploaded resumes' content to the conversation history
            conversation_history.extend([{'role': 'system', 'content': resume_text} for resume_text in uploaded_resumes])
            # Generate the response using the updated conversation history
            response = generate_response(openai_api_key, user_query, job_title, qualifications, candidates_info)
            # Append the assistant's response to the conversation history
            st.session_state.conversation_history.append({'role': 'assistant', 'content': response})

# Check if the user clicked the "Select Candidates" button
if select_candidates:
    with st.spinner('Selecting candidates...'):
        # Generate the response for candidate selection based on the job title and qualifications
        select_candidates_query = "select_candidates"
        response = generate_response(openai_api_key, select_candidates_query, job_title, qualifications, candidates_info)
        # Append the assistant's response to the conversation history
        st.session_state.conversation_history.append({'role': 'assistant', 'content': response})


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
