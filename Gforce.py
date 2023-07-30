import streamlit as st
import os
import pdfplumber
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
def generate_response(uploaded_file, openai_api_key, query_text):
    # Load document if file is uploaded
    if uploaded_file is not None:
        df = extract_data(uploaded_file)
        documents = [df.read().decode()]
    # Split documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=10000000, chunk_overlap=0)
    texts = text_splitter.create_documents(documents)
    # Select embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    # Create a vectorstore from documents
    db = Chroma.from_documents(texts, embeddings)
    # Create retriever interface
    retriever = db.as_retriever()
    # Create QA chain
    qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key), chain_type='stuff', retriever=retriever)
    return qa.run(query_text)

# Page title
st.set_page_config(page_title='Gforce Resume Reader')
st.title('Gforce Resume Reader')

# File upload
uploaded_file  = st.file_uploader('Please upload you resumes', type='pdf')

def extract_data(feed):
    data = []
    with pdfplumber.load(feed) as pdf:
        pages = pdf.pages
        for p in pages:
            data.append(p.extract_tables())
    return None # build more code to return a dataframe
# Query text
query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.', disabled=not uploaded_file)

with st.form('myform', clear_on_submit=True):
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    submitted = st.form_submit_button('Submit', disabled=not(uploaded_file and query_text))
    if submitted and openai_api_key.startswith('sk-'):
        with st.spinner('Calculating...'):
            response = generate_response(uploaded_file, openai_api_key, query_text)
            result.append(response)
            del openai_api_key
# Form input and query
result = []

def create_temp_file(uploaded_file):
    """
    Create a temporary file from an uploaded file.

    :param uploaded_file: The uploaded file to create a temporary file from.

    :return: The path to the temporary file.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as temp_file:
        if uploaded_file.type == 'application/pdf':
            temp_file.write(pdf_to_text(uploaded_file))
        else:
            temp_file.write(uploaded_file.getvalue())
    return temp_file.name
if len(result):
    st.info(response)


def pdf_to_text(pdf_file):
    """
    Convert a PDF file to a string of text.

    :param pdf_file: The PDF file to convert.

    :return: A string of text.
    """
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = StringIO()
    for i in range(len(pdf_reader.pages)):
        p = pdf_reader.pages[i]
        text.write(p.extract_text())
    return text.getvalue().encode('utf-8')
