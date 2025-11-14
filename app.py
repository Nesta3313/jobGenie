import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_classic.agents import initialize_agent
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import Tool
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

api_key = os.getenv('API_KEY')

st.set_page_config(page_title='Career Coach Agent', page_icon='🧞‍♂️')
st.title('🧞‍♂️ Job Genie Agent')
st.write('Your personal AI career assistant — ask questions, get insights, or upload your resume for feedback!')

# base llm
llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.3, api_key=api_key)

# search tool
search_tool = DuckDuckGoSearchRun()

uploaded_file = st.file_uploader('Upload your resume (PDF or TXT)', type=["pdf", "txt"])

retriever_tool = None

if uploaded_file is not None:
    if uploaded_file.name.endswith('.pdf'):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        loader = PyPDFLoader(tmp_file_path)
    else:
        text = uploaded_file.read().decode('utf-8')
        with open('resume_temp.txt', 'w') as f:
            f.write(text)
        loader = TextLoader('resume_temp.txt')

    docs = loader.load()

    # split and chunk
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    # embed chunks and store in FAISS
    embeddings = OpenAIEmbeddings(model=os.getenv('EMBED_MODEL'))
    db = FAISS.from_documents(documents=chunks, embedding=embeddings)

    retriever_tool = db.as_retriever()


tools = [
    Tool(
        name='Job Search',
        func=search_tool.run,
        description='Search for job openings or company information.'
    )
]

if retriever_tool:
    prompt = ChatPromptTemplate.from_template(
        "Use the following resume content to answer:\n\n{context}\n\nQuestion: {input}")

    # Build chain components
    doc_chain = create_stuff_documents_chain(llm, prompt)
    resume_qa = create_retrieval_chain(retriever_tool, doc_chain)


    def analyze_resume(query: str):
        response = resume_qa.invoke({"input": query})
        return response.get("answer", response)


    tools.append(
        Tool(
            name='Resume Analyzer',
            func=analyze_resume,
            description='Analyze or extract information from the uploaded resume.'
        )
    )

agent = initialize_agent(tools, llm, agent_type="zero-shot-react-description", verbose=True)

query = st.text_area("Ask your Career Coach:", placeholder="e.g., 'What jobs fit my skills?' or 'Review my resume for improvements.'")

if st.button("Get Advice"):
    if not query.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Thinking..."):
            response = agent.run(query)
            st.success(response)







