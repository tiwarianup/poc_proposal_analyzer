import streamlit as st
import tempfile
import os
from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, AzureChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(layout="wide")

# Sidebar
with st.sidebar:
  st.image("./images/pwc.png")
  st.title("Contract Analyzer")

# Azure OpenAI settings
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")

def load_document(file):
  with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
      tmp_file.write(file.getvalue())
      tmp_file_path = tmp_file.name

  loader = PyPDFLoader(tmp_file_path)
  documents = loader.load()
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
  docs = text_splitter.split_documents(documents)
  
  os.unlink(tmp_file_path)
  return docs

def create_vector_db(docs, path):
  embedding_function = AzureOpenAIEmbeddings(
            openai_api_type = "azure",
            openai_api_key = os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
            deployment = "text-embedding-ada-002",
            model = "text-embedding-ada-002"
        )
  db = FAISS.from_documents(docs, embedding_function)
  db.save_local(path)
  return db

def load_vector_db(path):
  embedding_function = AzureOpenAIEmbeddings(
            openai_api_type = "azure",
            openai_api_key = os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
            deployment = "text-embedding-ada-002",
            model = "text-embedding-ada-002"
        )
  return FAISS.load_local(path, embedding_function, allow_dangerous_deserialization=True)

def genrating_eligbility(rfp_text):
    llm= AzureChatOpenAI(
            openai_api_key = os.getenv("AZURE_OPENAI_API_KEY"),
            openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
            temperature = 0,
            
        )

    template = """
    You are an AI contract analyzer. Your task is to Find all the eligibility criteria listed in a Request for Proposal (RFP).

    RFP Content:
    {rfp_text}

    """

    prompt = PromptTemplate(input_variables=["rfp_text"], template=template)
    chain = LLMChain(llm=llm, prompt=prompt)

    return chain.run(rfp_text=rfp_text)

def analyze_eligibility(rfp_content, proposal_content):
  llm= AzureChatOpenAI(
                openai_api_key = os.getenv("AZURE_OPENAI_API_KEY"),
                openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION"),
                azure_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
                azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
                temperature = 0,
                
            )

  template = """
  You are an AI contract analyzer. Your task is to compare the eligibility criteria listed in a Request for Proposal (RFP) with the details provided in a proposal.

  RFP Content:
  {rfp_content}

  Proposal Content:
  {proposal_content}

  For each eligibility criterion in the RFP, provide the following:

  Eligibility Criterion: [Insert eligibility criterion from RFP]
  Eligibility Met (Yes/No): [Yes/No]
  Reason: [Provide a detailed explanation of how the eligibility criterion is met or not met based on the proposal]

  Please provide your analysis in a clear, structured format.
  """

  prompt = PromptTemplate(input_variables=["rfp_content", "proposal_content"], template=template)
  chain = LLMChain(llm=llm, prompt=prompt)
  
  return chain.run(rfp_content=rfp_content, proposal_content=proposal_content)

def main():
  st.title("Contract Analyzer")

  col1, col2 = st.columns(2)

  with col1:
      rfp_file = st.file_uploader("Upload RFP (PDF)", type="pdf")
  with col2:
      proposal_file = st.file_uploader("Upload Proposal (PDF)", type="pdf")

  if rfp_file and proposal_file:
      if st.button("Analyze"):
          with st.spinner("Analyzing documents..."):
              # Load and process RFP
              rfp_docs = load_document(rfp_file)
              rfp_db = create_vector_db(rfp_docs, "vectorstore/RFP/")
              
              # Load and process Proposal
              proposal_docs = load_document(proposal_file)
              proposal_db = create_vector_db(proposal_docs, "vectorstore/Proposals/")
              
              # Retrieve relevant content
              rfp_content = rfp_db.similarity_search("eligibility criteria", k=20)
              proposal_content = proposal_db.similarity_search("company background and qualifications", k=20)
              
              # Combine retrieved content
              rfp_text = " ".join([doc.page_content for doc in rfp_content])
              proposal_text = " ".join([doc.page_content for doc in proposal_content])

              #Generate RFP Elgibility
              ref_eligibility= genrating_eligbility(rfp_text)
              
              # Analyze eligibility
              analysis = analyze_eligibility(ref_eligibility, proposal_text)
              
              st.subheader("Eligibility Analysis")
              st.write(analysis)

if __name__ == "__main__":
  main()