from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tempfile
from pypdf import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from io import BytesIO
import streamlit as st
import tempfile
import os
from langchain_community.vectorstores import FAISS
import requests
# from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, AzureChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()

# Azure OpenAI settings
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")

app = FastAPI()

def load_document(text):
    # with tempfile.SpooledTemporaryFile(mode='wb') as tmp_file:
    #     tmp_file.write(file.read())
    #     tmp_file.seek(0)
    # loader = PyPDFLoader(tmp_file.name)
    # documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
    # docs = text_splitter.split_documents(documents)
    docs= text_splitter.create_documents([text])
    return docs

def create_vector_db(docs, path):
  embedding_function = AzureOpenAIEmbeddings(
      openai_api_type="azure",
      openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
      azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
      deployment="text-embedding-ada-002",
      model="text-embedding-ada-002",
  )
  db = FAISS.from_documents(docs, embedding_function)
  db.save_local(path)
  return db

def load_vector_db(path):
  embedding_function = AzureOpenAIEmbeddings(
      openai_api_type="azure",
      openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
      azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
      deployment="text-embedding-ada-002",
      model="text-embedding-ada-002",
  )
  return FAISS.load_local(
      path, embedding_function, allow_dangerous_deserialization=True
  )


def genrating_eligbility(rfp_text):
  llm = AzureChatOpenAI(
      openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
      openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
      azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
      azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
      temperature=0,
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

  class EligibilityCriterion(BaseModel):
      criterion: str = Field(
          ..., description="Description of the eligibility criterion."
      )
      eligibility_met: str = Field(
          ..., description="Whether the eligibility criterion is met (Yes/No)."
      )
      reason: str = Field(..., description="Reason for the eligibility status.")

  class EligibilityData(BaseModel):
      eligibility_criteria: List[EligibilityCriterion]

  parser = JsonOutputParser(pydantic_object=EligibilityData)

  llm = AzureChatOpenAI(
      openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
      openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
      azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
      azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
      temperature=0,
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
  \n{format_instructions}
  """

  prompt = PromptTemplate(
      template=template,
      input_variables=["rfp_content", "proposal_content"],
      partial_variables={"format_instructions": parser.get_format_instructions()},
  )

  json_chain = prompt | llm | parser

  result = json_chain.invoke(
      {"rfp_content": rfp_content, "proposal_content": proposal_content}
  )

  return result

@app.post("/count-characters/")
async def count_characters(rfp_file: UploadFile = File(...), proposal_file: UploadFile = File(...)):
    if not all([AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT, AZURE_DEPLOYMENT]):
      raise HTTPException(status_code=500, detail="Azure OpenAI settings are not properly configured")
    try:
        # Read the uploaded file
        r_content = await rfp_file.read()
        p_content= await proposal_file.read()
        r_pdf_file = BytesIO(r_content)
        p_pdf_file = BytesIO(p_content)
    
        
        # Create a PDF reader object
        r_pdf_reader = PdfReader(r_pdf_file)
        p_pdf_reader = PdfReader(p_pdf_file)
        
        # Extract text from all pages
        # Load and process RFP
        
        
        r_text = ""
        for page in r_pdf_reader.pages:
            r_text += page.extract_text()
        r_docs=load_document(r_text)
        rfp_db = create_vector_db(r_docs, f"vectorstore/RFP/{rfp_file.filename}")

        # Extract text from all pages
        # Load and process Proposal doc
        p_text = ""
        for page in p_pdf_reader.pages:
            p_text += page.extract_text()
        p_docs=load_document(p_text)
        proposal_db = create_vector_db(p_docs, f"vectorstore/Proposals/{proposal_file.filename}")

        # Retrieve relevant content
        rfp_content = rfp_db.similarity_search("eligibility criteria", k=20)
        proposal_content = proposal_db.similarity_search("company background and qualifications", k=20)

        # Combine retrieved content
        rfp_text = " ".join([doc.page_content for doc in rfp_content])
        proposal_text = " ".join([doc.page_content for doc in proposal_content])

        # Generate RFP Eligibility
        ref_eligibility = genrating_eligbility(rfp_text)

        # Analyze eligibility
        analysis = analyze_eligibility(ref_eligibility, proposal_text)
        

        return {"Result":analysis}

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)