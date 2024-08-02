from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pypdf import PdfReader
from io import BytesIO
import os
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, AzureChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Azure OpenAI settings
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")

# Initialize Azure OpenAI services
embedding_function = AzureOpenAIEmbeddings(
    openai_api_type="azure",
    openai_api_key=AZURE_OPENAI_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    deployment="text-embedding-ada-002",
    model="text-embedding-ada-002",
)

llm = AzureChatOpenAI(
    openai_api_key=AZURE_OPENAI_KEY,
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_deployment=AZURE_DEPLOYMENT,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    temperature=0,
)

def load_document(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
    return text_splitter.create_documents([text])

def create_vector_db(docs, path):
    db = FAISS.from_documents(docs, embedding_function)
    db.save_local(path)
    return db

def load_vector_db(path):
    return FAISS.load_local(path, embedding_function, allow_dangerous_deserialization=True)

def generating_eligibility(rfp_text):
    template = """
    You are an AI contract analyzer. Your task is to find all the eligibility criteria listed in a Request for Proposal (RFP).

    RFP Content:
    {rfp_text}
    """
    prompt = PromptTemplate(input_variables=["rfp_text"], template=template)
    # chain = LLMChain(llm=llm, prompt=prompt)
    # return chain.run(rfp_text=rfp_text)
    chain = prompt | llm
    return chain.invoke({"rfp_text": rfp_text})

class EligibilityCriterion(BaseModel):
    criterion: str = Field(..., description="Description of the eligibility criterion.")
    eligibility_met: str = Field(..., description="Whether the eligibility criterion is met (Yes/No).")
    reason: str = Field(..., description="Reason for the eligibility status.")

class EligibilityData(BaseModel):
    eligibility_criteria: List[EligibilityCriterion]

def analyze_eligibility(rfp_content, proposal_content):
    parser = JsonOutputParser(pydantic_object=EligibilityData)
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
    return json_chain.invoke({"rfp_content": rfp_content, "proposal_content": proposal_content})

@app.post("/analyze/")
async def analyze_eligibility_endpoint(rfp_file: UploadFile = File(...), proposal_file: UploadFile = File(...)):
    if not all([AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT, AZURE_DEPLOYMENT]):
        raise HTTPException(status_code=500, detail="Azure OpenAI settings are not properly configured")
    
    try:
        r_content = await rfp_file.read()
        p_content = await proposal_file.read()
        
        r_text = extract_text_from_pdf(BytesIO(r_content))
        p_text = extract_text_from_pdf(BytesIO(p_content))
        
        r_docs = load_document(r_text)
        p_docs = load_document(p_text)
        
        rfp_db = create_vector_db(r_docs, f"vectorstore/RFP/{rfp_file.filename}")
        proposal_db = create_vector_db(p_docs, f"vectorstore/Proposals/{proposal_file.filename}")
        
        rfp_content = " ".join([doc.page_content for doc in rfp_db.similarity_search("eligibility criteria", k=20)])
        proposal_content = " ".join([doc.page_content for doc in proposal_db.similarity_search("company background and qualifications", k=20)])
        
        ref_eligibility = generating_eligibility(rfp_content)
        analysis = analyze_eligibility(ref_eligibility, proposal_content)
        
        # return {"Result": analysis}
        return analysis
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    return " ".join(page.extract_text() for page in pdf_reader.pages)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)