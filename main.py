import tempfile
import warnings
import streamlit as st 

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import OpenAIEmbeddings, AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI

warnings.filterwarnings("ignore")
st.set_page_config(layout="wide")

with st.sidebar:
    st.image("images//pwc.png")  
    # st.text("Proposal Eligibility Analyzer")
    # open_ai_api = st.text_input('Open AI API', 'Enter your open AI API key')
    # if not open_ai_api.startswith('sk-'):
    #             st.error("Please enter a valid OpenAI API key.")
    #             api_keys_entered = False

# os.environ['OPENAI_API_KEY'] = open_ai_api

def main():

    st.title("Proposal Analyzer")

    c1,c2,c3= st.columns(3)

    with c1:
        upload_rfp = st.file_uploader("Upload a RFP (PDF)")
    with c2:
        upload_p = st.file_uploader("Upload a Proposal (PDF)")
    with c3:
        upsert = st.button("Upsert Vector DB", type="primary")
        run    = st.button("Run", type="primary")
    
    if upsert:

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(upload_rfp.getvalue())
            tmp_file_path_1 = tmp_file.name

        loader = PyMuPDFLoader(tmp_file_path_1)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)

        # create the open-source embedding function
        embedding_function = AzureOpenAIEmbeddings(
            openai_api_type = "azure",
            openai_api_key = "cd7bc404225847f984753fcf60d1f3cf",
            azure_endpoint = "https://aai-aiml-apps.openai.azure.com/",
            deployment = "text-embedding-ada-002",
            model = "text-embedding-ada-002"
        )

        print(embedding_function)

        db = FAISS.from_documents(docs, embedding_function)
        db.save_local("vectorstore/RFP/")

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(upload_p.getvalue())
            tmp_file_path_2 = tmp_file.name

        loader = PyMuPDFLoader(tmp_file_path_2)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)

        # create the open-source embedding function
        embedding_function = AzureOpenAIEmbeddings(
            openai_api_type = "azure",
            openai_api_key = "cd7bc404225847f984753fcf60d1f3cf",
            azure_endpoint = "https://aai-aiml-apps.openai.azure.com/",
            deployment = "text-embedding-ada-002",
            model = "text-embedding-ada-002"
        )
        db = FAISS.from_documents(docs, embedding_function)
        db.save_local("vectorstore/Proposals/")
    
    if run:
        embedding_function = AzureOpenAIEmbeddings(
            openai_api_type = "azure",
            openai_api_key = "cd7bc404225847f984753fcf60d1f3cf",
            azure_endpoint = "https://aai-aiml-apps.openai.azure.com/",
            deployment = "text-embedding-ada-002",
            model = "text-embedding-ada-002"
        )

        template ="""
        Your objective is to analyze wheather a proposal meets the PRE-QUALIFICATION/ ELIGIBILITY CRITERIA mentioned with a RFP
    

        Please refer to the following RFP's PRE-QUALIFICATION/ ELIGIBILITY CRITERIA statement: 

        {RFP}

        Please refer to the following proposal submited by the vendor:

        {Proposal}

        Please let us know for every eligibility criteria mentioned how does the proposal meets this criteria. If a proposal does not meet please let us know that as well 
        A.
        """

        prompt = PromptTemplate(input_variables=["RFP","Proposal"],template = template)

        DB_FAISS_PATH_1 = "vectorstore\\RFP\\"
        rfp_db = FAISS.load_local(DB_FAISS_PATH_1, embedding_function,allow_dangerous_deserialization=True)
        query_1 = "Get me the points of the PRE-QUALIFICATION/ ELIGIBILITY CRITERIA in the documents"
        docs_1 = rfp_db.similarity_search(query_1 , k = 4)
        context1_rfp = docs_1[0].page_content + docs_1[1].page_content + docs_1[2].page_content + docs_1[3].page_content

        print(query_1, docs_1, context1_rfp)

        DB_FAISS_PATH_2 = "vectorstore\\Proposals\\"
        p_db = FAISS.load_local(DB_FAISS_PATH_2, embedding_function,allow_dangerous_deserialization = True)
        query_2 = "Information about organization history and background. Along with any eligibility criteria mentiond by them"
        docs_2 = p_db.similarity_search(query_2 , k = 4)
        context2_proposals = docs_2[0].page_content + docs_2[1].page_content + docs_2[2].page_content + docs_2[3].page_content

        os.environ.pop('OPENAI_API_BASE', None)

        llm= AzureChatOpenAI(
                openai_api_key = "cd7bc404225847f984753fcf60d1f3cf",
                openai_api_version = "2023-03-15-preview",
                # model = "gpt-35-turbo",
                azure_deployment = "gpt-35-turbo",
                azure_endpoint = "https://aai-aiml-apps.openai.azure.com/",
                temperature = 0,
                # model = "gpt-3.5-turbo"
            )

        print(llm)
        Objective = LLMChain(
            llm= llm,
            prompt=prompt,
            output_key="solutions")
        
        st.header("Eligiblity:")
        st.write(Objective.predict(RFP = context1_rfp, Proposal = context2_proposals))

        template ="""
        Your objective is to analyze the objective portion of an RFP with a proposal and give your inputs to the company wheather the vendor has met the criteria of the RFP.
    

        Please refer to the following RFP's objective statement: 

        {RFP}

        Please refer to the following proposal submited by the vendor:

        {Proposal}

        You need to analyze from the respective portion wheather the vendor is meeting the ojective crietera and if so how are they meeting if not what are they lacking, give answer in great detail and solve the problem step by step

        A.
        """

        prompt = PromptTemplate(input_variables=["RFP","Proposal"],template = template)

        DB_FAISS_PATH_1 = "vectorstore\\RFP\\"
        rfp_db = FAISS.load_local(DB_FAISS_PATH_1, embedding_function,allow_dangerous_deserialization=True)
        query_1 = "What are the main points of ojective of the RFP that the vendor needs to fullfill"
        docs_1 = rfp_db.similarity_search(query_1 , k = 4)
        context1_rfp = docs_1[0].page_content + docs_1[1].page_content + docs_1[2].page_content + docs_1[3].page_content

        print(query_1, docs_1, context1_rfp)

        DB_FAISS_PATH_2 = "vectorstore\\Proposals\\"
        p_db = FAISS.load_local(DB_FAISS_PATH_2, embedding_function,allow_dangerous_deserialization = True)
        query_2 = "Get me a brief summary of the objective of the vendor in detail"
        docs_2 = p_db.similarity_search(query_2 , k = 4)
        context2_proposals = docs_2[0].page_content + docs_2[1].page_content + docs_2[2].page_content + docs_2[3].page_content

        Objective = LLMChain(
            llm=llm,
            prompt=prompt,
            output_key="solutions")
        
        st.header("Objective:")
        st.write(Objective.predict(RFP = context1_rfp, Proposal = context2_proposals))

        template ="""
        Your objective is to analyze the methdology portion of an RFP with a proposal and give your inputs to the company wheather the vendor has met the criteria of the RFP.
            

        Please refer to the following RFP's methodology statement: 

        {RFP}

        Please refer to the following proposal's methodology submited by the vendor:

        {Proposal}

        You need to analyze from the respective portion wheather the vendor is meeting the methodology crietera and if so how are they meeting if not what are they lacking, give answer in great detail and solve the problem step by step

        A.
        """

        prompt = PromptTemplate(
            input_variables=["RFP","Proposal"],
            template = template                      
        )

        DB_FAISS_PATH_1 = "vectorstore\\RFP\\"
        rfp_db = FAISS.load_local(DB_FAISS_PATH_1, embedding_function,allow_dangerous_deserialization=True)
        query_1 = "What are the main points of methodology parts mentioned in the RFP"
        docs_1 = rfp_db.similarity_search(query_1 , k = 4)
        context1_rfp = docs_1[0].page_content + docs_1[1].page_content + docs_1[2].page_content + docs_1[3].page_content

        DB_FAISS_PATH_2 = "vectorstore\\Proposals\\"
        p_db = FAISS.load_local(DB_FAISS_PATH_2, embedding_function,allow_dangerous_deserialization = True)
        query_2 = "What are the main points of methodology that the vendor has mentioned"
        docs_2 = p_db.similarity_search(query_2 , k = 4)
        context2_proposals = docs_2[0].page_content + docs_2[1].page_content + docs_2[2].page_content + docs_2[3].page_content


        Objective = LLMChain(
            llm=llm,
            prompt=prompt,
            output_key="solutions"
        )

        st.header("Methodology:")
        st.write(Objective.predict(RFP = context1_rfp, Proposal = context2_proposals))

if __name__=="__main__":
    main()
