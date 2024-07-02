import streamlit as st
import os
import openai
from dotenv import load_dotenv
from htmlTemplates import css, bot_template, user_template
from langchain.document_loaders import PDFPlumberLoader
from langchain.text_splitter import  RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate

openai.api_type = "azure"
openai.api_version = "2023-05-15" 
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")  
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")

os.environ["OPENAI_API_TYPE"] = 'azure'
os.environ["OPENAI_API_BASE"] = str(os.getenv("AZURE_OPENAI_ENDPOINT"))
os.environ["OPENAI_API_KEY"] = str(os.getenv("AZURE_OPENAI_API_KEY"))
os.environ["OPENAI_API_VERSION"] = str(os.getenv("OPEN_AI_API_VERSION"))

def upload_pdf_to_destination(uploaded_file,storage_dir,file_name):
    os.makedirs(storage_dir, exist_ok=True)
    output_path = os.path.join(storage_dir, file_name)
    with open(output_path, "wb") as output_file:
        output_file.write(uploaded_file.read())
    return output_path

def text_splitter(splitter:str,raw_documents):
    text_splitters: dict ={'RecursiveCharacter':RecursiveCharacterTextSplitter(
            chunk_size = 1000, chunk_overlap  = 0, length_function = len, add_start_index = True).split_documents(raw_documents)}
    documents = text_splitters.get(splitter)
    return documents
    
def text_embedding(max_retries): 
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", chunk_size = 1, max_retries=max_retries, show_progress_bar=True)
    return embeddings 

def vector_db(documents, embeddings):
    db=FAISS.from_documents(documents, embedding=embeddings)
    return db

def llm_model():    
    model = AzureChatOpenAI(openai_api_version="2023-03-15-preview",
                                deployment_name="central-ai-gpt4",
                                model_name="gpt-4",
                                # max_tokens = 200,
                                request_timeout = 20,
                                temperature = 0.1
                                )
    return model

def get_conversation_chain(vectordb):
    llm = llm_model()
    custom_template = """Given the following conversation and a follow up question about provided "construction submittals", rephrase the follow up question to be a standalone question.
                        Chat History:
                        {chat_history}
                        Follow Up Input: {question}
                        Standalone question:"""
    CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)
    memory = ConversationBufferMemory(memory_key="chat_history",
                                       return_messages=True)
    conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        condense_question_prompt=CUSTOM_QUESTION_PROMPT,
        retriever=vectordb.as_retriever(search_kwargs=dict(k=10)),
        memory = memory
    )
    return conversation

## Streamlit adjustment
def handle_userinput(user_question):
    ###The bellow line help to remeber all conversation
    response = st.session_state.conversation({'question':user_question})
    st.session_state.chat_history = response['chat_history']    
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    ##Streamlit adjustment
    st.set_page_config(page_title='Chat with your Submittal',
                        page_icon=":building_construction:"
                        )
    st.write(css,unsafe_allow_html = True)
    ##initializing the conversation with None if conversation is not initialized
    if "conversation" not in  st.session_state:
        st.session_state.conversation = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with your Submittal :books:"
              )
    user_question = st.text_input("Ask your question about uploaded submittal here:",
                  )   
    if user_question:
        handle_userinput(user_question)
 
    with st.sidebar:
        st.subheader('Submittal')
        pdf_docs = st.file_uploader(
            'Upload your submittal PDF here and click on process',
            type = ["pdf"] ,
            accept_multiple_files=False
            )
        if pdf_docs is not None:
            path = upload_pdf_to_destination(pdf_docs,'./file_storage_',pdf_docs.name)
            # st.success(f"File uploaded successfully!")
       
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_documents = PDFPlumberLoader(path).load()
                
                # get the text chunks
                documents = text_splitter('RecursiveCharacter' , raw_documents)
                
                # create vector store
                embeddings = text_embedding(10)                
                vectorstore = vector_db(documents,embeddings)

                st.success("File processed successfully!")

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == '__main__':
    main()