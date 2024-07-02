import streamlit as st
import os
import openai
from dotenv import load_dotenv
from langchain.document_loaders import PDFPlumberLoader
from langchain.docstore import InMemoryDocstore
from langchain.text_splitter import CharacterTextSplitter , RecursiveCharacterTextSplitter, SpacyTextSplitter, NLTKTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
##Facebook AI Similarity Search (Faiss)  is a library for efficient similarity search and clustering of dense vectors. It contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM. It also contains supporting code for evaluation and parameter tuning.
from langchain.vectorstores import FAISS
#the bellow required for keeping the conversation chain in chatgpt
from langchain.memory import ConversationBufferMemory, VectorStoreRetrieverMemory
#The bellow allow us to chat with text(chat with vectorstore)
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import faiss
from langchain.vectorstores import Chroma
from langchain.chains import LLMChain,ConversationChain

openai.api_type = "azure"
openai.api_version = "2023-05-15" 
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")  # Your Azure OpenAI resource's endpoint value.
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")

os.environ["OPENAI_API_TYPE"] = 'azure'
os.environ["OPENAI_API_BASE"] = str(os.getenv("AZURE_OPENAI_ENDPOINT"))
os.environ["OPENAI_API_KEY"] = str(os.getenv("AZURE_OPENAI_API_KEY"))
os.environ["OPENAI_API_VERSION"] = str(os.getenv("OPEN_AI_API_VERSION"))

def upload_pdf_to_destination(uploaded_file,storage_dir,file_name):
    ##Suggesting to use cloud for storing a copy of files
    # storage_dir = './file_storage_'
    os.makedirs(storage_dir, exist_ok=True)
    output_path = os.path.join(storage_dir, file_name)
    with open(output_path, "wb") as output_file:
        output_file.write(uploaded_file.read())
    return output_path


def text_splitter(splitter:str,raw_documents):
    text_splitters: dict ={'RecursiveCharacter':RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap  = 0,
            length_function = len,
            add_start_index = True,
            ).split_documents(raw_documents),              
            'NLTK' : NLTKTextSplitter(chunk_size = 1000).split_documents(raw_documents),
            'Character':CharacterTextSplitter(        
            separator = "\n\n",
            chunk_size = 1000,
            chunk_overlap  = 0,
            length_function = len,
            is_separator_regex = False,
            ).split_documents(raw_documents)
    }
    documents = text_splitters.get(splitter)
    return documents
    
def text_embedding(embedding): 
    ## Creating a temproraly storage for storing the vectors
    ## Embeddings are a numerical representation of text that can be used to measure the relatedness between two pieces of text. 
    ## Huggingface model name is comming from "https://huggingface.co/spaces/mteb/leaderboard"
    ## Keep in mind to check `https://openai.com/pricing` for embeding models proce if approved to purchase: it is very cheap
    embedding_options: dict={
        'HuggingfaceInstruct':HuggingFaceInstructEmbeddings(model_name = "thenlper/gte-large"),
        'OpenAI':OpenAIEmbeddings(model="text-embedding-ada-002", chunk_size = 1, max_retries=10, show_progress_bar=True)
    }
    embeddings = embedding_options.get(embedding)
    return embeddings 

def vector_db(documents, embeddings , store):
    store_options:dict={
        'FAISS':FAISS.from_documents(documents, embedding=embeddings),
        # 'Chroma':Chroma.from_documents(documents,embedding=embeddings)##Chroma gives me back error needs work on that if is required
    }
    db = store_options.get(store)
    return db

def llm_model(gpt):
    models: dict={
        'gpt_3':AzureChatOpenAI(openai_api_version="2023-03-15-preview",
                                deployment_name="gpt-35-turbo",
                                model_name="gpt-35-turbo",
                                # max_tokens = 200,
                                request_timeout = 20,
                                temperature = 0.1
                                ),
        'gpt_4':AzureChatOpenAI(openai_api_version="2023-03-15-preview",
                                deployment_name="central-ai-gpt4",
                                model_name="gpt-4",
                                # max_tokens = 200,
                                request_timeout = 20,
                                temperature = 0.1)
    }
    model = models.get(gpt)
    return model

def get_conversation_chain(vectordb, conversation_chain = "ConversationalRetrievalChain"):
    ## Notice that we `return_messages=True` to fit into the MessagesPlaceholder
    ## Notice that `"chat_history"` aligns with the MessagesPlaceholder name.
    llm = llm_model('gpt_4')
    if conversation_chain == 'ConversationalRetrievalChain':
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
    elif conversation_chain == 'LLM':
        retriever = vectordb.as_retriever(search_kwargs=dict(k=10))
        memory = VectorStoreRetrieverMemory(retriever=retriever,memory_key="chat_history",return_messages=True)  
        prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                "You are a nice chatbot having a conversation with a human about the submitted Construction Submittal."
            ),
            # The `variable_name` here is what must align with memory
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{question}")
        ]
    )      
        conversation = LLMChain(
            llm=llm,
            prompt=prompt,
            verbose=True,            
            memory=memory
        )

    elif conversation_chain == 'Conversational':        
        retriever = vectordb.as_retriever(search_kwargs=dict(k=10))
        memory = VectorStoreRetrieverMemory(retriever=retriever)  
        _DEFAULT_TEMPLATE = """The following is a technical conversation between a human and an AI. The AI is talkative and provides lots of specific details from its construction submittal context.

                            Relevant pieces of previous conversation:
                            {chat_history}
                            Current conversation:
                            Human: {question}
                            AI:"""
        PROMPT = PromptTemplate(
                                input_variables=["chat_history", "question"], template=_DEFAULT_TEMPLATE
                            )
        conversation = ConversationChain(llm=llm, 
                                           prompt=PROMPT,
                                           # We set a very low max_token_limit for the purposes of testing.
                                           memory=memory,
                                           verbose=True
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
                embeddings = text_embedding('OpenAI')                
                vectorstore = vector_db(documents,embeddings,'FAISS')

                st.success("File processed successfully!")

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore,"ConversationalRetrievalChain")


if __name__ == '__main__':
    main()


