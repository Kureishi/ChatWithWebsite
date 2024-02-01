import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms.gpt4all import GPT4All
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import torch
import os
from dotenv import load_dotenv


# Use CPU to avoid GPU out-of-memory errors (since Embeddings use PyTorch)
torch.cuda.is_available = lambda : False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load environment variable from .env file
load_dotenv()

# define path to LLM
PATH = r"C:\Users\Kureishi Shivanand\AppData\Local\nomic.ai\GPT4All\gpt4all-falcon-newbpe-q4_0.gguf"




# return vectorstore from webpage
def get_vectorstore_from_url(url):

    # get text in document form
    loader = WebBaseLoader(url)
    document = loader.load()

    # split document into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)

    # set embeddings model
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")

    # create vectorstore from chunks
    vectorstore = Chroma.from_documents(document_chunks, embeddings)
    
    return vectorstore



# context retriever chain (retrieve documents relevant to entire conversation)
def get_context_retriever_chain(vector_store):

    # define LLM to use
    llm = GPT4All(model=PATH, backend="gptj", verbose=False)

    # allow to retrieve vectors
    retriever = vector_store.as_retriever()

    # initialize prompt to take array of messages (includes past messages as history)
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),      # populate only if history exists
        ('user', "{input}"),                                    # 'input' value populated with what passed to chain
        ('user', "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")   # find chunks of text (documents) relevant to entire conversation
    ])

    # combine retriever chain components together
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain



# use retrieved documents along with history to generate response to query
def get_conversational_rag_chain(retriever_chain):

    # define LLM to use
    llm = GPT4All(model=PATH, backend="gptj", verbose=False)

    # define prompt (value of placeholder variables (context, input) will be provided later using LCEL)
    prompt = ChatPromptTemplate.from_messages([
        ('system', "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name='chat_history'),                                  # provide history if available
        ('user', "{input}")
    ])

    # get response using documents and context
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    # return retrieval chain using 'retrieval_chain' with relevant document (runnable as opposed to staple vectorstore) and stuff_documents (containing context)
    return create_retrieval_chain(retriever=retriever_chain, combine_docs_chain=stuff_documents_chain)




# simulate logic behind ChatBot
def get_response(user_input):
    # get retriever_chain from vector_store
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)

    # get the conversational RAG chain using relevant documents and context
    conv_rag_chain = get_conversational_rag_chain(retriever_chain)

    # get response using relative documents and context (invoke chain with actual values)
    response = conv_rag_chain.invoke({
        'chat_history': st.session_state.chat_history,
        'input': user_query
    })

    return response['answer']   # return just 'answer' component of response (not along with 'context' and 'chat_history')





# streamlit format
st.set_page_config(page_title="Chat with Websites", page_icon="🤖")
st.title("Chat with Website")

with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")

# handles if no URL is present
if website_url is None or website_url == "":
    st.info("Please enter website URL")

else:

    # put chat_history in session_state so not re-initialize everytime event happens (so app not overwrite history)
    # once chat_history in session_state -> not re-initialize it
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a bot. How may I help you?")
        ]

    # add vector_store to session_state to not re-embedding documents everytime make a query
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(website_url)       # get vectorstore from URL

    # create text box for user to input into chat
    user_query = st.chat_input("Type your message here...")

    # check if user typed something
    if user_query is not None and user_query != "":

        # simulate logic
        response = get_response(user_query)

        # append messages to chat history
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))     

    # conversation
    for message in st.session_state.chat_history:

        # check if an AIMessage
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)

        # check if an AIMessage
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)