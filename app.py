import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationSummaryBufferMemory
from langchain.callbacks import StreamingStdOutCallbackHandler


st.set_page_config(
    page_title="FullstackGPT Home",
    page_icon="😀"
)

class ChatCallbackHandler(BaseCallbackHandler):
    message = ""
    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()
    def on_llm_end(self,*args, **kwargs):
        save_message(self.message, "ai")
    def on_llm_new_token(self, token, *args,  **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

@st.cache_data(show_spinner="Embedding...")
def embed_file(this):
    file_content = file.read()
    file_path = f"./db/files/{file.name}"

    with open(file_path, "wb") as f:
        f.write(file_content)

    cache_dir = LocalFileStore(f'./db/embeddings/{file.name}')
    spliter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100
    )
    loader = UnstructuredFileLoader(f"./db/files/{file.name}")
    docs = loader.load_and_split(text_splitter=spliter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings,
        cache_dir
    )
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
            """
            Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.
            
            Context: {context}
            
            History : {history}
            """,        
    ),
    ("human", "{question}")
])


with st.sidebar:
    github_url = "https://github.com/jocelynseong/fullstack-gpt/app.py"
    st.sidebar.write(f"[View on GitHub]({github_url})")
    openai_api_key = st.text_input("openai api key")
    file = st.file_uploader(
        "upload file (.txt .pdf .docx)", 
        type=["pdf", "txt", "docx"]
    )


st.markdown(
    """
    Welcome!

    Use this chatbot to ask questions to an ai about your files!
                
    Set your api key and Upload your files on the sidebar.
    """)



if file and openai_api_key:
    llm = ChatOpenAI(
        temperature=0.1,
        streaming=True,
        callbacks=[
            ChatCallbackHandler(),
            StreamingStdOutCallbackHandler()
        ],
        openai_api_key=openai_api_key
    )
    if llm:

        memory = ConversationSummaryBufferMemory(
            llm=llm,
            max_token_limit=120,
            return_messages=True
        )
        def load_memory(input):
            history = memory.load_memory_variables({}).get("history", [])
            print(f"Loaded history: {history}")
            return history
        retriever = embed_file(file)
        send_message("i'm ready! ask away!", "ai", save=False)
        paint_history()
        message = st.chat_input("Ask anything about your file")
        if message:
            send_message(message, "human")
            chain = {
                "context" : retriever | RunnableLambda(format_docs),
                "history" : RunnableLambda(load_memory),
                "question" : RunnablePassthrough()
            } |  prompt | llm

            with st.chat_message("ai"):
                response = chain.invoke(message)
                memory.save_context({"input": message}, {"output": response.content})
    else:
        st.session_state["messages"] = []
else:
    st.session_state["messages"] = [] 