import hashlib
from langchain.document_loaders import AsyncChromiumLoader, SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_transformers import Html2TextTransformer
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
import streamlit as st


def generate_hash(url: str, algorithm: str = "sha256") -> str:
    # Encode URL to bytes
    url_bytes = url.encode('utf-8')

    # Select hash algorithm
    if algorithm == "md5":
        hash_object = hashlib.md5(url_bytes)
    elif algorithm == "sha1":
        hash_object = hashlib.sha1(url_bytes)
    elif algorithm == "sha256":
        hash_object = hashlib.sha256(url_bytes)
    else:
        raise ValueError("Unsupported algorithm. Use 'md5', 'sha1', or 'sha256'.")

    # Return hex digest
    return hash_object.hexdigest()

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=800,
    chunk_overlap=200
)

def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return str(soup.get_text()).replace("\n", "").replace("\xa0", "").replace("\t", "")


@st.cache_data(show_spinner="Loading website ...")
def load_website():
    try :
        url = 'https://developers.cloudflare.com/sitemap-0.xml'
        loader = SitemapLoader(url, 
                        filter_urls=[
                           r"^(.*\/(?:ai-gateway|vectorize|workers-ai)\/).*"
                        ],
                        parsing_function=parse_page
                        )
        loader.requests_per_second = 5
        docs = loader.load_and_split(text_splitter=splitter)
        cache_dir = LocalFileStore(f'../../../db/embeddings/{generate_hash(url, "md5")}')
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
            OpenAIEmbeddings(),
            cache_dir
        )
        vector_store = FAISS.from_documents(docs, cached_embeddings)
        retriever = vector_store.as_retriever()
        return retriever
    except Exception as e:
        print(e)
        return None