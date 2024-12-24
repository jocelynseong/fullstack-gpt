import streamlit as st
from langchain.document_loaders import AsyncChromiumLoader, SitemapLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

llm = ChatOpenAI(
    temperature=0.1
)

st.title("Site GPT!")

html2text_transformer = Html2TextTransformer()

answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  
    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!

    Question: {question}
"""
)

def get_answers(inputs):
    docs = inputs['docs']
    question = inputs['question']
    answers_chain = answers_prompt | llm
    # answers = []
    # for doc in docs:
    #     result = answers_chain.invoke({
    #         "question" : question,
    #         "context" : doc.page_content
    #     })
    #     answers.append(result.content)

    return {
        "question" : question, 
        "answers" : [
            {
            "answer" : answers_chain.invoke({
                "question" : question,
                "context" : doc.page_content
            }).content,
            "source" : doc.metadata["source"]
            }for doc in docs
        ]
    }

choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite sources and return the sources of the answers as they are, do not change them.

            Answers: {answers}
            """,
        ),
        ("human", "{question}")
    ]
)

def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\n"
        for answer in answers
        )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
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
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=800,
        chunk_overlap=200
    )
    loader = SitemapLoader(url, 
                           filter_urls=[
                               r"^(.*\/careers\/)."
                           ],
                           parsing_function=parse_page
                           )
    loader.requests_per_second = 5
    docs = loader.load_and_split(text_splitter=splitter)
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vector_store.as_retriever()


st.write("""
    Ask Question about the content of a website.\n
    Start by writeing the URㅣL of the website on the sidebar
""")

# '''
# playwright,chromium 
# - 대형 웹사이트에 sitemap이 없거나, javascript가 load되는 것을 기다려야 하는 경우에 사용됨
# - browser control을 할 수 있는 패키지
# - (~= selenium 마찬가지로 browser control)
# -- 1) AsyncChromiumLoader

# sitemap
# - 일부 웹파이트들은 sitemap을 갖고 있다. /stiemap.xml
# - open api들이 다 각각 만들어놓은 것. 모든 url의 directory를 볼 수 있음
# - openai.com/sitemap.xml
# - 웹사이트에서 검색 엔진에 웹 페이지 구조를 알려주는데 사용되는 파일 중요한 페이지를 나열하며, 검색 엔진이 효율적으로 크롤링하고 색인화할 수 있도록 도움
# - 목적
# -- 검색엔진 최적화 (SEO)
# -- 다국어 페이지 지원 
# --- hreflang 테그를 사용해 언어별/지역별 페이지를 명확히 구분할 수 있습니다.
# '''

with st.sidebar:
    url = st.text_input("Write down a URL", placeholder="https:///example.com")

if url:
    # 1) async chromium loader
    # loader = AsyncChromiumLoader([url])
    # docs = loader.load()
    # transformed = html2text_transformer.transform_documents(docs)
    # st.write(docs)
    # 2 ) 
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a Sitemap URL")
    else:
        retriver = load_website(url)
        query = st.text_input("Ask a question to the website")
        if query:
            ## map re ReRank
            chain = {
                "docs" : retriver, 
                "question" : RunnablePassthrough()
                } | RunnableLambda(get_answers) | RunnableLambda(choose_answer)

            response = chain.invoke(query)
            response.content