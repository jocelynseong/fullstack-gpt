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
from utils import *

st.set_page_config(
    page_title="FullstackGPT Home",
    page_icon="😀"
)

with st.sidebar:
    github_url = "https://github.com/jocelynseong/fullstack-gpt/"
    st.sidebar.write(f"[View on GitHub]({github_url})")
    ## api key
    open_api_key = st.text_input("openai api key")
    llm=None
    if open_api_key:
        llm = get_llm(open_api_key)
    ## topic
    topic = st.text_input("주제를 입력해주세요")
    level = st.selectbox(
        "시험 난이도를 선택하세요",
        (
            "high","medium","low"
        )
    )

if llm and topic and level:
    docs = wiki_search(topic)
    res = run_quiz_chain(docs, topic, level, llm, open_api_key)
    if res is None:
        st.markdown(
        """
        퀴즈를 생성해서 실력을 점검해볼 수 있습니다 :) \n
        만점이 아닌 경우 시험을 다시 치를 수 있습니다. \n
        자체 Open API 키를 사용해주세요 ~!
        """
        )
    else:
        with st.form("q_form"):
            user_cnt = 0
            for index, q in enumerate(res["questions"]):
                value = st.radio(f"Q{index+1}] {q['question']}",
                                [answer['answer'] for answer in q['answers']],
                                index=None)
                if {'answer' : value , 'correct' : True} in q["answers"]:
                    st.success("정답입니다.")
                    user_cnt+=1
                elif value is not None:
                    correct_answers = [item["answer"] for item in q["answers"] if item["correct"]]
                    st.error(f"틀렸습니다. 정답은 {correct_answers}입니다")
            button = st.form_submit_button()
            if user_cnt == 10:
                st.balloons()

else:
    st.markdown(
    """
    퀴즈를 생성해서 실력을 점검해볼 수 있습니다 :) \n
    만점이 아닌 경우 시험을 다시 치를 수 있습니다. \n
    자체 Open API 키를 사용해주세요 ~!
    """
    )