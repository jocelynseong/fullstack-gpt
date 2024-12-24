from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage
from openai.error import AuthenticationError

import streamlit as st
import loader

llm = ChatOpenAI(
    temperature=0.1,
    openai_api_key=None
)

memory = ConversationBufferMemory(
    llm=llm,
    return_messages=True
)

def add_messages(input, output):
    memory.save_context({"input" : input}, {"output" : output})

def get_history():
    return memory.load_memory_variables({})

def load_memory():
    ## 메모리 로드
    return memory.load_memory_variables({})['history']

def save_memory(question, answer):
    existing_data = memory.load_memory_variables({"input": question})
    if "output" in existing_data and existing_data["output"] == answer:
        print("이미 동일한 입력과 출력이 메모리에 저장되어 있습니다.")
        return False  # 저장하지 않음
    else:
        memory.save_context({"input" : question}, {"output":answer})

choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent and long ones.

            Cite sources and return the sources of the answers as they are, do not change them.

            Answers: {answers}
            """
        ),
        ("human", "{question}")
    ]
)

answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  
    Then, give a score to the answer between 0 and 100.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 100
                                                  
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

def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\n\nSource:{answer['source']}\n"
        for answer in answers
        )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )

def set_api_key(api_key):
    llm.openai_api_key=api_key


def find_similar_question_answer(question):
    history = memory.load_memory_variables({})['history']
    print('history-->', history)
    if len(history) == 0:
        return None
    user_questions = '\n'.join([
        message.content for message in history if isinstance(message, HumanMessage)
    ])
    prompt = f"""
        Choose only one of Memory questions which has same meaning compare to Current question.

        Current question: {question}
        Memory questions:
        {user_questions}

        if there is no same question answer "XXX!".
        if exists. answer the Memory question exactly.
    """
    response = llm.predict(prompt)
    if "XXX!" in response:
        return None
    else:
        cached_answer =  memory.load_memory_variables({"input" : response})
        recent_ai_message = next(
            (message.content for message in reversed(cached_answer['history']) if isinstance(message, AIMessage)),
            None)
        return recent_ai_message


@st.cache_data(show_spinner="searching...")
def get_answer(question):
    try : 
        cached_answer = find_similar_question_answer(question)
        if cached_answer:
            return cached_answer
        else:
            chain = {
                "docs" : loader.load_website(), 
                "question" : RunnablePassthrough()
            } | RunnableLambda(get_answers) | RunnableLambda(choose_answer)
            response = chain.invoke(question)
            return response.content
    except AuthenticationError as ae:
        print(ae)
        return "AuthError"
    except Exception as e:
        print(ae)
        return "Error"
