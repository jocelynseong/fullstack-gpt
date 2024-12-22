import streamlit as st
import json
from langchain.retrievers import WikipediaRetriever
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseOutputParser
from langchain.callbacks import StreamingStdOutCallbackHandler

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

questions_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a helpful assistant that is role playing as a teacher.
         
    Based ONLY on the following context make 10 (TEN) questions minimum to test the user's knowledge about the text.
    
    Each question should have 4 answers, three of them must be incorrect and one should be correct.
    
    Provide 10 questions in Korean with a difficulty level of {level}, categorized among high, medium, and low difficulty levels.

    Your turn!
         
    Context: {context}
""",
        )
    ]
)


@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5)
    return retriever.get_relevant_documents(term)


class ContextFormatter(BaseOutputParser):
    def parse(self, docs, level):
        return {"context": format_docs(docs), "level" : level}

@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, topic, level, _llm, api_key):
    try: 
        formatter = ContextFormatter()
        parsed_input = formatter.parse(_docs, level)
        questions_chain =  questions_prompt | _llm
        response = questions_chain.invoke(parsed_input)
        response = json.loads(response.additional_kwargs["function_call"]["arguments"])
        return response
    except Exception as e:
        print(e)
        return None


question_schema = {
        "name": "create_quiz",
        "description": "function that takes a list of questions and answers and returns a quiz",
        "parameters": {
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                            },
                            "answers": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "answer": {
                                            "type": "string",
                                        },
                                        "correct": {
                                            "type": "boolean",
                                        },
                                    },
                                    "required": ["answer", "correct"],
                                },
                            },
                        },
                        "required": ["question", "answers"],
                    },
                }
            },
            "required": ["questions"],
        },
    }

def get_llm(open_api_key):
    llm = ChatOpenAI(
        temperature=0,
        streaming=True,
        callbacks=[
            StreamingStdOutCallbackHandler()
        ],
        api_key=open_api_key
    ).bind(
        function_call={
            "name" : "create_quiz"
        },
        functions =[
            question_schema
        ]
    )
    return llm
 