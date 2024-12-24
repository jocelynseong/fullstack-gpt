import streamlit as st
import ai, time

def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

def save_message(message, role , question=False):
    st.session_state["messages"].append({"message": message, "role": role})
    if question:
        ai.save_memory(message, response)

st.set_page_config(
    page_title="Site GPT",
    page_icon="❓"
)

st.write("""
    Ask Question about the content of a website.\n
    url: developers.cloudflare.com\n
    category : ai gateway, cloudflare vector, workers ai
""")

with st.sidebar:
    api_key = st.text_input("openai api key")


def stream_response(response):
    message_box = st.empty()
    displayed_message = ""
    for char in response:
        displayed_message += char
        message_box.markdown(displayed_message)
        time.sleep(0.005)  

if api_key:
    ai.set_api_key(api_key)
    send_message("i'm ready! ask away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about you file")
    if message:
        send_message(message, "human")
        with st.chat_message("ai"):
            response = ai.get_answer(message)
            if response == "AuthError":
                stream_response("open ai key is not correct")
            elif response == "Error":
                stream_response("Something is wrong plz refresh and try again.")
            else:
                stream_response(response)
                save_message(response, "ai", question=message)


else:
    st.session_state['messages'] = [] 

