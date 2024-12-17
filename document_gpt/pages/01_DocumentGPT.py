import streamlit as st
import time

st.title("Document GPT")


def send_message(message,role, save=True):
    with st.chat_message(role):
        st.write(message)
    if save:
        st.session_state["messages"].append({"message": message, "role" : role})


if "messages" not in st.session_state:
    st.session_state["messages"] = []
else:
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)

message = st.chat_input("Send a message to the ai")

if message:
    send_message(message, "human", save=True)
    time.sleep(2)
    send_message(f"You said : {message}", "ai")

    with st.sidebar:
        st.write(st.session_state["messages"])