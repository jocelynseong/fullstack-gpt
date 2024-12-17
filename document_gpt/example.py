import streamlit as st
from langchain.prompts import PromptTemplate
from datetime import datetime
import time

today = datetime.today().strftime("%H:%M:%S")
 
st.title(today)

# st.title("Hello world");
# st.subheader("welcome to streamlit!")
# st.markdown("""
    ### I love it!
# """)

# st.write("jj")

# [0,1,2,3]

# PromptTemplate

# p = PromptTemplate.from_template("xxxx")
# st.write(p)

model = st.selectbox("choose your model", ["gpt-3", "gpt-4"])

st.write(model)

if model == "gpt-3":
    st.write("cheap")
else:
    st.write("not cheap")

name = st.text_input("what is you name?")

st.write(name)

value = st.slider("tempreature", min_value=0.1, max_value=1.0)

value

with st.sidebar:
    st.title("sidebar title")
    st.text_input("xxx")

tab_1, tab_2, tab_3 = st.tabs(["A", "B", "C"])

with tab_1:
    st.write("1")
with tab_2:
    st.write("2")
with tab_3:
    st.write("3")

with st.status("Enbedding file ...", expanded=True) as status:
    time.sleep(3)
    st.write("Getting the file")
    time.sleep(3)
    st.write("Embedding the file")
    time.sleep(3)
    st.write("Caching the file")
    status.update(label="Error", state="error")
