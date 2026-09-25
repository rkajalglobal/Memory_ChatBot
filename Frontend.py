import streamlit as st
import Backend as demo


user_icon = """
            <svg xmlns="http://w3.org" width="200" height="200" viewBox="0 0 200 200">
            <image width="200" height="200" 
       style="filter: invert(48%) sepia(79%) saturate(2476%) hue-rotate(202deg) brightness(118%) contrast(119%);" 
       href="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..." />
            </svg>
        """

st.set_page_config(
    page_title="Ruchi OpenAI Chat Bot",
    page_icon= user_icon,
    layout = "centered",
)

st.title(" \U0001F680 Ruchi Memory Chat Bot")
st.caption("Powered by OpenAI and Langchain Magic")

with st.sidebar:
    st.header("Controls")
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.memory = demo.demo_memory()
        st.session_state.chat_history = []
        st.rerun()

        st.divider()
        st.markdown(
            "*****About this Bot***********/n/n"
            "Ruchi Assistant"
        )

if "memory" not in st.session_state:
    st.session_state.memory = demo.demo_memory()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["text"])

input_text = st.chat_input("Ask if you want to ask.......")
if input_text:
    with st.chat_message("user"):
        st.markdown(input_text)    
    st.session_state.chat_history.append({"role": "user", "text": input_text})

    with st.chat_message("Ruchi"):
        with st.spinner("Stop interrupting I am thinking........" ):
            result, updated_memory= demo.demo_conversation(
                input_text=input_text,
                memory=st.session_state.memory,
            )
        st.markdown(result)

    st.session_state.memory = updated_memory
    st.session_state.chat_history.append({"role": "Ruchi", "text": result})
