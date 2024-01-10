import streamlit as st
from openai import OpenAI
import os

openai_api_key = os.environ["API_KEY"]
client = OpenAI(
    api_key= openai_api_key
)

# Initialize 'text_for_RAG' if not present in session_state
if 'text_for_RAG' not in st.session_state:
    st.session_state.text_for_RAG = ""

st.title("Chatbot 2")

# Initialize 'messages' if not present in session_state
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Hallo, wie kann ich dir weiterhelfen?"):
    # Add user message to chat history

    
    if 'chatVerlauf_UserInteraction' not in st.session_state :
        st.chatVerlauf_UserInteraction = []
        st.chatVerlauf_UserInteraction.append({
        "role": "system",
           "content": f"You are a polite, courteous and helpful assistant who should help the user find the right shoes.." 
        })          


    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Update 'text_for_RAG' with user input
    user_input = prompt

    st.chatVerlauf_UserInteraction.append({"role": "user", "content": user_input})

    chat_User = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=st.chatVerlauf_UserInteraction
    )
    antwort_Message = chat_User.choices[0].message.content
    st.chatVerlauf_UserInteraction.append(antwort_Message)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown(antwort_Message)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown(antwort_Message)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": antwort_Message})
    print(st.chatVerlauf_UserInteraction)
