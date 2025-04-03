import streamlit as st
from langchain_ollama import OllamaLLM 
from langchain_core.prompts import ChatPromptTemplate
import os

# Streamlit page config
st.set_page_config(page_title="AI Chatbot", page_icon="🤖")

# Title
st.title("🤖 AI Chatbot (LangChain + Streamlit)")

# LangChain model setup
template = """
Answer the question below.

Here is the conversation history: {context}

Question:{question}

Answer: 
"""
# Initialize model
try:
    model = OllamaLLM(model="llama3")  # Ensure Ollama is installed and running
except Exception as e:
    st.error(f"Error initializing model: {e}")
    st.stop()

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Load context from file
def load_context(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    return "No context available."

# Define context file path (Update with your actual path)
context_file_path = "context.txt"  # Ensure the file exists
context = load_context(context_file_path)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = context

# Chat UI
st.write("### Chat with AI:")
chat_container = st.container()

# Display conversation history
for message in st.session_state.chat_history.split("\n"):
    if message.startswith("User:"):
        st.chat_message("user").write(message.replace("User: ", ""))
    elif message.startswith("AI:"):
        st.chat_message("assistant").write(message.replace("AI: ", ""))

# User input
user_input = st.chat_input("Type your message here...")

if user_input:
    if user_input.lower() == "exit":
        st.write("Exiting chat. Goodbye!")
    else:
        with st.spinner("Thinking..."):
            try:
                # Get AI response
                response = chain.invoke({"context": st.session_state.chat_history, "question": user_input})
                
                # Update history
                st.session_state.chat_history += f"\nUser: {user_input}\nAI: {response}"

                # Display messages
                st.chat_message("user").write(user_input)
                st.chat_message("assistant").write(response)
            except Exception as e:
                st.error(f"Error getting response: {e}")
