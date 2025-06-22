import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import OpenAI
import os

st.set_page_config(page_title="SmartChat AI", page_icon="🤖")
st.title("🤖 SmartChat AI - Personal AI Assistant")

# 🔑 OpenAI API key input
openai_key = st.text_input("Enter your OpenAI API Key", type="password")

if openai_key:
    # 🧠 Setup AI and memory
    llm = OpenAI(temperature=0.7, openai_api_key=openai_key)
    memory = ConversationBufferMemory()

    # 💬 Prompt template
    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template="""
You are a helpful and intelligent assistant.
{history}
Human: {input}
AI:"""
    )

    chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

    # 🗣️ User input
    user_input = st.text_input("You:")
    if user_input:
        response = chain.run(input=user_input)
        st.markdown(f"**AI:** {response}")
else:
    st.info("Please enter your OpenAI API key above to start chatting.")