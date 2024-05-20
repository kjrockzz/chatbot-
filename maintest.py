

# !pip install python-dotenv
# !pip install streamlit
# !pip install openai
# !pip install langchain
# !pip install -U langchain-community
# !pip install tiktoken
# !pip install fastapi[all]
# !pip install pydantic==1.*
# !pip install faiss-cpu
import faiss
import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os
load_dotenv()

# 1. Vectorise the sales response csv data
loader = CSVLoader(file_path="salaries.csv")
documents = loader.load()

embeddings = OpenAIEmbeddings()
index_path ="faiss_index"

index_path = "faiss_index"

if os.path.exists(index_path):
    db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
else:    
    db = FAISS.from_documents(documents, embeddings)
    db.save_local(index_path)


def retrieve_info(query):
    similar_response = db.similarity_search(query, k=10)

    page_contents_array = [doc.page_content for doc in similar_response]

    # print(page_contents_array)

    return page_contents_array

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

template = """
You are a world class analyst .
Answer every question with the help of data given, you may use the data accordingly as per convinience and something ignore the data too,
as data comes from bakend the user may not aware of data
and you will follow ALL of the rules below:

1/ Response should be accurate to data provided and reponse should be formal ,

2/ If the data is  irrelevant or incomplete, then try to make certain assumption stating the reason for assumption

3/ reponse should be like an analyst and up to point not to long not too short

4/ you should not include request for data as data is not revied from user but fromm database

5/ if message just include greeting then you should also greet yourself and ignore the data part

Below is the Question:
{message}

Here is a data retireved from database having maximum similarity code:
{best_practice}

Please write the best  reply to following question :
"""

prompt = PromptTemplate(
    input_variables=["message", "best_practice"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)

def generate_response(message):
    best_practice = retrieve_info(message)
    print(best_practice)
    response = chain.run(message=message, best_practice=best_practice)
    print(response)
    return response

def main():
    
      print("write question")
      message = input()
      if message:
        result = generate_response(message)

      print(result)



import random
import time
def response_generator(message):
    response = generate_response(message)
    for word in response.split():
        yield word + " "
        time.sleep(0.05)



def stl_gen():
    st.title("Simple chat")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            response = st.write_stream(response_generator(prompt))
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

        

if __name__ == '__main__':
    # main()
    stl_gen()


