# from langchain_community.document_loaders import GitbookLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# from fastapi import FastAPI
# from langserve import add_routes

# from langchain.chains import RetrievalQA
# from langchain.chains import create_history_aware_retriever
# from langchain import hub





import streamlit as st
# from langchain.chains import ConversationalRetrievalChain


# from langchain.chains import create_history_aware_retriever
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.messages import HumanMessage, AIMessage

# from langchain.callbacks.base import BaseCallbackHandler



# from typing import AsyncIterator, Iterator

# from langchain_core.document_loaders import BaseLoader
# from langchain_core.documents import Document

# import json


# # from langchain_core.pydantic_v1 import BaseModel, Field
# from pydantic import BaseModel, Field
# from typing import Optional

# from langchain.chains.query_constructor.ir import (
#     Comparator,
#     Comparison,
#     Operation,
#     Operator,
#     StructuredQuery,
# )
# from langchain.retrievers.self_query.chroma import ChromaTranslator

# from langchain_community.query_constructors.chroma import ChromaTranslator


# # from babel.dates import format_date, format_datetime, format_time
# from datetime import datetime, timedelta



from langchain.globals import set_debug
set_debug(True)



import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

from langchain_community.embeddings import OllamaEmbeddings



from rag.chain import rag_chain_constructor, construct_time_filter
from utils import set_emb_llm


from collections import defaultdict

from langchain.load.dump import dumps



st.set_page_config(
    page_title="BDC Bot",
    page_icon="static/bot-light-32x32.png"
)


with open( "style.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)
    

logo = "static/bdc-bot-logo-2.png"
bot_icon = "static/bot-32x32.png"
user_icon = "static/user-32x32.png"


@st.cache_resource
def init_vars(retriever_top_k = 5, default_rag_filter = None):
    emb, llm, DB_PATH = set_emb_llm()

    

    vectorstore = Chroma(persist_directory=DB_PATH,embedding_function=emb)

    if default_rag_filter is None:
        default_rag_filter = construct_time_filter()
    
    # print("default_rag_filter: ", default_rag_filter)
    
    default_retriever = vectorstore.as_retriever(
        search_kwargs = {"k": retriever_top_k, 
                         "filter": default_rag_filter, 
                         })
    # default_retriever = vectorstore.as_retriever(search_kwargs = {"k": retriever_top_k})
    
    
    return llm, emb, vectorstore, default_retriever, retriever_top_k













llm, emb, vectorstore, default_retriever, retriever_top_k = init_vars(retriever_top_k=5)


default_rag_chain = rag_chain_constructor(default_retriever, llm, vectorstore)




doc_type_dict = defaultdict(lambda: "Source")
doc_type_dict['page'] = "BDC Web Page"
doc_type_dict['fellow'] = "BDC Fellow"
doc_type_dict['update'] = "BDC Update"
doc_type_dict['event'] = "BDC Event"




def parse_text(answer, context) -> str:

    # print("\n\nanswer: ```"+answer+"```\n\n")
    
    
    
    output = answer
    docs = context
    
    sources = []
    titles = []
    contents = []
    
    if not docs:
        return output
    
    for doc in docs:
        # source = doc.metadata["file_path"]
        if 'page_url' in doc.metadata:
            source = doc.metadata['page_url']
        else:
            source = doc.metadata['remote_file_path']      
        
        if not source in sources:
            doc_type = doc.metadata['doc_type']
            sources.append(source)
            if 'title' in doc.metadata:
                titles.append(f"{doc_type_dict[doc_type]}: {doc.metadata['title']}")
            elif 'name' in doc.metadata:
                titles.append(f"{doc_type_dict[doc_type]}: {doc.metadata['name']}")
            elif 'page_url' in doc.metadata:
                # only use the last part of the page_url
                titles.append(f"{doc_type_dict[doc_type]}: {doc.metadata['page_url'].split('/')[-1]}")
            else:
                titles.append(f"{doc_type_dict[doc_type]}: {doc.metadata['file_path']}")
            contents.append(doc.page_content)


    output += "\n###### "
    if len(sources) == 1:
        output += "Source:\n"
    elif len(sources) > 1:
        output += "Sources:\n"

    for i, source in enumerate(sources):
        # remove interim-bdc-website/ from the source path
        # path = source.replace("interim-bdc-website/", "")
        # output += f"###### {i + 1}. [{titles[i]}](https://github.com/stagecc/interim-bdc-website/tree/main/{path})\n\n"
        
        output += f"###### {i + 1}. [{titles[i]}]({source})\n\n"
        # output += f"```\n{contents[i][:100]}\n```\n\n\n"


    return output












# Set the title for the Streamlit app
#st.title("BDC Bot")
st.image(logo, width=200)


# Initialize chat history
if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'displayed_history' not in st.session_state:
    st.session_state['displayed_history'] = []



# Create containers for chat history and user input
response_container = st.container()
container = st.container()



current_chain = default_rag_chain

if prompt := st.chat_input("Ask a question"):
    display_text = ""
    context = None
    # st.session_state.messages.append({"role": "user", "content": prompt})

    
    for i in range(len(st.session_state['displayed_history'])):
        role, content = st.session_state['displayed_history'][i]
        st.chat_message(role, avatar=user_icon if role == "user" else bot_icon).write(content)
    
    
    
    with st.chat_message("user", avatar=user_icon):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=bot_icon):
        container = st.empty()

        
        container.markdown("Thinking...")
        
        
        stream = current_chain.stream({"input": prompt, "chat_history": st.session_state['history']})

        for chunk in stream:
            if 'context' in chunk:
                context = chunk['context']
            
            if 'answer' in chunk:
                display_text += chunk['answer']
            container.markdown(display_text)
        
        
        # res = current_chain.invoke({"input": prompt, "chat_history": st.session_state['history']})
        # context = res["context"]
        # answer = res["answer"]
        # display_text += answer
        
        
        
        # display_text = parse_message(stream)
        answer = display_text
        display_text = parse_text(answer, context)
        container.markdown(display_text)


        
        # result = conversational_chat(prompt)
        # answer = result["answer"]
        # display_text = parse_text(answer, result["context"])
        
        # container.markdown(display_text)
        


    
    # st.session_state['history'].extend([HumanMessage(content=prompt), answer])
    st.session_state['history'].extend([dumps(HumanMessage(content=prompt)), dumps(AIMessage(content=answer))])

    

    # st.session_state.messages.append({"role": "assistant", "content": response})
    
    st.session_state['displayed_history'].append(('user', prompt))
    st.session_state['displayed_history'].append(('assistant', display_text))





















