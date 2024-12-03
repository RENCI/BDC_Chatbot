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
    
    if not docs:
        return output
    
    for doc in docs:
        url = ""

        # source = doc.metadata["file_path"]
        if 'page_url' in doc.metadata:
            url = doc.metadata['page_url']
        elif 'remote_file_path' in doc.metadata:
            url = doc.metadata['remote_file_path']
        
        if not any(source.get('url') == url for source in sources):
            source = {
                'url': url,
                'doc_type': doc.metadata['doc_type'],                
            }

            if 'title' in doc.metadata:
                source['title'] = doc.metadata['title']
            elif 'name' in doc.metadata:
                source['title'] = doc.metadata['name']
            elif 'page_url' in doc.metadata:
                # only use the last part of the page_url
                source['title'] = doc.metadata['page_url'].split('/')[-1]
            else:
                source['title'] = doc.metadata['file_path']

            source['similarity_score'] = doc.metadata['similarity_score']

            sources.append(source)

    return output, sources


def draw_sources(sources):
    print(sources)

    if len(sources) == 0: 
        return    

    with st.expander(f"Generated from {len(sources)} source{'s' if len(sources) > 1 else ''}"):
        output = ""

        for i, source in enumerate(sources):                    
            output += f"###### {i + 1}. [{source['doc_type']}: {source['title']}]({source['url']})\n\n"

        st.markdown(output)

# Set the title for the Streamlit app
#st.title("BDC Bot")
st.image(logo, width=200)
#st.logo(logo)


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
        display_text, sources = parse_text(answer, context)
        container.markdown(display_text)

        draw_sources(sources)
        
        # result = conversational_chat(prompt)
        # answer = result["answer"]
        # display_text = parse_text(answer, result["context"])
        
        # container.markdown(display_text)


    
    # st.session_state['history'].extend([HumanMessage(content=prompt), answer])
    st.session_state['history'].extend([dumps(HumanMessage(content=prompt)), dumps(AIMessage(content=answer))])

    

    # st.session_state.messages.append({"role": "assistant", "content": response})
    
    st.session_state['displayed_history'].append(('user', prompt))
    st.session_state['displayed_history'].append(('assistant', display_text))





















