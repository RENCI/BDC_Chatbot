# from langchain_community.document_loaders import GitbookLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# from fastapi import FastAPI
# from langserve import add_routes

# from langchain.chains import RetrievalQA
# from langchain.chains import create_history_aware_retriever
# from langchain import hub





import streamlit as st

import streamlit.components.v1 as components
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



from src.rag.chain import rag_chain_constructor, construct_time_filter
from src.utils import set_emb_llm


from collections import defaultdict

from langchain.load.dump import dumps

import json


st.set_page_config(
    page_title="BDC Bot",
    page_icon="static/bot-light-32x32.png"
)

with open( "style.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)

logo = "static/bdc-bot-logo-2.png"
bot_icon = "static/bot-32x32.png"
user_icon = "static/user-32x32.png"





# <p id="p1"><a href="https://cnet.com">Cnet</a></p>
# <p id="p2"><a href="https://codegena.com">Codegena</a></p>
# <p id="p3"><a href="https://apple.com">Apple</a></p>


# js_script = """<script src="https://ajax.googleapis.com/ajax/libs/jquery/2.0.3/jquery.min.js"></script>
#   <link href="https://codegena.com/assets/css/image-preview-for-link.css" rel="stylesheet">     
#   <script type="text/javascript">
#     $(function() {
#                 $('a.web_preview').miniPreview({ prefetch: 'parenthover' });
#             });
#   </script> <script src="https://codegena.com/assets/js/image-preview-for-link.js"></script>
#   """
js_script = """<script src="https://ajax.googleapis.com/ajax/libs/jquery/2.0.3/jquery.min.js"></script>
<style>
.preview-tooltip {
    display: none;
    position: absolute;
    z-index: 1000;
    background: white;
    border: 1px solid #ddd;
    padding: 5px;
    max-width: 300px;
}
</style>
<script>
$(document).ready(function() {
    $('a[title]').each(function() {
        var preview = $(this).attr('data-preview');
        var tooltip = $('<div class="preview-tooltip"><img src="' + preview + '" width="300"></div>');
        $(this).append(tooltip);
        
        $(this).hover(
            function() { tooltip.show(); },
            function() { tooltip.hide(); }
        );
    });
});
</script>"""
# components.html(f"{js_script}")




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
    default_retriever.search_type = "similarity_score_threshold"
    # default_retriever = vectorstore.as_retriever(search_kwargs = {"k": retriever_top_k})
    
    
    return llm, emb, vectorstore, default_retriever, retriever_top_k








llm, emb, vectorstore, default_retriever, retriever_top_k = init_vars(retriever_top_k=10)


default_rag_chain = rag_chain_constructor(default_retriever, llm, vectorstore, retriever_top_k=retriever_top_k, score_threshold=0.5)




doc_type_dict = defaultdict(lambda: "Source")
doc_type_dict['page'] = "BDC Web Page"
doc_type_dict['fellow'] = "BDC Fellow"
doc_type_dict['update'] = "BDC Update"
doc_type_dict['event'] = "BDC Event"



def filter_sources(docs):
    # sort docs by score
    #docs = sorted(docs, key=lambda x: x.metadata["score"], reverse=True)
    
    # Split by the maximum distance between scores
    # XXX: Could use something more sophisticated such as Otsu thresholding...
    max_diff = 0
    max_diff_index = 0
    for i in range(len(docs)-1):        
        diff = docs[i+1].metadata["score"] - docs[i].metadata["score"]

        if (diff > max_diff):
            max_diff = diff
            max_diff_index = i

    top_docs = docs[0:max_diff_index+1]

    print(f"Kept {len(top_docs)} of {len(docs)}")

    return top_docs


def parse_text(answer, context) -> str:

    # print("\n\nanswer: ```"+answer+"```\n\n")
    
    
    
    output = answer
    docs = context

    if not docs:
        return output, []
    
    sources = []    
    
    top_docs = filter_sources(docs)
    
    for doc in top_docs:
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
                'metadata': doc.metadata,
                'content': doc.page_content,
                'score': doc.metadata['score']   
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
            
            sources.append(source)

    return output, sources




# Function to display the image on hover
def link_with_preview_on_hover(url, text, i):
    image_url = f"https://api.microlink.io?url={url}&screenshot=true&embed=screenshot.url"
    
    
    # Generate unique class names for each image
    hover_class = f'hoverable_{i}'
    tooltip_class = f'tooltip_{i}'
    image_popup_class = f'image-popup_{i}'

    # Define the unique CSS for each image
    hover_css = f'''
        .{hover_class} {{
            position: relative;
            display: inline-block;
            cursor: pointer;
        }}
        .{hover_class} .{tooltip_class} {{
            opacity: 0;
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            transition: opacity 0.5s;
            background-color: rgba(0, 0, 0, 0.8);
            color: #fff;
            padding: 4px;
            border-radius: 4px;
            text-align: center;
            white-space: nowrap;
        }}
        .{hover_class}:hover .{tooltip_class} {{
            opacity: 1;
        }}
        .{image_popup_class} {{
            position: absolute;
            display: none;
            background-image: none;
            width: 350px;
            height: 200px;
        }}
        .{hover_class}:hover .{image_popup_class} {{
            display: block;
            background-image: url({image_url});
            background-size: cover;
            z-index: 999;
        }}
    '''
    tooltip_css = f"<style>{hover_css}</style>"

    # Define the html for each image
    image_hover = f'''
        <div class="{hover_class}">
            <a href="{url}">{text}</a>
            <div class="{tooltip_class}">{url}</div>
            <div class="{image_popup_class}"></div>
        </div>
    '''
    
    # Write the dynamic HTML and CSS to the content container
    st.markdown(f'<p>{image_hover}{tooltip_css}</p>', unsafe_allow_html=True)








def draw_sources(sources, showSources):
    if len(sources) == 0: 
        return    

    with st.expander(f"Source{'s' if len(sources) > 1 else ''}", expanded=showSources):
        output = ""
        
        for i, source in enumerate(sources):                    
            output = ""                  

            link_with_preview_on_hover(source['url'], source['title'], i)
            
            metadata_str = json.dumps(source['metadata'], indent=4)

            output += f"<details>\n<summary>Metadata</summary>\n<p>{metadata_str}</p>\n</details>\n\n"
            
            output += f"<details>\n<summary>Content</summary>\n<p>{source['content']}</p>\n</details>\n\n"

            st.markdown(output, unsafe_allow_html=True)



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
        role, content, sources = st.session_state['displayed_history'][i]
        with st.chat_message(role, avatar=user_icon if role == "user" else bot_icon):
            st.markdown(content)

            if sources:
                 draw_sources(sources, False)
    
    
    
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
        container.markdown(display_text, unsafe_allow_html=True)

        draw_sources(sources, True)
        
        # result = conversational_chat(prompt)
        # answer = result["answer"]
        # display_text = parse_text(answer, result["context"])
        
        # container.markdown(display_text)
        


    
    # st.session_state['history'].extend([HumanMessage(content=prompt), answer])
    st.session_state['history'].extend([dumps(HumanMessage(content=prompt)), dumps(AIMessage(content=answer))])

    

    # st.session_state.messages.append({"role": "assistant", "content": response})
    
    st.session_state['displayed_history'].append(('user', prompt, None))
    st.session_state['displayed_history'].append(('assistant', display_text, sources))





















