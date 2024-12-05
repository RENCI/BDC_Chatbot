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


st.set_page_config(page_title="BioData Catalyst Chatbot")

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




def parse_text(answer, context) -> str:

    # print("\n\nanswer: ```"+answer+"```\n\n")
    
    
    
    output = answer
    docs = context
    
    sources = []
    titles = []
    contents = []
    
    
    metadatas = []
    
    # sort docs by score
    docs = sorted(docs, key=lambda x: x.metadata["score"], reverse=True)
    
    
    
    score_diff = []
    for i in range(len(docs)-1):
        score_diff.append(docs[i].metadata["score"] - docs[i+1].metadata["score"])
    
    max_score_diff = max(score_diff)
    max_score = max([d.metadata["score"] for d in docs])
    
    print("max_score_diff: ", max_score_diff)
    print("max_score: ", max_score)
    

    
    
    if not docs:
        return output
    
    
    # get docs with max score - max score diff
    top_docs = [docs[i] for i in range(len(docs)) if docs[i].metadata["score"] >= max_score - max_score_diff]
    
    # at least 1 doc
    if len(top_docs) == 0:
        top_docs = [docs[0],]
    
    
    for doc in top_docs:
        # source = doc.metadata["file_path"]
        if 'page_url' in doc.metadata:
            source = doc.metadata['page_url']
        else:
            source = doc.metadata['remote_file_path']      
        
        if not source in sources:
            doc_type = doc.metadata['doc_type']
            sources.append(source)
            metadatas.append(doc.metadata)
            contents.append(doc.page_content)
            
            if 'title' in doc.metadata:
                titles.append(f"{doc_type_dict[doc_type]}: {doc.metadata['title']}")
            elif 'name' in doc.metadata:
                titles.append(f"{doc_type_dict[doc_type]}: {doc.metadata['name']}")
            elif 'page_url' in doc.metadata:
                # only use the last part of the page_url
                titles.append(f"{doc_type_dict[doc_type]}: {doc.metadata['page_url'].split('/')[-1]}")
            else:
                titles.append(f"{doc_type_dict[doc_type]}: {doc.metadata['file_path']}")
            

    
    
    
    

    if len(sources) == 1:
        output += "\n\n#### Source:\n"
    elif len(sources) > 1:
        output += "\n\n#### Sources:\n"

    for i, source in enumerate(sources):
        # Add the link
        output += f'###### {i + 1}. <a href="{source}" class="web_preview">{titles[i]}</a>\n'
        
        # Add a small preview using st.image in an expander
        if source.startswith(('http://', 'https://')):
            output += f"""<details>
                <summary>Preview</summary>
                <img src="https://api.microlink.io?url={source}&screenshot=true&embed=screenshot.url" width="300">
                </details>\n"""
        # # Update the link format to include the preview image
        # if source.startswith(('http://', 'https://')):
        #     preview_url = f"https://api.microlink.io?url={source}&screenshot=true&embed=screenshot.url"
        #     output += f'###### {i + 1}. <a href="{source}" title="Preview" data-preview="{preview_url}">{titles[i]}</a>\n'
        # else:
        #     output += f'###### {i + 1}. <a href="{source}">{titles[i]}</a>\n'



        metadata_str = json.dumps(metadatas[i], indent=4)
        # output += f"""<details>\n<summary markdown="span">Metadata</summary>\n```json\n{metadata_str}\n```\n</details>\n\n"""
        output += f"<details>\n<summary>Metadata</summary>\n<p>{metadata_str}</p>\n</details>\n\n"
        

        output += f"<details>\n<summary>Content</summary>\n<p>{contents[i]}</p>\n</details>\n\n"
        
    # print("\n\n\n"+output[:1000]+"\n\n\n")
    
    return output












# Set the title for the Streamlit app
st.title("BioData Catalyst Chatbot")



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
        st.chat_message(role).markdown(content, unsafe_allow_html=True)
    
    
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
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
        container.markdown(display_text, unsafe_allow_html=True)


        
        # result = conversational_chat(prompt)
        # answer = result["answer"]
        # display_text = parse_text(answer, result["context"])
        
        # container.markdown(display_text)
        


    
    # st.session_state['history'].extend([HumanMessage(content=prompt), answer])
    st.session_state['history'].extend([dumps(HumanMessage(content=prompt)), dumps(AIMessage(content=answer))])

    

    # st.session_state.messages.append({"role": "assistant", "content": response})
    
    st.session_state['displayed_history'].append(('user', prompt))
    st.session_state['displayed_history'].append(('assistant', display_text))





















