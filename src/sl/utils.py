from langchain_chroma import Chroma

from src.rag.chain import rag_chain_constructor, construct_time_filter
from src.utils import set_emb_llm

import streamlit as st
import json

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