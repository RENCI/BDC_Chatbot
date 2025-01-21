import streamlit as st
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage
from langchain.globals import set_debug
from utils.rag.chain import rag_chain_constructor, construct_time_filter
from utils import set_emb_llm
from collections import defaultdict
from langchain.load.dump import dumps
from components.preview_link import preview_link

set_debug(True)

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
    
    default_retriever = vectorstore.as_retriever(
        search_kwargs = {"k": retriever_top_k, 
                         "filter": default_rag_filter, 
                         })
    default_retriever.search_type = "similarity_score_threshold"
    
    return llm, emb, vectorstore, default_retriever, retriever_top_k

llm, emb, vectorstore, default_retriever, retriever_top_k = init_vars(retriever_top_k=10)

default_rag_chain = rag_chain_constructor(default_retriever, llm, vectorstore, retriever_top_k=retriever_top_k, score_threshold=0.5)

doc_type_dict = defaultdict(lambda: "Source")
doc_type_dict['page'] = "BDC Web Page"
doc_type_dict['fellow'] = "BDC Fellow"
doc_type_dict['update'] = "BDC Update"
doc_type_dict['event'] = "BDC Event"

def filter_sources(docs):
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

def draw_sources(sources, showSources):
    if len(sources) == 0: 
        return    
    with st.expander(f"Source{'s' if len(sources) > 1 else ''}", expanded=showSources):
        for source in sources:
            preview_link(source['url'], source['title'], source["doc_type"])

current_chain = default_rag_chain

# Set the title for the Streamlit app
st.image(logo, width=200)

# Initialize chat history
if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'displayed_history' not in st.session_state:
    st.session_state['displayed_history'] = []

greeting = """
Hello! I’m your BioData Catalyst chatbot, here to assist you with finding
information about our program. I can answer questions about our initiatives,
events, newsletters, and content from our [website](https://biodatacatalyst.nhlbi.nih.gov).

While I strive to provide accurate and helpful responses, please note that
my answers may not always be 100% accurate. When in doubt, it's best to consult
official resources or contact our [support team](https://bdcatalyst.freshdesk.com/support/home) for clarification.

Feel free to use one of these sample prompts to get our conversation started.
"""

sample_prompts = [
    "What are the key features of BioData Catalyst?",
    "Tell me about upcoming BioData Catalyst events.",
    "How can I get started with BioData Catalyst?",
    "Can you summarize the latest BioData Catalyst updates?",
    "What is BDC's fellows program?",
    "What published research is based on BioData Catalyst?"
]

# Callback function to update the state
def handle_click_sample_prompt(prompt):
    st.session_state['sample_prompt_button_pressed'] = prompt

with st.chat_message('assistant', avatar=bot_icon):
    st.markdown(greeting)

    with st.container():
        # Initialize button state in session state
        if 'sample_prompt_button_pressed' not in st.session_state:
            st.session_state['sample_prompt_button_pressed'] = ""
        
        # sample prompt buttons
        button_rows = [st.columns(3), st.columns(3)]
        for r, row in enumerate(button_rows):
            for c, prompt in enumerate(sample_prompts[0:3]):
                button_rows[r][c].button(
                    prompt,
                    key=f"example_prompt_{r}_{c}",
                    on_click=handle_click_sample_prompt, 
                    args=(prompt,)
                )

    st.subheader("_How can I assist you today?_")


if prompt := (st.chat_input("Ask a question") or st.session_state['sample_prompt_button_pressed']):
    display_text = ""
    context = None
    
    for i in range(len(st.session_state['displayed_history'])):
        role, content, sources = st.session_state['displayed_history'][i]
        with st.chat_message(role, avatar=user_icon if role == "user" else bot_icon):
            st.markdown(content)
            if sources:
                draw_sources(sources, False)
    
    with st.chat_message("user", avatar=user_icon):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=bot_icon):
        response_container = st.empty()
        response_container.markdown("Thinking...")

        stream = current_chain.stream({"input": prompt, "chat_history": st.session_state['history']})

        for chunk in stream:
            if 'context' in chunk:
                context = chunk['context']
            if 'answer' in chunk:
                display_text += chunk['answer']
            response_container.markdown(display_text)
        
        # display_text = parse_message(stream)
        answer = display_text
        display_text, sources = parse_text(answer, context)
        response_container.markdown(display_text, unsafe_allow_html=True)

        draw_sources(sources, True)
    
    st.session_state['history'].extend([dumps(HumanMessage(content=prompt)), dumps(AIMessage(content=answer))])
    st.session_state['displayed_history'].append(('user', prompt, None))
    st.session_state['displayed_history'].append(('assistant', display_text, sources))

with st.sidebar:
    st.header("BDC Resources")
    st.link_button("Website", "https://biodatacatalyst.nhlbi.nih.gov/", icon="🌐", use_container_width=True)
    st.link_button("Documentation", "https://bdcatalyst.gitbook.io/", icon="📖", use_container_width=True)
    st.link_button("Support", "https://bdcatalyst.freshdesk.com/", icon="🛟", use_container_width=True)
