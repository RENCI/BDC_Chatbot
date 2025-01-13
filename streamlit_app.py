
import re
from langchain_chroma import Chroma




import streamlit as st



from langchain_core.messages import HumanMessage, AIMessage





from langchain.globals import set_debug


set_debug(True)




from dotenv import load_dotenv



from langchain_community.embeddings import OllamaEmbeddings



from src.rag.chain import rag_chain_constructor



from collections import defaultdict

from langchain.load.dump import dumps

import json



from src.sl.utils import init_vars, parse_text, draw_sources


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
def initialize_variables(retriever_top_k = 5, default_rag_filter = None):
    return init_vars(retriever_top_k=retriever_top_k, default_rag_filter=default_rag_filter)


llm, emb, vectorstore, default_retriever, retriever_top_k = initialize_variables(retriever_top_k=10)




default_rag_chain = rag_chain_constructor(default_retriever, llm, vectorstore, retriever_top_k=retriever_top_k, score_threshold=0.5)




doc_type_dict = defaultdict(lambda: "Source")
doc_type_dict['page'] = "BDC Web Page"
doc_type_dict['fellow'] = "BDC Fellow"
doc_type_dict['update'] = "BDC Update"
doc_type_dict['event'] = "BDC Event"





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





















