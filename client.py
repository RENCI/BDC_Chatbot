import streamlit as st
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.globals import set_debug, set_verbose
from utils.rag.chain import create_main_chain, create_time_filter
from utils import set_emb_llm
from collections import defaultdict
from langchain.load.dump import dumps
from langserve import RemoteRunnable
from streamlit_d3graph import d3graph
import math

set_verbose(True)

st.set_page_config(
    page_title="BDC Bot",
    page_icon="static/bot-light-32x32.png"
)

# Hide Streamlit menu
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

with open( "style.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)

# logo = "static/bdc-bot-logo-2.png"
bot_icon = "static/bot-32x32.png"
user_icon = "static/user-32x32.png"


#default_rag_chain = RemoteRunnable(url="http://localhost:8000/bdc-bot")
default_rag_chain = RemoteRunnable(url="https://bdcbot-s.apps.renci.org/bdc-bot")


doc_type_dict = defaultdict(lambda: "Source")
doc_type_dict['page'] = "BDC Web Page"
doc_type_dict['docs'] = "BDC Documentation"
doc_type_dict['update'] = "BDC Update"
doc_type_dict['event'] = "BDC Event"
doc_type_dict['faq'] = "BDC FAQ"

# Initialize D3 graph
d3 = d3graph(support=None)

def filter_sources(docs):
    # Split by the maximum distance between scores
    # XXX: Could use something more sophisticated such as Otsu thresholding...
    
    return docs
    
    # sort docs by score
    docs.sort(key=lambda x: x["metadata"]["score"], reverse=True)
    
    max_diff = 0
    max_diff_index = 0
    for i in range(len(docs)-1):        
        diff = docs[i+1]["metadata"]["score"] - docs[i]["metadata"]["score"]

        if (diff > max_diff):
            max_diff = diff
            max_diff_index = i

    top_docs = docs[0:max_diff_index+1]

    print(f"Kept {len(top_docs)} of {len(docs)}")

    return top_docs


def parse_context(context):
    docs = context

    if not docs:
        return []
    
    sources = []    
    
    top_docs = filter_sources(docs)
    
    for doc in top_docs:
        url = ""

        # source = doc["metadata"]["file_path"]
        if 'page_url' in doc["metadata"]:
            url = doc["metadata"]['page_url']
        elif 'remote_file_path' in doc["metadata"]:
            url = doc["metadata"]['remote_file_path'] 
        
        if not any(source.get('url') == url for source in sources):
            source = {
                'url': url,
                'doc_type': doc["metadata"]['doc_type'],    
                'metadata': doc["metadata"],
                'content': doc["page_content"],
                'retriever_type': doc["metadata"].get('retriever_type', 'NA'),
                'score': doc["metadata"].get('score', 'NA')
            }
            
            if 'title' in doc["metadata"]:
                source['title'] = doc["metadata"]['title']
            elif 'name' in doc["metadata"]:
                source['title'] = doc["metadata"]['name']
            elif 'file_name' in doc["metadata"]:
                source['title'] = doc["metadata"]['file_name']
            elif 'page_url' in doc["metadata"]:
                # only use the last part of the page_url
                source['title'] = doc["metadata"]['page_url'].split('/')[-1]
            else:
                source['title'] = doc["metadata"]['file_path']
            
            sources.append(source)
        else:
            print("Duplicate source found:", url)

    return output, sources

def source_link(url, title, type):
    return st.text(f"[{type}] {title}")

doc_type_order = [
    "faq",
    "docs",
    "page",
    "update",
    "event"
]

def draw_sources(sources, showSources):
    if not sources:
        return
    with st.expander(f"Source{'s' if len(sources) > 1 else ''}", expanded=showSources):
        # Group sources by doc_type using source_order
        grouped_sources = {doc_type: [] for doc_type in doc_type_order}
        for source in sources:
            doc_type = source.get('doc_type')
            if doc_type in grouped_sources:
                grouped_sources[doc_type].append(source)
            else:
                print(f"Unknown doc_type: {doc_type} for source {source['title']}")

        # Display sources by group in order
        for doc_type in doc_type_order:
            group = grouped_sources[doc_type]
            if group:
                header = f"**{doc_type_dict.get(doc_type, doc_type)}**"
                source_lines = [header]
                for source in group:
                    line = f"<a href='{source['url']}' target='_blank'>{source['title']}</a>"
                    source_lines.append(line)
                # Join lines with a line break and render via markdown
                st.markdown("<br>".join(source_lines), unsafe_allow_html=True)

def string_to_color(s):
    # Simple hash to color hex (for demonstration)
    import hashlib
    if not s:
        s = "default"
    return '#' + hashlib.md5(s.encode()).hexdigest()[:6]

def process_kg(kg):
    # Check if kg has no nodes or edges before displaying
    if not kg or 'nodes' not in kg or 'edges' not in kg:
        return None, None

    # Build node and edge lists
    nodes = kg.get('nodes', [])
    edges = kg.get('edges', [])

    if not nodes or not edges:
        return None, None

    # Build node dataframe for d3graph
    import pandas as pd
    node_ids = []
    node_labels = []
    node_categories = []
    node_colors = []
    for node in nodes:
        node_id = node.get('id')
        node_ids.append(node_id)
        node_labels.append(node.get('name', node_id))
        category = node.get('category', ["biolink:NamedThing"])[0]
        node_categories.append(category.replace("biolink:", ""))
        node_colors.append(string_to_color(category))
    df = pd.DataFrame({
        'label': node_labels,
        'category': node_categories,
        'color': node_colors
    }, index=node_ids)

    # Build adjacency matrix for d3graph
    adjmat = pd.DataFrame(0, index=node_ids, columns=node_ids)
    for edge in edges:
        source = edge.get('subject')
        target = edge.get('object')
        if source in node_ids and target in node_ids:
            adjmat.at[source, target] = 1

    return adjmat, df

def draw_additional_response(response, response_title, show_response, kg=None):
    with st.expander(response_title, expanded=show_response):
        st.markdown(response)

        if kg is not None:
            adjmat, df = process_kg(kg.get('knowledge_graph'))

            if adjmat is None or df is None:
                return
            
            st.markdown("\n---\nKnowledge Graph:")
            
            d3.graph(adjmat)
            d3.set_node_properties(label=df['label'].values, color=df['color'].values)
            d3.show(show_slider=False, save_button=False)
                
current_chain = default_rag_chain



def format_predefined_response(predefined_response_list, predefined_context, main_response):
    predefined_response = ""
    if predefined_response_list:
        for i, predefined_response in enumerate(predefined_response_list):
            predefined_response += f"{predefined_context[i]["topic"]} response: {predefined_response}\n"
    else:
        predefined_response = "No predefined response found"
    
    
    
    if main_response:
        predefined_response = f"Main response: {main_response}\n\n {predefined_response}"

    return predefined_response


#with st.sidebar:
#    st.header("BDC Resources")
#    st.link_button("Website", "https://biodatacatalyst.nhlbi.nih.gov/", icon="🌐", use_container_width=True)
#    st.link_button("Documentation", "https://bdcatalyst.gitbook.io/", icon="📖", use_container_width=True)
#    st.link_button("Support", "https://bdcatalyst.freshdesk.com/", icon="🛟", use_container_width=True)

introduction = """
This is a prototype of the BDCBot in development at RENCI.
If you are a tester, please complete [this form](https://example.com) after your test.
If you have navigated to this page in error, please close your browser window.
If you wish to reach someone regarding this prototype, please contact 
[David Borland](mailto:borland@renci.org) or [Nathalie Volkheimer](mailto:nathalie@renci.org).

---
"""

# Set the title for the Streamlit app
# st.image(logo, width=200)
st.text('[BDCBOT TEST]')
st.markdown(introduction)

# Initialize chat history
if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'displayed_history' not in st.session_state:
    st.session_state['displayed_history'] = []

if 'metadatas' not in st.session_state:
    st.session_state['metadatas'] = []

greeting = """
Hello! I am the NHLBI BioData Catalyst® Chatbot, also known as BDCBot.
I am AI powered, and here to support you on your blood, heart, lung
or sleep research journey.

I have been trained on public websites, but specifically developed to answer questions based on approved
BDC documentation. My answers will be as accurate and as current as the
documentation I have available. If you want to double check my answers I
would encourage you to check the sources outlined in my responses and/or
contact the [BDC HelpDesk](https://biodatacatalyst.nhlbi.nih.gov/help-and-support/contact-us/). 
BDC’s support team isn’t just AI powered; we have humans to help you one on one in live video chat by appointment too!

Not sure what to ask? Here are some example questions.
"""

sample_prompts = [
    "How can I find datasets in BDC?",
    "Can I download data from BDC?",
    "Does BDC have TOPMed data in it?",
    "Where can I find the RECOVER dataset?",
    #"Does BDC use AWS, Azure or Google?",
    "Does BDC cost money to use?",
    #"Can I import tools into BDC?",
    "Does BDC meet the Fisma-moderate security environment requirements?",
    # "Can I bring PHI into BDC?",
]

# Randomly select six prompts
#random_prompts = random.sample(sample_prompts, 4)

# Callback function to update the state
def handle_click_sample_prompt(prompt):
    st.session_state['sample_prompt_button_pressed'] = prompt

with st.chat_message('bdc-assistant'):
    st.markdown(greeting)

    with st.container():
        # Initialize button state in session state
        if 'sample_prompt_button_pressed' not in st.session_state:
            st.session_state['sample_prompt_button_pressed'] = ""
        
        st.markdown(
            """
            <style>
            /* these styles align button sizes in the sample button grid */
            .stButton {
                display: flex;
                & > button {
                    padding: 1rem;
                    font-size: 1rem;
                    flex: 1;
                    height: 4rem;
                }
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        
        # sample prompt buttons
        num_rows = math.floor(len(sample_prompts) / 2)
        button_rows = [st.columns(2), st.columns(2), st.columns(2)]
        for r, row in enumerate(button_rows):
            this_row_prompts = sample_prompts[0 + r*2:2 + r*2]
            for c, prompt in enumerate(this_row_prompts):
                button_rows[r][c].button(
                    prompt,
                    key=f"example_prompt_{r}_{c}",
                    on_click=handle_click_sample_prompt, 
                    args=(prompt,)
                )

if prompt := (st.chat_input("Ask a question") or st.session_state['sample_prompt_button_pressed']):
    display_text = ""
    context = None
    
    for i in range(len(st.session_state['displayed_history'])):
        role, content, sources = st.session_state['displayed_history'][i]
        bdc_response = ""
        dug_response = ""
        if i%2 == 1:
            j = i//2
            bdc_response = st.session_state['metadatas'][j].get("bdc_response", "")
            dug_response = st.session_state['metadatas'][j].get("dug_response", "")
        with st.chat_message(role):
            st.markdown(content)
            if sources:
                draw_sources(sources, False)
            if bdc_response and dug_response:
                #draw_additional_response(bdc_response, "BDC Response", False)
                draw_additional_response(dug_response, "DugBot Response", False)
        
    with st.chat_message('using-bdc'):
        st.markdown(prompt)

    with st.chat_message('bdc-assistant'):
        response_container = st.empty()

        # Add spinner while thinking
        with st.spinner("Generating response...", show_time=True):

            #print("chat_history: ", st.session_state['history'])
            
            res = current_chain.invoke({"input": prompt, "chat_history": st.session_state['history']})
            
            #print("current_chain.invoke: \n", res)                   

            # print keys from res
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            print("res keys: ", res.keys())
            
            # Create answer from response
            answer = ""


            # XXX             
            # FIRST: Get working with answer string, as is, ignoring code
            # SECOND: Refactor to store the full response in session state and create a function that displays here and for previous responses

            if res.get("guardrail_response", None):
                answer = res["guardrail_response"]
            else:
                bdc_response = res.get("bdc_response", "")
                dug_response = res.get("dug_response", "")
                dug_response += "\n\n*Visit [DugBot](https://search-dev.biodatacatalyst.renci.org/chat-v2/) to continue this conversation.*"
                dug_kg = res.get("dug_context", {}).get("knowledge_graph", None)
                combined_response = res.get("combined_response", f"{bdc_response}\n\n{dug_response}")
                
                if bdc_response:
                    answer = bdc_response
                elif dug_response:
                    answer = dug_response
                else:
                    answer = ""
                

                
                # format_predefined_response(predefined_response_list, predefined_context, main_response):
#                if res.get("prededined_context", {}):
#                    if res["prededined_context"].get("flag", None) == 'r':
#                        answer = format_predefined_response(res.get("predefined_response", []), res["prededined_context"], None)
#                    elif res["prededined_context"].get("flag", None) == 'a':
#                        answer = format_predefined_response(res.get("predefined_response", []), res["prededined_context"], answer)
            

            
            
            
            if res.get("predefined_response", None):
                answer = res.get("predefined_response", ["predefined_response (not found)"])[0]
            elif res.get("bdc_response", None) and res.get("dug_response", None):
                #answer = res.get("response", "")
                # For demo, use bdc response instead of combined response
                answer = res["bdc_response"]
            elif res.get("bdc_response", None):
                answer = res["bdc_response"]
            elif res.get("dug_response", None):
                answer = res["dug_response"] 
            

            
            
            context = res.get("context", [])
            
            for i, doc in enumerate(context):
                context[i] = doc.dict()
            
            
            print("bot answer: ", answer)
            
            display_answer = answer
            # if res.get("flag", None) == 'a':
            #     display_answer += "\n\n" + res.get("predefined_response", "predefined_response (not found)")
            #if res.get("dug_response", None):
            #    display_answer += "\n\nVisit the [DUG Bot](https://search-dev.biodatacatalyst.renci.org/chat-v2/) for more information."
            
            
            # bdc_response = res.get("bdc_response", "")
            # dug_response = res.get("dug_response", "")
            # dug_kg = res.get("dug_kg")
            # print("flag: ", res["flag"])
            
            # display_text += answer

            # if res.get("dug_response", None):
            #     dug_response += "\n\n*Visit [DugBot](https://search-dev.biodatacatalyst.renci.org/chat-v2/) to continue this conversation.*"

            sources = parse_context(context)


            response_container.markdown(answer, unsafe_allow_html=True)

            draw_sources(sources, False)

            if bdc_response and dug_response:
                #draw_additional_response(bdc_response, "BDC Response", False)
                draw_additional_response(dug_response, "DugBot Response", False, dug_kg)
    
    # st.session_state['history'].extend([dumps(HumanMessage(content=prompt)), dumps(AIMessage(content=answer))])
    st.session_state['history'].extend([(HumanMessage(content=prompt)), (AIMessage(content=answer))])
    st.session_state['displayed_history'].append(('using-bdc', prompt, None))
    st.session_state['displayed_history'].append(('bdc-assistant', answer, sources))
    st.session_state['metadatas'].append(res)

st.markdown(
    """
<style>
    .disclaimer {
        display: block;
        position: fixed !important;
        bottom: 1.5rem;
        left: 0;
        right: 1.5rem;
        height: 0;
        margin-top: -1rem;
        text-align: right;
        font-style: italic;
        font-size: 80%;
        color: #988;
        z-index: 9999;
    }
</style>
<div class="disclaimer">
    <a href="https://github.com/renci/bdc_chatbot">Click here</a> for more information on how this works.
</div>
    """,
    unsafe_allow_html=True
)