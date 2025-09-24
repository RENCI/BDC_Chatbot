import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from collections import defaultdict
from langserve import RemoteRunnable
from streamlit_d3graph import d3graph
import math
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description="BDC Bot")
parser.add_argument("--timer", action="store_true", help="Enable timer display")
args = parser.parse_args()
timer_enabled = args.timer

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
    st.markdown( f"<style>{css.read()}</style>" , unsafe_allow_html= True)

# logo = "static/bdc-bot-logo-2.png"
bot_icon = "static/bot-32x32.png"
user_icon = "static/user-32x32.png"

# Set the server to use, local or remote
#default_rag_chain = RemoteRunnable(url="http://localhost:8000/bdc-bot")
default_rag_chain = RemoteRunnable(url="https://bdcbot-s.apps.renci.org/bdc-bot")

# Dict for source document types
doc_type_dict = defaultdict(lambda: "Source")
doc_type_dict["page"] = "BDC Web Page"
doc_type_dict["docs"] = "BDC Documentation"
doc_type_dict["update"] = "BDC Update"
doc_type_dict["event"] = "BDC Event"
doc_type_dict["faq"] = "BDC FAQ"

# Initialize D3 graph for knowledge graph visualization
d3 = d3graph(support=None)

def filter_sources(docs):
    # Split by the maximum distance between scores
    # XXX: Could use something more sophisticated such as Otsu thresholding...
    
    # Just returning all for now
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

# Parse bdc documentation context into sources
def parse_bdc_context(context):
    if not context:
        return []
  
    docs = []  
    for doc in context:
        docs.append(doc.model_dump())  
    
    top_docs = filter_sources(docs)
    
    sources = []  
    for doc in top_docs:
        url = ""

        # source = doc["metadata"]["file_path"]
        if "page_url" in doc["metadata"]:
            url = doc["metadata"]["page_url"]
        elif "remote_file_path" in doc["metadata"]:
            url = doc["metadata"]["remote_file_path"] 
        
        if not any(source.get("url") == url for source in sources):
            source = {
                "url": url,
                "doc_type": doc["metadata"]["doc_type"],    
                "metadata": doc["metadata"],
                "content": doc["page_content"],
                "retriever_type": doc["metadata"].get("retriever_type", "NA"),
                "score": doc["metadata"].get("score", "NA")
            }
            
            if "title" in doc["metadata"]:
                source["title"] = doc["metadata"]["title"]
            elif "name" in doc["metadata"]:
                source["title"] = doc["metadata"]["name"]
            elif "file_name" in doc["metadata"]:
                source["title"] = doc["metadata"]["file_name"]
            elif "page_url" in doc["metadata"]:
                # only use the last part of the page_url
                source["title"] = doc["metadata"]["page_url"].split("/")[-1]
            else:
                source["title"] = doc["metadata"]["file_path"]
            
            sources.append(source)
        else:
            print("Duplicate source found:", url)

    return sources

# Order to display document types
doc_type_order = [
    "faq",
    "docs",
    "page",
    "update",
    "event"
]

# Draw BDC documentation sources
def draw_sources(sources, showSources):
    if not sources:
        return
    with st.expander(f"Source{"s" if len(sources) > 1 else ""}", expanded=showSources):
        # Group sources by doc_type using source_order
        grouped_sources = {doc_type: [] for doc_type in doc_type_order}
        for source in sources:
            doc_type = source.get("doc_type")
            if doc_type in grouped_sources:
                grouped_sources[doc_type].append(source)
            else:
                print(f"Unknown doc_type: {doc_type} for source {source["title"]}")

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

# Color for knowledge graph nodes
def string_to_color(s):
    # Simple hash to color hex (for demonstration)
    import hashlib
    if not s:
        s = "default"
    return "#" + hashlib.md5(s.encode()).hexdigest()[:6]

# Process the dug knowledge graph for visualization
def process_kg(kg):
    # Check if kg has no nodes or edges before displaying
    if not kg or "nodes" not in kg or "edges" not in kg:
        return None, None

    # Build node and edge lists
    nodes = kg.get("nodes", [])
    edges = kg.get("edges", [])

    if not nodes or not edges:
        return None, None

    # Build node dataframe for d3graph
    import pandas as pd
    node_ids = []
    node_labels = []
    node_categories = []
    node_colors = []
    for node in nodes:
        node_id = node.get("id")
        node_ids.append(node_id)
        node_labels.append(node.get("name", node_id))
        category = node.get("category", ["biolink:NamedThing"])[0]
        node_categories.append(category.replace("biolink:", ""))
        node_colors.append(string_to_color(category))
    df = pd.DataFrame({
        "label": node_labels,
        "category": node_categories,
        "color": node_colors
    }, index=node_ids)

    # Build adjacency matrix for d3graph
    adjmat = pd.DataFrame(0, index=node_ids, columns=node_ids)
    for edge in edges:
        source = edge.get("subject")
        target = edge.get("object")
        if source in node_ids and target in node_ids:
            adjmat.at[source, target] = 1

    return adjmat, df


def draw_additional_response(response, response_title, show_response, kg=None):
    with st.expander(response_title, expanded=show_response):
        st.markdown(response)

        if kg is not None:
            adjmat, df = process_kg(kg)

            if adjmat is None or df is None:
                return
            
            st.markdown("\n---\nKnowledge Graph:")
            
            d3.graph(adjmat)
            d3.set_node_properties(label=df["label"].values, color=df["color"].values)
            d3.show(show_slider=False, save_button=False)

# Set the current chain to use to get response from server
current_chain = default_rag_chain

#with st.sidebar:
#    st.header("BDC Resources")
#    st.link_button("Website", "https://biodatacatalyst.nhlbi.nih.gov/", icon="🌐", use_container_width=True)
#    st.link_button("Documentation", "https://bdcatalyst.gitbook.io/", icon="📖", use_container_width=True)
#    st.link_button("Support", "https://bdcatalyst.freshdesk.com/", icon="🛟", use_container_width=True)

# Introduction text
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
st.text("[BDCBOT TEST]")
st.markdown(introduction)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "history" not in st.session_state:
    st.session_state["history"] = []

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
    st.session_state["sample_prompt_button_pressed"] = prompt

# Display user input
def display_input(input):
    st.markdown(input)

# Display a question and response 
def display_response(response, showBDCSources=False):
    if response.get("guardrail_response", None):
       st.write(response["guardrail_response"])
       return
    if response.get("predefined_response", None):
        for predefined in response.get("predefined_response", []):
            st.write(predefined)
    if response.get("bdc_response", None):
        st.write(response["bdc_response"])

        context = response.get("bdc_context", [])      
        draw_sources(parse_bdc_context(context), showBDCSources)
    if response.get("dug_response", None):

        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        print(response.get("dug_context", {}))

        dug_kg = response.get("dug_context", {}).get("knowledge_graph", None)
        draw_additional_response(response["dug_response"], "DugBot Response", False, dug_kg)
    #if response.get("combined_response", None):
    #    st.write(response["combined_response"])
    if response.get("code_response", None):
        st.write(response["code_response"])

with st.chat_message("bdc-assistant"):
    st.markdown(greeting)

    with st.container():
        # Initialize button state in session state
        if "sample_prompt_button_pressed" not in st.session_state:
            st.session_state["sample_prompt_button_pressed"] = ""
        
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

if prompt := (st.chat_input("Ask a question") or st.session_state["sample_prompt_button_pressed"]):   
    for history in st.session_state["history"]:        
        with st.chat_message("using-bdc"):
            st.empty()
            display_input(history.get("input", ""))
        with st.chat_message("bdc-assistant"):
            # Need empty to avoid stale greyed out elements with spinner
            st.empty()
            display_response(history.get("response", {}))

    with st.chat_message("using-bdc"):
        display_input(prompt)

    with st.chat_message("bdc-assistant"):
        # Add spinner while thinking
        with st.spinner("Generating response...", show_time=timer_enabled):            
            # Get response from server
            response = current_chain.invoke({"input": prompt, "chat_history": st.session_state["chat_history"]})
        
        display_response(response, showBDCSources=True)
        
        # Create answer from response to store for chat history
        answer = ""
        separator = "\n\n"

        # Combine multiple responses if they exist
        if response.get("guardrail_response", None):
            answer += response["guardrail_response"]
        if response.get("predefined_response", None):
            for predefined in response.get("predefined_response", []):
                answer += separator
                answer += predefined
        if response.get("bdc_response", None):
            answer += separator
            answer += response["bdc_response"]
        if response.get("dug_response", None):
            answer += separator
            answer += response["dug_response"]
        #if response.get("combined_response", None):
        #    answer += separator
        #    answer += response["combined_response"]
        if response.get("code_response", None):
            answer += separator
            answer += response["code_response"]

    # Store prompts and answer for history to send as context for converstion   
    st.session_state["chat_history"].extend([(HumanMessage(content=prompt)), (AIMessage(content=answer))])

    # Store full response for UI display
    history_object = {
        "input": prompt,
        "response": response
    }   

    st.session_state["history"].append(history_object)

# Disclaimer at bottom right
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