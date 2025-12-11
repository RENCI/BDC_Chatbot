from matplotlib.pylab import source
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from collections import defaultdict
from langserve import RemoteRunnable
from streamlit_d3graph import d3graph
from d3graph import vec2adjmat, import_example
import math
import argparse
import json
from utils.client.colors import ColorScale

# Parse command line arguments
parser = argparse.ArgumentParser(description="BDC Bot")
parser.add_argument("--show_timer", action="store_true", help="Enable timer display")
parser.add_argument("--use_remote_server", action="store_true", help="Use remote server on Sterling")
args = parser.parse_args()
show_timer = args.show_timer
use_remote_server = args.use_remote_server


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
runnable_url = "https://bdcbot-s.apps.renci.org/bdc-bot" if use_remote_server else "http://localhost:8000/bdc-bot"
default_rag_chain = RemoteRunnable(url=runnable_url)

# Dict for source document types
doc_type_dict = defaultdict(lambda: "Source")
doc_type_dict["page"] = "BDC Web Page"
doc_type_dict["docs"] = "BDC Documentation"
doc_type_dict["update"] = "BDC Update"
doc_type_dict["event"] = "BDC Event"
doc_type_dict["faq"] = "BDC FAQ"
doc_type_dict["video"] = "BDC Video"

# Initialize D3 graph for knowledge graph visualization
d3 = d3graph(support=None)

def filter_sources(docs, max_sources=5):    
    # Just returning top max_sources for now
    return docs[0:max_sources]

    # Split by the maximum distance between scores
    # XXX: Could use something more sophisticated such as Otsu thresholding...
    
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
    index = 0
    for doc in top_docs:
        url = ""

        # source = doc["metadata"]["file_path"]
        if "page_url" in doc["metadata"]:
            url = doc["metadata"]["page_url"]
        elif "remote_file_path" in doc["metadata"]:
            url = doc["metadata"]["remote_file_path"] 
        elif "timestamp_url" in doc["metadata"]:
            url = doc["metadata"]["timestamp_url"]
        
        if not any(source.get("url") == url for source in sources):
            source = {
                "index": index,
                "url": url,
                "doc_type": doc["metadata"]["doc_type"],    
                "metadata": doc["metadata"],
                "content": doc["page_content"],
                "retriever_type": doc["metadata"].get("retriever_type", None),
                "score": doc["metadata"].get("score", None)
            }
            index += 1
            
            if "title" in doc["metadata"]:
                source["title"] = doc["metadata"]["title"]
            elif "name" in doc["metadata"]:
                source["title"] = doc["metadata"]["name"]
            elif "page_url" in doc["metadata"]:
                # only use the last part of the page_url
                source["title"] = doc["metadata"]["page_url"].split("/")[-1]
            elif "file_name" in doc["metadata"]:
                source["title"] = doc["metadata"]["file_name"]
            else:
                source["title"] = doc["metadata"]["file_path"]

            # Add start seconds to title for videos
            if "start_seconds" in doc["metadata"]:
                start_seconds = int(doc["metadata"]["start_seconds"])
                minutes = start_seconds // 60
                seconds = start_seconds % 60
                source["title"] += f" @{minutes}m {seconds}s"
            
            sources.append(source)
        else:
            print("Duplicate source found:", url)

    return sources

# Order to display document types
doc_type_order = [
    "faq",
    "docs",
    "page",
    "video",
    "update",
    "event",
]

# Draw BDC documentation sources
def draw_sources(sources, showSources):
    if not sources:
        return
    with st.expander(f":material/source: Source{"s" if len(sources) > 1 else ""}", expanded=showSources):
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

# Process the dug knowledge graph for visualization
def process_kg(kg):
    # Check if kg has no nodes or edges before displaying
    if not kg or "nodes" not in kg or "edges" not in kg:
        return None, None

    with open("knowledge_graph.json", "w") as f:
        json.dump(kg, f, indent=2)

    # Build node and edge lists
    nodes = kg.get("nodes", [])
    edges = kg.get("edges", [])

    if not nodes or not edges:
        return None, None

    # Build node dataframe for d3graph
    import pandas as pd
    
    node_data = {}
    node_ids = []
    
    for node in nodes:
        node_id = node.get("id")
        if not node_id:
            continue  # Skip nodes without IDs
            
        name = node.get("name", "Unknown")
        categories = node.get("category", ["Unknown"])
        categories = [cat.replace("biolink:", "") for cat in categories]
        description = node.get("description", "")

        node_ids.append(node_id)
        node_data[node_id] = {
            "name": name,
            "category": categories[0],
            "tooltip": f'Name: {name}\n\nCategor{"ies" if len(categories) > 1 else "y"}: {", ".join(categories)}\n\n{"Description: " + description if description else ""}\n\nID: {node_id}'
        }

    # Create DataFrame with node_ids as index
    node_df = pd.DataFrame([node_data[nid] for nid in node_ids], index=node_ids)

    # Build adjacency matrix for d3graph
    adjmat_df = pd.DataFrame(0, index=node_ids, columns=node_ids)
    for edge in edges:
        source = edge.get("subject")
        target = edge.get("object")
        if source in node_ids and target in node_ids:
            adjmat_df.at[source, target] = 1
            adjmat_df.at[target, source] = 1

    return adjmat_df, node_df

def draw_dug_response(response, response_title, show_response, kg=None):
    with st.expander(f":material/find_in_page: {response_title}", expanded=show_response):
        st.markdown(response)
        #st.markdown("*Powered by DugBot* [:material/launch:](https://search-dev.biodatacatalyst.renci.org/chat-v2/)")

        if kg is not None:
            adjmat, df = process_kg(kg)

            if adjmat is None or df is None:
                return
            
            st.markdown("\n---\nKnowledge Graph:")
            
            d3.graph(adjmat)
            d3.set_node_properties(
                label="",
                color=df["category"].values,
                cmap="Set1",
                opacity="",
                tooltip=df["tooltip"].values,
            )
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
    "What is BDC?",
    #"How can I find datasets in BDC?",
    "What kind of data is in BDC?",
    "Can I download data from BDC?",
    "Does BDC meet the Fisma-moderate security environment requirements?",
    #"Where can I find the RECOVER dataset?",
    #"Does BDC use AWS, Azure or Google?",
    "Does BDC cost money to use?",
    #"Can I import tools into BDC?",
    # "Can I bring PHI into BDC?",
    "Does BDC have TOPMed data in it?",
]

# Randomly select six prompts
#random_prompts = random.sample(sample_prompts, 4)

# Callback function to update the state
def handle_click_sample_prompt(prompt):
    st.session_state["sample_prompt_button_pressed"] = prompt

# Display user input
def display_input(input):
    st.markdown(input)

# Response type icon
def response_type_icon(response_type):
    if response_type == "guardrail_response":
        return ":material/block: "
    elif response_type == "predefined_response":
        return ":material/verified: "

# Get response text
def get_response_text(response, key, missing_text):
    # XXX: Removing markdown code blocks for now, but this should be handle by the server
    return response[key].replace("```", "") if response.get(key, None) else missing_text


# Display a question and response 
def display_response(response, showBDCSources=False):
    keys = response.keys()

    if "guardrail_response" in keys:
        st.markdown(f"{get_response_text(response, "guardrail_response", "Missing guardrail response")}", help=":material/block: This response was generated by a guardrail as the user prompt was out of scope.")
        return
    if "predefined_response" in keys:
        for predefined in response.get("predefined_response", []):
            st.markdown(predefined, help=":material/verified: This is a predefined response to meet strict NHLBI guidelines.")
    if "bdc_response" in keys:        
        st.markdown(f"{get_response_text(response, "bdc_response", "Missing BDCBot response")}")
        context = response.get("bdc_context", [])
        if ("dug_response" in keys):
            st.markdown("*Open the DugBot response below for more detailed information on studies and datasets.*")
        draw_sources(parse_bdc_context(context), showBDCSources)
    if "dug_response" in keys:
        dug_kg = response.get("dug_context", {}).get("knowledge_graph", None)
        draw_dug_response(get_response_text(response, "dug_response", "Missing DugBot response"), "DugBot Response", False, dug_kg)
    #if response.get("combined_response", None):
    #    st.markdown(response["combined_response"])
    if "code_response" in keys:
        context = response.get("code_context", [])

        # Get the context in the correct format
        code_blocks = []  
        for code_block in context:
            code_blocks.append(code_block.model_dump().get("metadata", {})) 

        # Group by file name
        file_names = []
        grouped_blocks = []
        for code_block in code_blocks:
            file_name = code_block.get("file_name", "unknown_file")
            if file_name not in file_names:
                file_names.append(file_name)
                grouped_blocks.append({
                    "file_name": file_name, 
                    "url": code_block.get("url", "unknown_url"), 
                    "code_blocks": []})

        for code_block in code_blocks:
            for group in grouped_blocks:
                if group["file_name"] == code_block.get("file_name", "unknown_file"):
                    group["code_blocks"].append(code_block)
                    break

        # Sort blocks in each group by chunk number
        for group in grouped_blocks:
            group["code_blocks"].sort(key=lambda x: x.get("chunk_number", 0))

        # Show response
        st.markdown(f"{get_response_text(response, "code_response", "Missing code response")}", unsafe_allow_html=True)
        st.markdown("*View relevant code samples below*")
        for group in grouped_blocks:
            st.markdown(f"**{group['file_name']}** [:material/launch:]({group["url"]})")

            for code_block in group["code_blocks"]:
                with st.expander(f":material/code: {code_block["title"]}", expanded=False):
                    st.markdown(f"{code_block["original"]}")

with st.container():
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

            # Load and display knowledge graph from file if it exists
            try:
                with open("knowledge_graph.json", "r") as f:
                    kg_data = json.load(f)
                adjmat, nodes = process_kg(kg_data)
                if adjmat is not None and nodes is not None:
                    st.markdown("**Knowledge Graph:**")
                    d3.graph(adjmat)

                    color_scale = ColorScale()

                    # Setting per-node properties in d3.set_node_properties is not working, so do per node
                    for node_id in nodes.index:                        
                        d3.node_properties[node_id]['label'] = ""
                        d3.node_properties[node_id]['color'] = color_scale.get_color(nodes.at[node_id, 'category'])
                        d3.node_properties[node_id]['opacity'] = 1
                        d3.node_properties[node_id]['tooltip'] = nodes.at[node_id, 'tooltip']

                    d3.show(show_slider=False, save_button=False)
            except FileNotFoundError:
                pass

if prompt := (st.chat_input("Ask a question") or st.session_state["sample_prompt_button_pressed"]):   
    for history in st.session_state["history"]:        
        with st.container():
            with st.chat_message("using-bdc"):
                st.empty()
                display_input(history.get("input", ""))
            with st.chat_message("bdc-assistant"):
                # Need empty to avoid stale greyed out elements with spinner
                st.empty()
                display_response(history.get("response", {}))
    
    with st.container():
        with st.chat_message("using-bdc"):
            display_input(prompt)

        with st.chat_message("bdc-assistant"):
            # Add spinner while thinking
            with st.spinner("Generating response...", show_time=show_timer):            
                # Get response from server
                response = current_chain.invoke({"input": prompt, "chat_history": st.session_state["chat_history"]})
                
            display_response(response)
        
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