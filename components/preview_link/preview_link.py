import os

import streamlit as st
import streamlit.components.v1 as components

def preview_link(url, text, doc_type):
    # Render the frontend HTML, injecting variables via URL params
    frontend_path = os.path.join(os.getcwd(), "components", "preview_link", "frontend", "index.html")
    
    # Read HTML template
    with open(frontend_path, "r") as f:
        component_html = f.read()
    
    # Replace placeholders with actual values (URL params)
    component_html = component_html.replace("url=", f"url={url}")
    component_html = component_html.replace("text=", f"text={text}")
    component_html = component_html.replace("doc_type=", f"doc_type={doc_type}")

    # Display the component in Streamlit
    components.html(component_html, height=300)
