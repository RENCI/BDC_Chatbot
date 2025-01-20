import streamlit.components.v1 as components

def preview_link(url, text, doc_type):
    # load HTML
    with open('components/preview_link/frontend/index.html', 'r') as f:
        html_content = f.read()
    
    # inject the URL and text into our HTML
    html_content = html_content.replace("{{url}}", url).replace("{{text}}", text).replace("{{doc_type}}", doc_type)
    
    # render
    components.html(html_content)
