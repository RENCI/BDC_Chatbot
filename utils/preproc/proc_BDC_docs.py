import re
import yaml
import json
from tqdm import tqdm  

import os

from utils.preproc.utils import contextualize_chunk
from utils import set_emb_llm
emb, llm, guardian_llm, dugbot_chain, DB_PATH = set_emb_llm()


def get_bdc_docs_md_files(root_dir="../bdc-docs/docs/docs"):
    mdx_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.md') and filename not in ['index.md', 'glossary.md']:
                full_path = os.path.join(dirpath, filename)
                mdx_files.append(full_path)
    return mdx_files


def load_docs_md(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    content = re.sub(r'^---\n.*?\n---\n', '', content, flags=re.DOTALL)
    
    # Remove image references (e.g., ![][image1] or ![alt text](url))
    content = re.sub(r'!\[.*?\]\[.*?\]', '', content)  # Remove ![][image1] style
    content = re.sub(r'!\[.*?\]\(.*?\)', '', content)  # Remove ![](url) style
    
    # Remove image definitions (e.g., [image1]: url)
    content = re.sub(r'^\[.*?\]:\s*.*$', '', content, flags=re.MULTILINE)

    return content



def chunk_docs_md_by_headers(file_path):
    content = load_docs_md(file_path)
    file_name = os.path.basename(file_path)
    
    # Split content by headers
    header_pattern = r'^(#{1,6})\s+(.+)$'
    chunks_content = []
    chunks_metadata = []
    current_chunk = []
    current_headers = []
    
    for line in content.split('\n'):
        header_match = re.match(header_pattern, line, re.MULTILINE)
        
        if header_match:
            # Save previous chunk if it exists
            if current_chunk:
                chunk_content = '\n'.join(current_chunk).strip()
                if chunk_content:  # Skip empty chunks
                    chunks_content.append(chunk_content)
                    chunks_metadata.append({
                        "source": os.path.relpath(file_path),
                        "file_name": file_name,  
                        "hierarchy": ", ".join(current_headers),
                        "whole_document": content
                    })
            
            # Start new chunk (without including the header line)
            current_chunk = []  # Changed: Don't include header in current_chunk
            header_level = len(header_match.group(1))
            header_text = header_match.group(2)
            
            # Update headers based on level
            current_headers = current_headers[:header_level-1]
            current_headers.append(header_text)
        else:
            current_chunk.append(line)
    
    # Don't forget the last chunk
    if current_chunk:
        chunk_content = '\n'.join(current_chunk).strip()
        if chunk_content:  # Skip empty chunks
            chunks_content.append(chunk_content)
            chunks_metadata.append({
                "source": os.path.relpath(file_path),
                "file_name": file_name,  
                "hierarchy": ", ".join(current_headers),
                "whole_document": content
            })
    
    return chunks_metadata, chunks_content









