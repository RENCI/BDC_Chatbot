import re
import json
import os
from tqdm import tqdm

from pprint import pprint  

from pathlib import Path

import pickle

import chromadb
import uuid

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# from utils.preproc.proc_BDC_docs import chunk_docs_md_by_headers
from utils.preproc.utils import contextualize_chunk

from utils.rag.chain import get_summary
from utils import set_emb_llm
emb, llm, guardian_llm, dugbot_chain, DB_PATH = set_emb_llm()


code_root = "../Access-to-Data-using-PIC-SURE-API/NHLBI_BioData_Catalyst"
code_lang = "python"
root_dir = "./data/code_doc/"

def get_ipynb_files(root_dir):
    ipynb_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.ipynb'):
                full_path = os.path.join(dirpath, filename)
                ipynb_files.append(full_path)
    return ipynb_files

# append lang to root_dir


ipynb_file_paths = get_ipynb_files(Path(code_root)/code_lang)
# ipynb_file_paths = get_ipynb_files(f"{code_root}/{code_lang}")

print(f"number of ipynb files: {len(ipynb_file_paths)}")

# print(ipynb_file_paths[1])
# with open(ipynb_file_paths[1], 'r') as f:
#     ipynb_data = json.load(f)

# pprint(ipynb_data)

def ipynb_to_markdown(ipynb_data, code_lang="python"):
    md_lines = []
    for cell in ipynb_data.get("cells", []):
        if cell.get("cell_type") == "markdown":
            md_lines.extend(cell.get("source", []))
            md_lines.append("\n\n")
        elif cell.get("cell_type") == "code":
            md_lines.append(f"```{code_lang}\n")
            md_lines.extend(cell.get("source", []))
            if not md_lines[-1].endswith("\n"):
                md_lines[-1] += "\n"
            md_lines.append("```\n\n")
    return "".join(md_lines)

# md_content = ipynb_to_markdown(ipynb_data, code_lang)

def load_ipynb_as_md(file_path, code_lang="python"):
    with open(file_path, 'r') as f:
        ipynb_data = json.load(f)
    return ipynb_to_markdown(ipynb_data, code_lang)

def chunk_docs_md_by_headers(file_path, code_lang="python"):
    content = load_ipynb_as_md(file_path, code_lang)
    file_name = os.path.basename(file_path)
    
    header_pattern = r'^(#{1,6})\s+(.+)$'
    chunks_content = []
    chunks_metadata = []
    current_chunk = []
    current_headers = []
    in_code_block = False
    
    code_sample = ""
    
    
    for line in content.split('\n'):
        # Toggle code block state
        
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            current_chunk.append(line)
            continue

        if not in_code_block:
            header_match = re.match(header_pattern, line)
            if header_match:
                
                # Save previous chunk if it exists
                if current_chunk:
                    chunk_content = '\n'.join(current_chunk).strip()
                    if chunk_content:
                        chunks_content.append(chunk_content)
                        chunks_metadata.append({
                            "source": os.path.relpath(file_path),
                            "file_name": file_name,  
                            "hierarchy": ", ".join(current_headers),
                            "title": current_headers[-1]
                        })
                # Start new chunk (without including the header line)
                current_chunk = []
                header_level = len(header_match.group(1))
                header_text = header_match.group(2)
                current_headers = current_headers[:header_level-1]
                current_headers.append(header_text)
                continue  # Don't add header line to chunk
        current_chunk.append(line)
    
    # last chunk
    if current_chunk:
        chunk_content = '\n'.join(current_chunk).strip()
        if chunk_content:
            chunks_content.append(chunk_content)
            chunks_metadata.append({
                "source": os.path.relpath(file_path),
                "file_name": file_name,  
                "hierarchy": ", ".join(current_headers),
                "title": current_headers[-1]
            })
    
    return chunks_metadata, chunks_content, content

def ipynb_to_hierarchy_json(file_path, code_lang="python"):
    md_content = load_ipynb_as_md(file_path, code_lang)
    
    header_pattern = re.compile(r'^(#{1,6})\s+(.+)$')
    lines = md_content.split('\n')
    root = {"cell": {"file": os.path.basename(file_path), 
                     "language": code_lang, 
                     "children": []}
            }
    
    stack = [(0, root["cell"])]  # (header_level, cell_dict)
    current_content = []
    in_code_block = False

    def flush_content(cell):
        content = '\n'.join(current_content).strip()
        if content:
            if cell["content"]:
                cell["content"] += '\n' + content
            else:
                cell["content"] = content

    for line in lines:
        # Toggle code block state
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            current_content.append(line)
            continue

        if not in_code_block:
            header_match = header_pattern.match(line)
            if header_match:
                # New header found, flush content to current cell
                flush_content(stack[-1][1])
                current_content = []

                header_level = len(header_match.group(1))
                header_text = header_match.group(2)

                # Pop stack to the parent of this header level
                while stack and stack[-1][0] >= header_level:
                    flush_content(stack[-1][1])
                    stack.pop()

                # Create new cell
                new_cell = {"title": header_text, "content": "", "children": []}
                stack[-1][1]["children"].append({"cell": new_cell})
                stack.append((header_level, new_cell))
                continue  # Don't add header line to content

        current_content.append(line)

    # Flush any remaining content
    flush_content(stack[-1][1])

    return root


all_contextualized_chunks = []
all_chunks_content = []
all_chunks_metadata = []
all_chunks_context = []





# for file_path in tqdm(ipynb_file_paths, total=len(ipynb_file_paths)):
for j, file_path in enumerate(ipynb_file_paths):
    chunks_metadata, chunks_content, whole_document = chunk_docs_md_by_headers(file_path)
    # save doc
    os.makedirs(root_dir, exist_ok=True)
    with open(f"{root_dir}{os.path.splitext(os.path.basename(file_path))[0]}.md", 'w') as f:
        f.write(whole_document)
    
    for i, (chunk_content, chunk_metadata) in enumerate(zip(chunks_content, chunks_metadata)):
        print(f"{j}-{i}: chunk_content length: {len(chunk_content)}, whole_document length: {len(whole_document)}")
        chunk_context = contextualize_chunk(llm, chunk_content, whole_document, return_context_only=True)
        contextualized_chunk = f"{chunk_context}\n\n{chunk_content}"
        all_contextualized_chunks.append(contextualized_chunk)
        all_chunks_content.append(chunk_content)
        all_chunks_metadata.append(chunk_metadata)
        all_chunks_context.append(chunk_context)



summary_prompt = ChatPromptTemplate.from_messages(
    [("system", "You are a text‐rewriting assistant. \n– Your task is to:\n1. Rephrase the entire passage in clear, fluent prose.\n2. Remove any literal code blocks or fenced code snippets.\n3. For each removed code block, replace it with a concise natural‐language description that conveys the same logic, intent, parameters, and behavior of that code, so that no semantic information is lost.\n4. Preserve the overall structure, meaning, and emphasis of the original text.\n\nOutput only the rewritten text with no additional commentary. \n\nUser input: \n{text}")]
)


code_doc_rephrase_chain = summary_prompt | llm

all_rephrased_chunks = []

# remove code block from all_chunks_content, add chunk content to metadata
# add code sample to metadata
for i, chunk_content in enumerate(all_chunks_content):
    all_chunks_metadata[i]["original"] = chunk_content
    all_chunks_content[i] = re.sub(r'```.*?```', '</code block>', all_chunks_content[2], flags=re.DOTALL)
    # use regex to find code samples (can be multiple), concat with new line
    all_chunks_metadata[i]["code_sample"] = "\n".join(re.findall(r'```.*?```', chunk_content, flags=re.DOTALL))
    # all_rephrased_chunks.append(f"{all_chunks_context[i]}\n\n{chunk_content}")
    all_rephrased_chunks.append(all_chunks_content[i])

# save all_chunks_content and all_chunks_metadata to pkl, as list of dicts
data = [{"content": content, "metadata": metadata} for content, metadata in zip(all_rephrased_chunks, all_chunks_metadata)]

with open(f"{root_dir}code.pkl", 'wb') as f:
    pickle.dump(data, f)



all_ids = [str(uuid.uuid4()) for _ in range(len(all_chunks_content))]
print("start embedding...")
all_embeddings = emb.embed_documents(all_rephrased_chunks)


print("start saving to chroma...")
persistent_client = chromadb.PersistentClient(path="./.chroma_db_code_doc/") # DB_PATH_CODE_DOC
collection = persistent_client.get_or_create_collection("code_doc")
collection.add(ids=all_ids, 
               documents=all_rephrased_chunks, 
               embeddings=all_embeddings, 
               metadatas=all_chunks_metadata)