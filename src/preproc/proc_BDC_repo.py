import re
import yaml
import json

import glob
import os
from datetime import datetime
    
from tqdm import tqdm  

from urllib.parse import quote

# from pprint import pprint   


def clean_path(path_str, root_dir):
    cleaned_root = re.sub(r'(?:\.{2}/)*', '', root_dir)
    
    pattern = f".*?{re.escape(cleaned_root)}"
    cleaned_path = re.sub(pattern, "", path_str)
    
    return cleaned_root + cleaned_path.lstrip("/")




def parse_fellow_files(file_path, root_dir):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            # Remove YAML delimiters
            yaml_content = content.replace('---\n', '', 1) 
            yaml_content = yaml_content.rsplit('---', 1)[0]
            
            fellow_data = yaml.safe_load(yaml_content)
            fellow_data["file_path"] = clean_path(file_path, root_dir)
            # remove the first part of the path
            fellow_data["relative_file_path"] = "/".join(fellow_data["file_path"].split("/")[1:])
            
            return fellow_data
    except yaml.YAMLError as e:
        print(f"Error parsing {file_path}: {e}")
        return None

# https://biodatacatalyst.nhlbi.nih.gov/about/bdc-fellows/#:~:text=from%20that%20time.-,Alexander%20Bick%2C%20MD%2C%20PhD,-Harrison%20Brand%2C%20PhD
# https://biodatacatalyst.nhlbi.nih.gov/about/bdc-fellows/#:~:text=Sheila%20Gaynor%2C%20PhD-,Sarah%20Gerard%2C%20PhD,-Steven%20Gilhool%2C%20PhD
# https://biodatacatalyst.nhlbi.nih.gov/about/bdc-fellows/#:~:text=Xuefang%20Zhao%2C%20PhD-,Yonghua%20Zhuang%2C%20PhD,-Since%20the%20conclusion


def get_fellow_files(fellow_dir, root_dir, base_url = None, remote_file_dir = None):
    fellows = []
    # Get all .md files in the directory
    md_files = glob.glob(os.path.join(fellow_dir, "*.md"))
    
    for file_path in tqdm(md_files, desc="Reading fellow files"):
        fellow_data = parse_fellow_files(file_path, root_dir)
        if fellow_data:
            if remote_file_dir and 'relative_file_path' in fellow_data:
                fellow_data['remote_file_path'] = remote_file_dir + fellow_data['relative_file_path']
            if base_url and 'name' in fellow_data:
                fellow_data['page_url'] = base_url + "about/bdc-fellows/#:~:text=" + quote(fellow_data['name'])
                
            fellows.append({"metadata": fellow_data, "content": json.dumps(fellow_data, indent=4)})
                
    return fellows


def parse_simple_mdx_files(file_path, root_dir):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
        # Split content into YAML and markdown 
        parts = content.split('---\n', 2)
        if len(parts) < 3:
            raise ValueError("File does not contain valid YAML frontmatter")
            
        # YAML
        metadata = yaml.safe_load(parts[1])
        
        # markdown
        markdown_content = parts[2].strip()
        
        metadata['file_path'] = clean_path(file_path, root_dir)
        metadata["relative_file_path"] = "/".join(metadata["file_path"].split("/")[1:])
        # print(metadata['file_path'])
        
        return metadata, markdown_content
        
    except (yaml.YAMLError, ValueError) as e:
        print(f"Error parsing {file_path}: {e}")
        return None, None


def get_data_mdx_files(updates_dir, root_dir, base_url = None, remote_file_dir = None):
    """Get all MDX files directly under the directory and index.mdx from subdirectories.
    """
    # paths = []
    # metadata_list = []
    # content_list = []
    res = []
    

    direct_mdx = glob.glob(os.path.join(updates_dir, "*.mdx"))
    subdir_mdx = glob.glob(os.path.join(updates_dir, "**/index.mdx"))
    all_mdx_paths = direct_mdx + subdir_mdx
    
    for file_path in tqdm(all_mdx_paths, desc="Parsing MDX files"):
        metadata, content = parse_simple_mdx_files(file_path, root_dir)
        if metadata and content:  # Only add if parsing was successful
            # paths.append(file_path)
            if base_url and 'path' in metadata:
                # remove the leading /
                metadata['page_url'] = base_url + metadata['path'].lstrip('/')
            if remote_file_dir and 'relative_file_path' in metadata:
                metadata['remote_file_path'] = remote_file_dir + metadata['relative_file_path']
            
            
            res.append({"metadata": metadata, "content": content})

    return res




def clean_mdx(file_path):
    """Clean MDX file of a BDC web page"""
    tags_to_remove = ["ButtonContainer", "NextStepsCard"]

    with open(file_path, 'r', encoding="utf8") as file:
        content = file.read()

    # Extract the YAML header/front matter
    yaml_header = re.search(r'^---\n(.*?)\n---', content, re.DOTALL)
    if yaml_header:
        yaml_content = yaml_header.group(1)
        header_dict = yaml.safe_load(yaml_content)
    else:
        header_dict = {}

    content = re.sub(r'^---\n.*?\n---', '', content, flags=re.DOTALL)

    # Extract PageContent
    page_content_match = re.search(r'<PageContent.*?>(.*?)</PageContent>', content, re.DOTALL)
    if page_content_match:
        page_content = page_content_match.group(1)
    else:
        page_content = ""

    # Extract FloatingContentWrapper
    floating_content_match = re.search(r'<FloatingContentWrapper.*?>(.*?)</FloatingContentWrapper>', page_content, flags=re.DOTALL)
    floating_content = floating_content_match.group(1) if floating_content_match else ""

    # Remove FloatingContentWrapper from original position
    page_content = re.sub(r'<FloatingContentWrapper.*?</FloatingContentWrapper>', '', page_content, flags=re.DOTALL)

    # Remove tags_to_remove
    for tag in tags_to_remove:
        page_content = re.sub(f'<{tag}.*?>.*?</{tag}>', '', page_content, flags=re.DOTALL)

    # <Link> tags to markdown links
    page_content = re.sub(r'<Link to="([^"]*)"[^>]*>(.*?)</Link>', r'[\2](\1)', page_content)

    # Remove all remaining JSX/JS parts
    cleaned_content = re.sub(r'<.*?>', '', page_content, flags=re.DOTALL)

    # Insert FloatingContentWrapper content after the first h2 section
    sections = re.split(r'\n##\s', cleaned_content)
    insert_index = 2
    if len(sections) > 1:
        sections[insert_index] = sections[insert_index] + "\n\n" + floating_content.strip() + "\n\n"
    cleaned_content = '## '.join(sections)

    return header_dict, cleaned_content.strip()



def get_all_mdx_paths(pages_dir, page_dir_paths, page_file_paths):
    all_paths = []
    relative_paths = []  # New list for relative paths
    
    # Process directory paths (get mdx files directly under it)
    for dir_path in page_dir_paths:
        full_dir_path = os.path.join(pages_dir, dir_path)
        if os.path.exists(full_dir_path):
            
            # Get immediate files in the directory
            for file in os.listdir(full_dir_path):
                if file.endswith('.mdx'):
                    full_path = os.path.join(full_dir_path, file)
                    relative_path = os.path.join(dir_path, file)  # Create relative path
                    all_paths.append(full_path)
                    relative_paths.append(relative_path)
    
    # Process individual file paths
    for file_path in page_file_paths:
        full_file_path = os.path.join(pages_dir, file_path)
        if os.path.exists(full_file_path):
            all_paths.append(full_file_path)
            relative_paths.append(file_path)  # Add original file path as relative path
    
    return all_paths, relative_paths









