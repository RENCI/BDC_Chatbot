import re
from ..rag.chain import create_chunk_contextualizer_chain



def concat_paths(*paths):
    result = paths[0].rstrip('/') if paths else ''

    for path in paths[1:]:
        path = path.lstrip('/')
        result = f"{result}/{path}"

    return result



def paths_to_urls(base_url, file_paths):
    urls = []
    for path in file_paths:
        
        clean_path = path.replace('.mdx', '')
        
        clean_path = re.sub(r'/index$', '', clean_path)
        
        full_url = f"{base_url.rstrip('/')}/{clean_path}"
        urls.append(full_url)
    
    return urls






def split_by_sections(text, return_dict = False):
    lines = text.split('\n')
    
    sections = []
    current_section = []
    current_header = None
    
    for line in lines:

        if line.startswith('##'):
            if current_section:
                sections.append({
                    'header': current_header,
                    'content': '\n'.join(current_section).strip()
                })
            # new section
            current_header = line.strip()
            current_section = []
        else:
            current_section.append(line)
    

    if current_section:
        sections.append({
            'header': current_header,
            'content': '\n'.join(current_section).strip()
        })
    
    if return_dict:
        return sections
    else:
        text_list = []
        for section in sections:
            header = section['header'] if section['header'] else ''
            content = section['content'] if section['content'] else ''
            text_list.append(header + '\n' + content)
        return text_list
        
        
        
        


def contextualize_chunk(llm, chunk_content: str, whole_document: str = None, metadata_context: str = None) -> str:
    
    if metadata_context is not None:
        context = metadata_context
        chain = create_chunk_contextualizer_chain(llm, use_metadata_context=True)
    else:
        context = whole_document
        chain = create_chunk_contextualizer_chain(llm, use_metadata_context=False)
    
    # chain = create_chunk_contextualizer_chain(llm, use_metadata_context=metadata_context is not None)
    
    context = chain.invoke({
        "context": context,
        "chunk_content": chunk_content
    })
    
    # Combine the context with the original chunk
    contextualized_chunk = f"{context}{chunk_content}"
    
    return contextualized_chunk