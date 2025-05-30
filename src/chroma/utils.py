import os

import chromadb
import uuid
import pickle




from ..rag.chain import get_summary

from tqdm import tqdm



def loadPKL(file_path, doc_type, emb, llm = None, use_summary = False, use_contextualized_chunk = True):
    
    # TODO: refactor for MultiVectorRetriever
    
    with open(file_path, 'rb') as f:
        pkl_list = pickle.load(f)
    
    page_contents = [row['content'] for row in pkl_list]
    
    metadata = [dict(row['metadata'], doc_type=doc_type) for row in pkl_list]
    
    
    text_to_embed = []
    for i, (page_content, meta) in enumerate(zip(page_contents, metadata)):
        if use_contextualized_chunk and 'contextualized_chunk' in meta:
            text_to_embed.append(meta['contextualized_chunk'])
        elif use_summary:
            summary = get_summary(page_content, llm)
            metadata[i] = dict(meta, summary=summary)
            text_to_embed.append(summary)
        elif 'text_to_embed' in meta:
            text_to_embed.append(meta['text_to_embed'])
        else:
            text_to_embed.append(page_content)
    
    # if use_summary:
    #     summaries = [get_summary(page_content, llm) for page_content in tqdm(page_contents, total=len(page_contents))]
    #     metadata = [dict(metadata, summary=summary) for metadata, summary in zip(metadata, summaries)]
    # else:
    #     summaries = page_contents
        
    embeddings = emb.embed_documents(text_to_embed)

    return page_contents, metadata, embeddings
    



def create_chroma_client(docs_path, db_path = ".chroma_db/", emb = None, llm = None, use_summary = False, file_name_list = None, doc_type_list = None):
    # TODO: refactor for MultiVectorRetriever
    
    all_contents = []
    all_metadatas = []
    all_embeddings = []
    
    for file_name, doc_type in zip(file_name_list, doc_type_list):
        print("Loading ", doc_type)
        page_contents, metadata, embeddings = loadPKL(os.path.join(docs_path, file_name), doc_type, emb, llm, use_summary)
        all_contents.extend(page_contents)
        all_metadatas.extend(metadata)
        all_embeddings.extend(embeddings)
    
    for metadata in all_metadatas:
        if len(metadata) == 0:
            metadata = None
    
    all_ids = [str(uuid.uuid4()) for _ in range(len(all_contents))]
    
    persistent_client = chromadb.PersistentClient(path=db_path)
    collection = persistent_client.get_or_create_collection("langchain")
    collection.add(ids=all_ids, documents=all_contents, embeddings=all_embeddings, metadatas=all_metadatas)
    
    
    return persistent_client





