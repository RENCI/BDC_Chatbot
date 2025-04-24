
from pathlib import Path
from tqdm import tqdm
import pickle

from .preproc.utils import contextualize_chunk, paths_to_urls, split_by_sections

from .preproc.proc_BDC_repo import get_fellow_files, get_data_mdx_files, clean_mdx, get_all_mdx_paths, clean_path
from .preproc.proc_freshdesk import scrape_freshdesk
from .preproc.proc_BDC_docs import get_bdc_docs_md_files, chunk_docs_md_by_headers


from . import set_emb_llm
emb, llm, guardian_llm, dugbot_chain, DB_PATH = set_emb_llm()
# process data dir

base_url = "https://biodatacatalyst.nhlbi.nih.gov/"
github_base_url = "https://github.com/stagecc/interim-bdc-website/tree/main/"

data_dir = "../interim-bdc-website/src/data/"
pages_dir = '../interim-bdc-website/src/pages/'

save_dir = "./data/"


# region: process fellows
print("Processing fellows...")
fellow_dir = data_dir + "fellows/"

fellows = get_fellow_files(Path(fellow_dir).resolve(), data_dir, base_url=base_url, remote_file_dir=github_base_url)

for fellow in fellows:
    fellow['metadata']['project_title'] = fellow['metadata']['project']['title']
    fellow['metadata']['project_abstract'] = fellow['metadata']['project']['abstract']
    del fellow['metadata']['project']





# print(f"Fellows: {len(fellows)}")

# process latest-updates
print("Processing latest-updates...")
updates_dir = data_dir + "latest-updates/"

updates = get_data_mdx_files(Path(updates_dir).resolve(), data_dir, base_url=base_url, remote_file_dir=github_base_url)

for update in updates:
    update['metadata']['date'] = update['metadata']['date'].strftime("%Y-%m-%d")
    update['metadata']['tags'] = ", ".join(update['metadata']['tags'])
    # del value if it is a dictionary or list of dictionaries
    del_keys = []
    for key, value in update['metadata'].items():
        if isinstance(value, dict):
            del_keys.append(key)
        elif isinstance(value, list):
            if isinstance(value[0], dict):
                del_keys.append(key)

    for key in del_keys:
        del update['metadata'][key]
    
    
    
# process events
print("Processing events...")
events_dir = data_dir + "events/"

events = get_data_mdx_files(Path(events_dir).resolve(), data_dir, base_url=base_url, remote_file_dir=github_base_url)


# convert event date to string
for event in events:
    event['metadata']['date'] = event['metadata']['date'].strftime("%Y-%m-%d")
    event['metadata']['tags'] = ", ".join(event['metadata']['tags'])
    
    del_keys = []
    for key, value in event['metadata'].items():
        if isinstance(value, dict):
            del_keys.append(key)
        elif isinstance(value, list):
            if isinstance(value[0], dict):
                del_keys.append(key)
    for key in del_keys:
        del event['metadata'][key]
    
    
    
# endregion


# region: process pages dir


print("Processing pages...")
page_dir_paths = ["use-bdc/analyze-data/", ]
page_file_paths = ["join-bdc/index.mdx", "use-bdc/share-data.mdx", 
                   "user-resources/terms-of-use.mdx", "user-resources/usage-costs.mdx", "user-resources/usage-terms.mdx",
                   "about/key-collaborations.mdx", "about/overview.mdx", "about/research-communities.mdx",
                   "use-bdc/explore-data/index.mdx"]


mdx_paths, relative_paths = get_all_mdx_paths(pages_dir, page_dir_paths, page_file_paths)

pages_url = paths_to_urls(base_url, relative_paths)


metadata_list = []
page_content_list = []
contextualized_chunk_list = []

# print(len(mdx_paths), mdx_paths[0])
for i, path in tqdm(enumerate(mdx_paths), desc="Processing pages", total=len(mdx_paths)):
    header, page_content = clean_mdx(path)
    header['file_path'] = clean_path(path, pages_dir)
    header['page_url'] = pages_url[i]
    
    # metadata_list.append(header)
    # page_content_list.append(page_content)

    section_list = split_by_sections(page_content)
    
    for section in section_list:
        contextualized_chunk = contextualize_chunk(llm, section, whole_document=page_content)
        contextualized_chunk_list.append(contextualized_chunk)
    
    
    metadata_list.extend([header]*len(section_list))
    page_content_list.extend(section_list)

for i in range(len(contextualized_chunk_list)):
    metadata_list[i]['contextualized_chunk'] = contextualized_chunk_list[i]




# Save pages data
pages_data = []
for metadata, content in tqdm(zip(metadata_list, page_content_list), desc="Processing pages (sections)", total=len(page_content_list)):
    if 'menu' in metadata.keys():
        metadata['headings'] = ', '.join([pair['heading'] for pair in metadata['menu']])
        metadata['hrefs'] = ', '.join([pair['href'] for pair in metadata['menu']])
        del metadata['menu']
    
    
    pages_data.append({'metadata': metadata, 'content': content})
    
    
    
# endregion



# region freshdesk FAQ
print("Processing freshdesk FAQ...")

freshdesk_base_url = "https://bdcatalyst.freshdesk.com"
faq_category_dict = {
    "BDC FAQs": "60000157358",
    "BDC Fellows Program FAQs": "60000294708",
}

content_list, metadata_list = scrape_freshdesk(faq_category_dict, freshdesk_base_url)



# TODO: MultiVectorRetriever
# new_metadata_list = metadata_list.copy()

metadata_keys = ['category', 'folder', 'title']
# for i in range(len(content_list)):
#     metadata_list[i]['contextualized_chunk'] = contextualize_chunk(llm, content_list[i], metadata_context={key: metadata_list[i][key] for key in metadata_keys})
#     # metadata_list[i]['text_to_embed'] = metadata_list[i]['title'] + '\n' + content_list[i]
    
# for i in range(len(new_metadata_list)):
#     new_metadata_list[i]['text_to_embed'] = metadata_list[i]['title']


faqs_data = []
for metadata, content in tqdm(zip(metadata_list, content_list), desc="Processing freshdesk FAQ", total=len(content_list)):
    metadata['contextualized_chunk'] = contextualize_chunk(llm, content, metadata_context={key: metadata[key] for key in metadata_keys})
    faqs_data.append({'metadata': metadata, 'content': content})



# endregion

# region: process docs
print("Processing docs...")
docs_data = []


md_file_paths = get_bdc_docs_md_files()


metadata_list = []
content_list = []
for file_path in tqdm(md_file_paths):
    chunks_metadata, chunks_content = chunk_docs_md_by_headers(file_path)
    metadata_list.extend(chunks_metadata)
    content_list.extend(chunks_content)


# contextualize_chunk, add remote_file_path
contextualized_chunk_list = []
for i, chunk in tqdm(enumerate(content_list), desc="Contextualizing chunks", total=len(content_list)):
    contextualized_chunk_list.append(contextualize_chunk(llm, chunk, whole_document=metadata_list[i]['whole_document']))
    metadata_list[i]['contextualized_chunk'] = contextualized_chunk_list[i]
    
    
    # TODO: don't hardcode the url, don't use github link
    metadata_list[i]['remote_file_path'] = "https://github.com/stagecc/bdc-docs/tree/main/" + metadata_list[i]['source'][12:]
    
    
    docs_data.append({'metadata': metadata_list[i], 'content': chunk})

# endregion






# save all content and metadata in pkl
with open(save_dir+'/fellows.pkl', 'wb') as f:
    pickle.dump(fellows, f)

with open(save_dir+'/latest_updates.pkl', 'wb') as f:
    pickle.dump(updates, f)

with open(save_dir+'/events.pkl', 'wb') as f:
    pickle.dump(events, f)

with open(save_dir+'/pages.pkl', 'wb') as f:
    pickle.dump(pages_data, f)

with open(save_dir+'/freshdesk.pkl', 'wb') as f:
    pickle.dump(faqs_data, f)

with open(save_dir+'/docs.pkl', 'wb') as f:
    pickle.dump(docs_data, f)
