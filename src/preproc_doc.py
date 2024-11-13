
from pathlib import Path
from tqdm import tqdm
import pickle

from preproc.proc_BDC_repo import get_fellow_files, get_data_mdx_files, clean_mdx, get_all_mdx_paths, clean_path


# process data dir

data_dir = "../interim-bdc-website/src/data/"
pages_dir = '../interim-bdc-website/src/pages/'

save_dir = "./data/"


# region: process fellows
print("Processing fellows...")
fellow_dir = data_dir + "fellows/"

fellows = get_fellow_files(Path(fellow_dir).resolve(), data_dir)

for fellow in fellows:
    fellow['metadata']['project_title'] = fellow['metadata']['project']['title']
    fellow['metadata']['project_abstract'] = fellow['metadata']['project']['abstract']
    del fellow['metadata']['project']





# print(f"Fellows: {len(fellows)}")

# process latest-updates
print("Processing latest-updates...")
updates_dir = data_dir + "latest-updates/"

updates = get_data_mdx_files(Path(updates_dir).resolve(), data_dir)

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

events = get_data_mdx_files(Path(events_dir).resolve(), data_dir)


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


mdx_paths = get_all_mdx_paths(pages_dir, page_dir_paths, page_file_paths)


metadata_list = []
page_content_list = []

for path in tqdm(mdx_paths):
    header, page_content = clean_mdx(path)
    header['file_path'] = clean_path(path, pages_dir)
    metadata_list.append(header)
    page_content_list.append(page_content)

# Save pages data
pages_data = []
for metadata, content in tqdm(zip(metadata_list, page_content_list)):
    if 'menu' in metadata.keys():
        metadata['headings'] = ', '.join([pair['heading'] for pair in metadata['menu']])
        metadata['hrefs'] = ', '.join([pair['href'] for pair in metadata['menu']])
        del metadata['menu']
    
    
    pages_data.append({'metadata': metadata, 'content': content})
    
    
    
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



