# BDC Chatbot (WIP)

Chatbot for Biodata Catalyst

## Install dependencies

```bash
pip install -r requirements.txt
```

## Local development

```bash
kubectl -n ner port-forward svc/vllm-llama-3-1-8b-instruct 8080:80  # vLLM port-forward

# kubectl -n ner port-forward svc/ollama 11434:11434                # Ollama port-forward
kubectl -n ner port-forward ollama-dd6845745-2txz8 11434:11434      


python ./src/preproc_doc.py                                         # preprocess BDC website repo
python -m src.prepare_chromadb                                      # create chroma db
streamlit run ./src/streamlit_app.py                                # run streamlit app
```

> [!WARNING]
>
> - To use ${\color{orange}\text{vLLM}}$ API for chat completion, remove `parallel_tool_calls=False` in `langchain_openai\chat_models\base.py`
> - Add `Comparator.IN, Comparator.NIN` in `langchain_community\query_constructors\chroma.py` under `allowed_comparators`

> [!IMPORTANT]
>
> - Chroma DB initialization might quit without error or warning, might be caused by compatibility issue with Windows.
> - Clone the [BDC website repo](https://github.com/stagecc/interim-bdc-website/tree/main) to the same directory as this repo, and run `python ./src/preproc_doc.py` to preprocess the website.
