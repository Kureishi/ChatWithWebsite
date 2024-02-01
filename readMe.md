Since application uses local LLM model (from GPT4All where n_ctx set), the number of tokens may exceed the allowed context window when querying. Note the embeddings model used is from HuggingFace.

Different results may be obtained if another embeddings model and/or LLM are used (ex: OpenAI).

##### Notes:
- LLM: gpt4all-falcon-newbpe-q4_0.gguf
- Embeddings Model: hkunlp/instructor-xl
- Primary Library: Langchain 0.1.0