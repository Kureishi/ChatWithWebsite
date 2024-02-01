Since application uses local LLM model (from GPT4All where n_ctx set), the number of tokens may exceed the allowed context window when querying. Note the embeddings model used is from HuggingFace.

Different results may be obtained if another embeddings model and/or LLM are used (ex: OpenAI).