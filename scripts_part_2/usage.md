Use LLM_embeddings.py to generate embeddings on documents. The difference between this script and the original is that it uses LLM based semantic chunking rather than a constant char count. The script makes an openapi call to do the chunking.

Command:

python scripts_part_2/LLM_embeddings.py --file-dir <documentsDirectory> --out-dir <outDirectory>