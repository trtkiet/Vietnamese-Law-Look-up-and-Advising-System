# Data Preprocessing for Vietnamese Law RAG

This directory contains scripts and modules for processing Vietnamese legal documents for the RAG system.

## Components

### `chunker.py`
The core logic for parsing and chunking Vietnamese legal documents.
- **Hierarchical Parsing**: Detects `Phần` (Part), `Chương` (Chapter), `Mục` (Section), and `Điều` (Article).
- **Article-based Chunking**: Treats each Article (`Điều`) as a logical unit.
- **Smart Fallback**: If an article exceeds the token limit (default 512), it uses `RecursiveCharacterTextSplitter`.
- **Context Awareness**: Prepends hierarchy information (e.g., `[Chương I | Điều 5]`) to the chunk content for better retrieval context.

### `chunk_pipeline.py`
A CLI utility to run the chunking process on the entire dataset.

## Usage

### Command Line Interface

Run the pipeline from the project root:

```bash
# Process all documents (dry run - just prints stats)
python -m preprocess.chunk_pipeline --dry-run

# Process and save to JSON
python -m preprocess.chunk_pipeline --output data/processed_chunks.json

# Customize parameters
python -m preprocess.chunk_pipeline \
    --docs-root ./law_crawler/vbpl_documents \
    --output data/chunks_1024.json \
    --max-tokens 1024 \
    --overlap 100 \
    --no-context-header
```

### Programmatic Usage

You can import `VietLegalChunker` in your own scripts:

```python
from preprocess.chunker import VietLegalChunker, ChunkerConfig

# Initialize
config = ChunkerConfig(max_tokens=512)
chunker = VietLegalChunker(config=config)

# Chunk a single file
docs = chunker.chunk_json_file("path/to/document.json")

# Chunk a directory
all_docs = chunker.chunk_directory("path/to/docs_root")

# Access metadata
for doc in docs:
    print(doc.metadata['document_title'])
    print(doc.metadata['dieu'])
```

## Metadata Fields

Each chunk contains rich metadata:
- `id`: Unique UUID for the chunk
- `document_id`: Original document ID
- `document_type`: Type of document (e.g., "Luật", "Nghị định")
- `document_title`: Title extracted from content or filename
- `phan`: Part title (if applicable)
- `chuong`: Chapter title (if applicable)
- `muc`: Section title (if applicable)
- `dieu`: Article title
- `chunk_split_index`: Index if an article was split into multiple chunks (0, 1, 2...)
