from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
    TokenTextSplitter
)
from llama_index.core import Document
from pathlib import Path
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
import os
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.tools import QueryEngineTool
from llama_index.core import StorageContext


embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)
Settings.embed_model = embed_model


def read_docs_from_dir(input_dir: Path):
    md_files = input_dir.glob("**/*.md")
    return md_files


def build_vector_index(input_dir: str, persist_dir: str):
    index_docs = []
    for doc in read_docs_from_dir(Path(input_dir)):
        with open(doc) as f:
            index_doc = Document(
                text=f.read(),
                metadata={"file_path": str(doc)}
            )
            index_docs.append(index_doc)

    pipeline = IngestionPipeline(
        name="document_parsing_pipeline",
        project_name="krisp_hackathon",
        transformations=[
            SemanticSplitterNodeParser(
                buffer_size=4,
                breakpoint_percentile_threshold=75,
                embed_model=embed_model
            )
        ]
    )
    nodes = pipeline.run(documents=index_docs)
    vector_index = VectorStoreIndex(nodes=nodes, use_async=True)
    vector_index.storage_context.persist(persist_dir)
    return vector_index


if __name__ == "__main__":
    # build_vector_index(input_dir="./data/parsed", persist_dir="app/storage/documents_index")
    build_vector_index(input_dir="./data/summaries", persist_dir="app/storage/financial_knowledge_base_index")