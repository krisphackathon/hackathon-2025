import asyncio
from pathlib import Path
from llama_cloud_services import LlamaParse
from llama_cloud_services.parse.types import JobResult
import shutil
import os

async def parse_and_save(input_dir: Path, output_dir: Path):
    """
    Parses all PDFs in a given directory and saves the results to an output directory.
    """
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    parser = LlamaParse(
        api_key=os.getenv("LLAMAPARSE_API_KEY"),
        page_separator="\n\n---\n\n",
        model="openai-gpt-4-1",
        parse_mode="parse_page_with_agent",
        high_res_ocr=True,
        adaptive_long_table=True,
        outlined_table_extraction=True,
        output_tables_as_HTML=True,
    )

    pdf_files = list(input_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return

    print(f"Parsing {len(pdf_files)} PDF files...")
    results: list[JobResult] = await parser.aparse(pdf_files) # type: ignore

    for result in results:
        original_filename = Path(result.file_name).stem
        markdown_documents = result.get_markdown_documents(split_by_page=True)
        for i, doc in enumerate(markdown_documents):
            md_path = output_dir / f"{original_filename}_page_{i+1}.md"
            md_path.write_text(doc.text)
            print(f"Saved Markdown for {original_filename} (page {i+1}) to {md_path}")

if __name__ == "__main__":
    input_directory = Path("./kb")
    output_directory = Path("./data/parsed")
    
    print(f"Ensure your PDF files are located in the '{input_directory}' directory.")
    asyncio.run(parse_and_save(input_directory, output_directory))