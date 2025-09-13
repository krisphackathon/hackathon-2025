#!/usr/bin/env python3
"""Convert PDFs to Markdown using Unstructured.io for better structure preservation."""

import argparse
import sys
from pathlib import Path
import json
import re
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import (
    Title, NarrativeText, ListItem, Table, 
    FigureCaption, Header, Footer, PageBreak
)


class UnstructuredPDFConverter:
    """Convert PDFs to Markdown using Unstructured.io."""
    
    def __init__(self):
        """Initialize the converter."""
        self.doc_type_patterns = {
            "earnings_transcript": re.compile(r"earnings.*transcript|q\d.*earnings.*call", re.I),
            "earnings_release": re.compile(r"earnings.*release|financial.*results|quarterly.*results", re.I),
            "earnings_slides": re.compile(r"earnings.*slides|investor.*presentation", re.I),
            "product_doc": re.compile(r"product|feature|announcement|google.*photos|ai.*mode", re.I)
        }
    
    def detect_document_type(self, filename: str) -> str:
        """Detect document type from filename."""
        for doc_type, pattern in self.doc_type_patterns.items():
            if pattern.search(filename):
                return doc_type
        return "general"
    
    def extract_metadata_from_filename(self, filename: str) -> Dict[str, Any]:
        """Extract metadata from filename."""
        metadata = {
            "original_filename": filename,
            "document_type": self.detect_document_type(filename)
        }
        
        # Extract year
        year_patterns = [
            r"20(\d{2})",  # 2024, 2025, etc.
            r"(\d{4})"     # Full year
        ]
        for pattern in year_patterns:
            match = re.search(pattern, filename)
            if match:
                year = match.group(0)
                if len(year) == 2:
                    year = f"20{year}"
                metadata["year"] = int(year)
                break
        
        # Extract quarter
        quarter_match = re.search(r"[qQ](\d)", filename)
        if quarter_match:
            metadata["quarter"] = f"Q{quarter_match.group(1)}"
        
        # Extract company
        if "alphabet" in filename.lower():
            metadata["company"] = "Alphabet"
        elif "google" in filename.lower():
            metadata["company"] = "Google"
        
        # Generate title
        title = filename.replace(".pdf", "").replace("-", " ").replace("_", " ")
        # Capitalize appropriately
        title_parts = []
        for part in title.split():
            if part.upper() in ["Q1", "Q2", "Q3", "Q4", "AI", "PDF"]:
                title_parts.append(part.upper())
            elif part.lower() in ["to", "at", "the", "of", "in", "on", "with", "can", "do", "you"]:
                title_parts.append(part.lower())
            else:
                title_parts.append(part.capitalize())
        
        metadata["title"] = " ".join(title_parts)
        
        return metadata
    
    def elements_to_markdown(self, elements: List[Any], metadata: Dict[str, Any]) -> str:
        """Convert Unstructured elements to Markdown format."""
        markdown_lines = []
        
        # Add frontmatter
        markdown_lines.append("---")
        markdown_lines.append(f"title: {metadata.get('title', 'Untitled')}")
        markdown_lines.append(f"source_file: {metadata.get('original_filename', 'unknown')}")
        markdown_lines.append(f"document_type: {metadata.get('document_type', 'general')}")
        if metadata.get('company'):
            markdown_lines.append(f"company: {metadata['company']}")
        if metadata.get('year'):
            markdown_lines.append(f"year: {metadata['year']}")
        if metadata.get('quarter'):
            markdown_lines.append(f"quarter: {metadata['quarter']}")
        markdown_lines.append(f"converted_at: {datetime.utcnow().isoformat()}")
        markdown_lines.append("---")
        markdown_lines.append("")
        
        # Track current section for better structure
        current_page = None
        
        for element in elements:
            # Handle page breaks
            if isinstance(element, PageBreak):
                markdown_lines.append("\n---\n")
                continue
            
            # Skip headers and footers
            if isinstance(element, (Header, Footer)):
                continue
            
            # Get element text
            text = str(element).strip()
            if not text:
                continue
            
            # Add page number if available
            if hasattr(element, 'metadata') and element.metadata.get('page_number'):
                page_num = element.metadata['page_number']
                if page_num != current_page:
                    current_page = page_num
                    markdown_lines.append(f"\n<!-- Page {page_num} -->\n")
            
            # Format based on element type
            if isinstance(element, Title):
                # Determine heading level based on context
                level = element.metadata.get('heading_level', 1) if hasattr(element, 'metadata') else 1
                markdown_lines.append(f"{'#' * level} {text}")
                markdown_lines.append("")
            
            elif isinstance(element, Table):
                # Convert table to markdown
                markdown_lines.append(self._table_to_markdown(element))
                markdown_lines.append("")
            
            elif isinstance(element, ListItem):
                # Handle list items
                markdown_lines.append(f"- {text}")
            
            elif isinstance(element, FigureCaption):
                # Handle figure captions
                markdown_lines.append(f"*Figure: {text}*")
                markdown_lines.append("")
            
            else:
                # Regular text/narrative
                # Check if it's a speaker line in transcript
                if metadata.get('document_type') == 'earnings_transcript':
                    speaker_match = re.match(r"^([A-Z][a-zA-Z\s\.]+)(?:\s*[-–—:]|\s+[-–—])\s*(.*)$", text)
                    if speaker_match:
                        speaker = speaker_match.group(1).strip()
                        content = speaker_match.group(2).strip()
                        markdown_lines.append(f"\n**{speaker}**: {content}")
                    else:
                        markdown_lines.append(text)
                else:
                    markdown_lines.append(text)
                
                markdown_lines.append("")
        
        return "\n".join(markdown_lines)
    
    def _table_to_markdown(self, table_element) -> str:
        """Convert table element to markdown format."""
        try:
            # Extract table data
            if hasattr(table_element, 'text'):
                # Simple text representation
                lines = table_element.text.split('\n')
                if len(lines) > 1:
                    # Create simple markdown table
                    md_lines = []
                    md_lines.append("| " + " | ".join(lines[0].split()) + " |")
                    md_lines.append("|" + "---|" * len(lines[0].split()))
                    for line in lines[1:]:
                        if line.strip():
                            md_lines.append("| " + " | ".join(line.split()) + " |")
                    return "\n".join(md_lines)
            
            # Fallback to string representation
            return f"```\n{str(table_element)}\n```"
            
        except Exception as e:
            print(f"Warning: Could not format table: {e}")
            return f"```\n{str(table_element)}\n```"
    
    def convert_pdf_to_markdown(
        self, 
        pdf_path: Path, 
        output_dir: Path,
        strategy: str = "hi_res"
    ) -> Optional[Path]:
        """Convert a single PDF to Markdown."""
        try:
            print(f"📄 Processing: {pdf_path.name}")
            
            # Extract metadata from filename
            metadata = self.extract_metadata_from_filename(pdf_path.name)
            
            # Determine strategy based on document type
            if metadata['document_type'] == 'earnings_transcript':
                # High resolution for better speaker detection
                strategy = "hi_res"
            elif metadata['document_type'] == 'earnings_release':
                # High resolution for table extraction
                strategy = "hi_res"
            else:
                # Fast strategy for simple documents
                strategy = "fast"
            
            print(f"   Using strategy: {strategy}")
            
            # Partition PDF
            elements = partition_pdf(
                filename=str(pdf_path),
                strategy=strategy,
                extract_images_in_pdf=False,
                infer_table_structure=True,
                include_page_breaks=True,
                languages=["eng"]
            )
            
            print(f"   Extracted {len(elements)} elements")
            
            # Convert to markdown
            markdown_content = self.elements_to_markdown(elements, metadata)
            
            # Determine output path
            doc_type_dir = output_dir / metadata['document_type']
            doc_type_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = doc_type_dir / f"{pdf_path.stem}.md"
            
            # Write markdown file
            output_file.write_text(markdown_content, encoding='utf-8')
            
            print(f"   ✅ Saved to: {output_file.relative_to(output_dir)}")
            
            return output_file
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            return None
    
    def convert_directory(
        self, 
        input_dir: Path, 
        output_dir: Path,
        pattern: str = "*.pdf"
    ) -> Dict[str, Any]:
        """Convert all PDFs in a directory."""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all PDFs
        pdf_files = list(input_dir.glob(pattern))
        print(f"\n🔍 Found {len(pdf_files)} PDF files in {input_dir}")
        
        results = {
            "total_files": len(pdf_files),
            "successful": 0,
            "failed": 0,
            "conversions": []
        }
        
        # Convert each PDF
        for pdf_file in sorted(pdf_files):
            output_file = self.convert_pdf_to_markdown(pdf_file, output_dir)
            
            if output_file:
                results["successful"] += 1
                results["conversions"].append({
                    "status": "success",
                    "input": str(pdf_file),
                    "output": str(output_file),
                    "size_kb": output_file.stat().st_size / 1024
                })
            else:
                results["failed"] += 1
                results["conversions"].append({
                    "status": "failed",
                    "input": str(pdf_file)
                })
        
        # Save conversion report
        report_file = output_dir / "conversion_report.json"
        report_file.write_text(json.dumps(results, indent=2))
        
        # Print summary
        print(f"\n📊 Conversion Summary:")
        print(f"   Total files: {results['total_files']}")
        print(f"   Successful: {results['successful']}")
        print(f"   Failed: {results['failed']}")
        print(f"   Report saved to: {report_file}")
        
        # Create index file
        self._create_index_file(output_dir, results)
        
        return results
    
    def _create_index_file(self, output_dir: Path, results: Dict[str, Any]):
        """Create an index file with links to all converted documents."""
        index_content = ["# Converted Documents Index\n"]
        index_content.append(f"Generated at: {datetime.utcnow().isoformat()}\n")
        
        # Group by document type
        by_type = {}
        for conv in results['conversions']:
            if conv['status'] == 'success':
                output_path = Path(conv['output'])
                doc_type = output_path.parent.name
                if doc_type not in by_type:
                    by_type[doc_type] = []
                by_type[doc_type].append(output_path)
        
        # Write sections
        for doc_type in sorted(by_type.keys()):
            index_content.append(f"\n## {doc_type.replace('_', ' ').title()}\n")
            
            for file_path in sorted(by_type[doc_type]):
                rel_path = file_path.relative_to(output_dir)
                file_name = file_path.stem.replace('_', ' ').replace('-', ' ')
                index_content.append(f"- [{file_name}]({rel_path})")
        
        # Write index file
        index_file = output_dir / "INDEX.md"
        index_file.write_text("\n".join(index_content))
        print(f"   Index created: {index_file}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Convert PDFs to Markdown using Unstructured.io"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="kb",
        help="Input directory containing PDFs (default: kb)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/markdown",
        help="Output directory for markdown files (default: data/markdown)"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.pdf",
        help="File pattern to match (default: *.pdf)"
    )
    
    args = parser.parse_args()
    
    # Create converter
    converter = UnstructuredPDFConverter()
    
    # Convert PDFs
    print(f"🚀 Starting PDF to Markdown conversion")
    print(f"   Input: {args.input}")
    print(f"   Output: {args.output}")
    
    results = converter.convert_directory(
        Path(args.input),
        Path(args.output),
        args.pattern
    )
    
    print("\n✨ Conversion complete!")


if __name__ == "__main__":
    main()
