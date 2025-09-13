#!/usr/bin/env python3
"""Simple PDF to Markdown conversion using pymupdf4llm."""

import argparse
import sys
from pathlib import Path
import json
import re
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import pymupdf4llm


class SimplePDFConverter:
    """Convert PDFs to Markdown using pymupdf4llm."""
    
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
        year_match = re.search(r"20(\d{2})", filename)
        if year_match:
            metadata["year"] = int(f"20{year_match.group(1)}")
        
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
        title_parts = []
        for part in title.split():
            if part.upper() in ["Q1", "Q2", "Q3", "Q4", "AI", "PDF"]:
                title_parts.append(part.upper())
            else:
                title_parts.append(part.capitalize())
        
        metadata["title"] = " ".join(title_parts)
        
        return metadata
    
    def convert_pdf_to_markdown(self, pdf_path: Path, output_dir: Path) -> Optional[Path]:
        """Convert a single PDF to Markdown."""
        try:
            print(f"📄 Processing: {pdf_path.name}")
            
            # Extract metadata
            metadata = self.extract_metadata_from_filename(pdf_path.name)
            
            # Convert to markdown
            markdown_text = pymupdf4llm.to_markdown(str(pdf_path))
            
            # Add frontmatter
            frontmatter = f"""---
title: {metadata.get('title', 'Untitled')}
source_file: {metadata.get('original_filename', 'unknown')}
document_type: {metadata.get('document_type', 'general')}
company: {metadata.get('company', 'Unknown')}
year: {metadata.get('year', 'Unknown')}
quarter: {metadata.get('quarter', 'Unknown')}
converted_at: {datetime.utcnow().isoformat()}
---

"""
            
            # Combine frontmatter and content
            full_content = frontmatter + markdown_text
            
            # Post-process for earnings transcripts
            if metadata['document_type'] == 'earnings_transcript':
                full_content = self._enhance_transcript_formatting(full_content)
            
            # Determine output path
            doc_type_dir = output_dir / metadata['document_type']
            doc_type_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = doc_type_dir / f"{pdf_path.stem}.md"
            
            # Write markdown file
            output_file.write_text(full_content, encoding='utf-8')
            
            print(f"   ✅ Saved to: {output_file.relative_to(output_dir)}")
            
            return output_file
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            return None
    
    def _enhance_transcript_formatting(self, content: str) -> str:
        """Enhance formatting for earnings transcripts."""
        lines = content.split('\n')
        enhanced_lines = []
        
        for line in lines:
            # Detect speaker patterns
            speaker_match = re.match(r"^([A-Z][a-zA-Z\s\.]+)(?:\s*[-–—:]|\s+[-–—])\s*(.*)$", line.strip())
            if speaker_match and len(speaker_match.group(1)) < 50:  # Reasonable name length
                speaker = speaker_match.group(1).strip()
                text = speaker_match.group(2).strip()
                enhanced_lines.append(f"\n**{speaker}**: {text}")
            else:
                enhanced_lines.append(line)
        
        return '\n'.join(enhanced_lines)
    
    def convert_directory(self, input_dir: Path, output_dir: Path) -> Dict[str, Any]:
        """Convert all PDFs in a directory."""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all PDFs
        pdf_files = list(input_dir.glob("*.pdf"))
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
        
        # Create index file
        self._create_index_file(output_dir, results)
        
        # Print summary
        print(f"\n📊 Conversion Summary:")
        print(f"   Total files: {results['total_files']}")
        print(f"   Successful: {results['successful']}")
        print(f"   Failed: {results['failed']}")
        print(f"   Report saved to: {report_file}")
        
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
        description="Convert PDFs to Markdown using pymupdf4llm"
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
    
    args = parser.parse_args()
    
    # Create converter
    converter = SimplePDFConverter()
    
    # Convert PDFs
    print(f"🚀 Starting PDF to Markdown conversion")
    print(f"   Input: {args.input}")
    print(f"   Output: {args.output}")
    
    results = converter.convert_directory(
        Path(args.input),
        Path(args.output)
    )
    
    print("\n✨ Conversion complete!")


if __name__ == "__main__":
    main()
