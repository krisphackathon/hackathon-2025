# PDF to Markdown Conversion Guide

## Overview

This script converts PDF files to Markdown format using Unstructured.io, which provides excellent structure preservation and table extraction capabilities.

## Prerequisites

Install required dependencies:
```bash
pip install -r backend/requirements.txt
```

Note: You may need to install additional system dependencies:
```bash
# On macOS
brew install tesseract poppler

# On Ubuntu/Debian
sudo apt-get install tesseract-ocr poppler-utils

# On Windows
# Install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
# Install poppler from: http://blog.alivate.com.au/poppler-windows/
```

## Usage

### Basic Usage
```bash
python scripts/convert_pdfs_to_markdown.py
```

This will:
- Read PDFs from the `kb` folder
- Convert them to Markdown files in `data/markdown`
- Organize by document type (earnings_transcript, earnings_release, etc.)

### Custom Directories
```bash
python scripts/convert_pdfs_to_markdown.py --input /path/to/pdfs --output /path/to/markdown
```

### Specific Pattern
```bash
python scripts/convert_pdfs_to_markdown.py --pattern "*earnings*.pdf"
```

## Output Structure

```
data/markdown/
├── earnings_transcript/
│   ├── 2024_Q1_Earnings_Transcript.md
│   ├── 2024_Q2_Earnings_Transcript.md
│   └── ...
├── earnings_release/
│   ├── 2024q1-alphabet-earnings-release.md
│   └── ...
├── product_doc/
│   ├── google-photos-create-tab.md
│   └── ...
├── conversion_report.json
└── INDEX.md
```

## Features

1. **Smart Document Classification**
   - Automatically categorizes documents based on filename
   - Applies appropriate parsing strategy per document type

2. **Metadata Extraction**
   - Extracts company, year, quarter from filenames
   - Adds frontmatter to each markdown file

3. **Structure Preservation**
   - Maintains document hierarchy with proper headings
   - Preserves tables in markdown format
   - Identifies speakers in transcripts

4. **Processing Strategies**
   - `hi_res`: For earnings transcripts and releases (better OCR and layout detection)
   - `fast`: For simple product documents

## Example Output

### Earnings Transcript
```markdown
---
title: Alphabet Q2 2024 Earnings Call Transcript
source_file: 2024-q2-earnings-transcript.pdf
document_type: earnings_transcript
company: Alphabet
year: 2024
quarter: Q2
converted_at: 2024-01-15T10:30:00Z
---

# Alphabet Q2 2024 Earnings Call

**Sundar Pichai**: Good afternoon everyone, and thank you for joining us...

**Ruth Porat**: Thanks Sundar. Our strong Q2 results reflect...
```

### Earnings Release with Table
```markdown
## Financial Summary

| Metric | Q2 2024 | Q2 2023 | Change |
|--------|---------|---------|--------|
| Revenue | $84.7B | $74.6B | +14% |
| Operating Income | $27.4B | $21.8B | +26% |
```

## Troubleshooting

1. **OCR Issues**: Ensure Tesseract is installed and in PATH
2. **Memory Issues**: For large PDFs, process in smaller batches
3. **Table Extraction**: Tables in scanned PDFs may not extract perfectly

## Next Steps

After conversion, ingest the markdown files:
```bash
python scripts/ingest_data.py --source data/markdown --type transcript --pattern "**/*.md"
```
