from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import json

import google.genai as genai


from schemas_and_prompts import (
    CALL1_SCHEMA, CALL1_PROMPT, 
    CALL2_SCHEMAS, CALL2_PROMPTS, 
    qa_answers_schema, generic_qa_prompt_template, qa_rules
)
from predefined_questions import slides_question_templates, release_question_templates


api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable is required")

MODEL = "gemini-2.5-pro"

qa_questions = {
    "earnings_release": release_question_templates,
    "earnings_slides": slides_question_templates,
}


def _client(api_key: Optional[str] = None) -> genai.Client:
    # api_key=None -> picks up GOOGLE_API_KEY
    return genai.Client(api_key=api_key)

def _upload(client: genai.Client, path: str):
    return client.files.upload(file=Path(path))

def _generate_json(client: genai.Client, file_part, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    resp = client.models.generate_content(
        model=MODEL,
        contents=[
            {"role": "user", "parts": [{"text": prompt}]},
            file_part
        ],
        config=genai.types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
            response_schema=schema
        ),
    )
    return json.loads(resp.text or "{}")


def detect_metadata_and_hints(pdf_path: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """Call 1: returns {document_type, document_date, ticker, quarter, source, page_hints}"""
    client = _client(api_key)
    part = _upload(client, pdf_path)
    return _generate_json(client, part, CALL1_PROMPT, CALL1_SCHEMA)

def extract_by_type(pdf_path: str, meta: Dict[str, Any], api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Call 2: runs the correct extractor based on meta['document_type'].
    Returns a strict JSON object conforming to the selected schema.
    """
    doc_type = meta.get("document_type") or "general"
    if doc_type not in CALL2_SCHEMAS:
        doc_type = "general"

    prompt = CALL2_PROMPTS[doc_type]
    # If we have useful page_hints, nudge the model to focus (soft constraint)
    hints = meta.get("page_hints") or {}
    hint_text = []
    for k, pages in hints.items():
        if pages:
            hint_text.append(f"{k}: focus pages {sorted(set(pages))}")
    if hint_text:
        prompt = f"Page focus hints:\n- " + "\n- ".join(hint_text) + "\n\n" + prompt

    client = _client(api_key)
    part = _upload(client, pdf_path)
    data = _generate_json(client, part, prompt, CALL2_SCHEMAS[doc_type])

    # Attach top-level fields you wanted consistently
    envelope = {
        "document_date": meta.get("document_date"),
        "document_type": doc_type,
        "ticker": meta.get("ticker"),
        "quarter": meta.get("quarter"),
        "extracted": data
    }
    return envelope

def extract_document(pdf_path: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience wrapper:
      1) Detect metadata + hints
      2) Run type-specific extraction
    """
    meta = detect_metadata_and_hints(pdf_path, api_key=api_key)
    return extract_by_type(pdf_path, meta, api_key=api_key)


def _safe(s: Optional[str]) -> str:
    return "" if s is None else str(s).strip()

def _fmt_title(meta: Dict[str, Any]) -> str:
    doc_type = _safe(meta.get("document_type")).replace("_", " ").title() or "Document"
    quarter = _safe(meta.get("quarter"))
    ticker = _safe(meta.get("ticker"))
    date = _safe(meta.get("document_date"))
    date_str = ""
    if date:
        try:
            dt = datetime.strptime(date.replace("/", "-"), "%Y-%m-%d")
            date_str = dt.strftime("%B %d, %Y")
        except Exception:
            date_str = date
    parts = [doc_type]
    if quarter: parts.append(quarter)
    if ticker: parts.append(f"({ticker})")
    if date_str: parts.append(f"— {date_str}")
    return " ".join(parts).strip()

def _mk_header(meta: Dict[str, Any]) -> str:
    title = _fmt_title(meta)
    summary = _safe(meta.get("summary"))
    out = [f"# {title}"]
    if summary:
        out += ["", "## Summary", summary]
    return "\n".join(out).strip() + "\n"

def _format_source_block(filename: str, page: Optional[int], quote: str) -> str:
    page_str = str(page) if isinstance(page, int) else "-"
    # Collapsible "Source" with label/value lines
    return (
        "  <details>\n"
        "    <summary>Source</summary>\n\n"
        f"    filename: {filename}\n\n"
        f"    page: {page_str}\n\n"
        f"    quote: {quote}\n\n"
        "  </details>"
    )

def _format_qa_item(item: Dict[str, Any], meta: Dict[str, Any], show_quote: bool) -> str:
    q = _safe(item.get("question"))
    a = _safe(item.get("answer")) or "unknown"
    pg = item.get("page", None)
    qt = _safe(item.get("quote"))
    filename = _safe(meta.get("filename", "document"))
    lines = [f"Q: {q}", f"A: {a}"]
    if show_quote and qt:
        lines.append(_format_source_block(filename, pg, qt))
    return "\n".join(lines)

def qa_to_markdown(
    qa: List[Dict[str, Any]],
    meta: Dict[str, Any],
    include_unknown: bool = False,
    show_quotes: bool = True,
    unknown_label: str = "unknown",
) -> str:
    """
    Build Markdown with meta header + Q&A.
    Format per item:
    - Q: question?
      A: answer
          Source > (collapsible)
              filename:
              page:
              quote:
    """
    header = _mk_header(meta)
    items = []
    for item in qa:
        ans = _safe(item.get("answer"))
        if not include_unknown and ans.lower() == unknown_label.lower():
            continue
        items.append(_format_qa_item(item, meta, show_quote=show_quotes))
    body = "## Q&A\n" + ("\n\n".join(items) if items else "_No Q&A entries to display._")
    return f"{header}\n\n{body}\n"

def write_markdown(
    qa: List[Dict[str, Any]],
    meta: Dict[str, Any],
    path: str,
    include_unknown: bool = False,
    show_quotes: bool = True,
    unknown_label: str = "unknown",
) -> str:
    md = qa_to_markdown(
        qa, meta,
        include_unknown=include_unknown,
        show_quotes=show_quotes,
        unknown_label=unknown_label,
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    return path

def _number_questions(questions: List[str]) -> str:
    return "\n".join(f"[{i}] {q}" for i, q in enumerate(questions))

def _blocks_from_meta(meta: Dict[str, Any]) -> Dict[str, str]:
    """Build the page_hints and metadata blocks from Call-1 meta."""
    hints = meta.get("page_hints") or {}
    ph_lines = [f"- {k}: focus pages {sorted(set(v))}" for k, v in hints.items() if isinstance(v, list) and v]
    page_hints_block = ("Page focus hints:\n" + "\n".join(ph_lines) + "\n\n") if ph_lines else ""

    md_keys = ["document_type", "quarter", "document_date", "ticker"]
    md_pairs = [f"{k}: {meta.get(k)}" for k in md_keys if meta.get(k)]
    metadata_block = ""
    if md_pairs:
        metadata_block = (
            "Metadata (for normalization only—PDF text is source of truth):\n"
            + "\n".join(md_pairs) + "\n\n"
        )
    return {"page_hints_block": page_hints_block, "metadata_block": metadata_block}

def build_prompt_for_qa_with_meta(meta: Dict[str, Any], qa_rules: Dict[str, Any]) -> str:
    blocks = _blocks_from_meta(meta)
    doc_type = meta["document_type"]
    rules = qa_rules[doc_type]
    questions = qa_questions[doc_type]
    return generic_qa_prompt_template.format(
        page_hints_block=blocks["page_hints_block"],
        metadata_block=blocks["metadata_block"],
        doc_rules=rules.strip(),
        numbered_questions=_number_questions(questions),
    )


def run_generic_qa_with_meta(
    pdf_path: str,
    meta: Dict[str, Any],
    *,
    response_schema: Dict[str, Any] = qa_answers_schema,
    qa_rules: Dict[str, Any] = qa_rules,
    model: str = MODEL,
    temperature: float = 0.1,
) -> Dict[str, Any]:
    """Meta-aware generic Q&A: prompt = (page_hints + metadata from meta) + doc_rules + numbered questions."""
    client = genai.Client(api_key=None)  # uses GOOGLE_API_KEY
    file_part = client.files.upload(file=Path(pdf_path))

    prompt = build_prompt_for_qa_with_meta(meta, qa_rules)

    resp = client.models.generate_content(
        model=model,
        contents=[{"role": "user", "parts": [{"text": prompt}]}, file_part],
        config=genai.types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json",
            response_schema=response_schema,
        ),
    )
    return json.loads(resp.text or "{}")



if __name__ == "__main__":
    pdf_directory = "path_to_pdf_directory"
    output_directory = "path_to_output_directory"

    os.makedirs(output_directory, exist_ok=True)

    for filename in os.listdir(pdf_directory):
        path = os.path.join(pdf_directory, filename)
        if not path.endswith(".pdf"):
            continue

        try:
            meta = detect_metadata_and_hints(path, api_key=api_key)

            if meta["document_type"] in ["earnings_slides", "earnings_release"]:
                response = run_generic_qa_with_meta(path, meta)

                write_markdown(response["answers"], meta, path=os.path.join(output_directory, filename.replace("pdf", "md")))
            else:
                # TODO: Implement transcript and product updates extraction
                pass
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
