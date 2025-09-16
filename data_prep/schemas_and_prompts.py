from typing import Dict, Any, List, Optional


CALL1_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "document_type": {
            "type": "string",
            "enum": [
                "earnings_release",
                "earnings_slides",
                "earnings_transcript",
                "product_updates",
                "general"
            ]
        },
        "document_date": {"type": "string"},  # ISO-8601 if possible
        "ticker": {"type": "string"},
        "quarter": {"type": "string"},        # e.g., "Q2 2025" or null
        "page_hints": {
            "type": "object",
            "properties": {
                "kpis": {"type": "array", "items": {"type": "integer"}},
                "guidance": {"type": "array", "items": {"type": "integer"}},
                "quotes": {"type": "array", "items": {"type": "integer"}},
                "figures": {"type": "array", "items": {"type": "integer"}}
            }
        },
        "summary": {"type": "string"}
    },
    "required": ["document_type", "summary"]
}

CALL1_PROMPT = """
You will read the attached PDF and identify metadata and helpful page hints.

Return JSON ONLY with keys:
- document_type: one of [earnings_release, earnings_slides, earnings_transcript, product_updates, general]
- document_date: best-effort date string (ISO if visible) or null
- ticker: best-effort (e.g., GOOG, GOOGL) or null
- quarter: best-effort (e.g., "Q2 2025") or null
- page_hints: { kpis: [pages], guidance: [pages], quotes: [pages], figures: [pages] } (omit arrays if not applicable)
- summary: 1–5 sentences, plain text, highlighting what the document is about.

Heuristics:
- 'earnings_release' usually contains official GAAP KPIs and summary tables.
- 'earnings_slides' are short pages with big figure callouts.
- 'earnings_transcript' has speaker turns (CEO/CFO, Q&A).
- 'product_updates' are short product/feature availability notes with dates.
- 'general' is anything else (policy/PR statements).

If unsure between two types, pick the best.
"""


earnings_transcript_schema = {
    "type": "object",
    "properties": {
        "transcript": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "speaker": {"type": "string"},
                    "company": {"type": "string"},
                    "role": {"type": "string"},
                    "text": {"type": "string"},
                    "page": {"type": "integer", "minimum": 1}
                },
                "required": ["speaker", "text", "page"]
            }
        }
    },
    "required": ["transcript"]
}

earnings_transcript_prompt = """
Read the PDF. Return JSON ONLY per schema.
Rules:
- Do not skip any turn (including operator speech)
- Each transcript item MUST have speaker and page. Include company/role if stated near the name. (Special case: Operator)
- In case the new paragraph starts without mentioning a speaker, it belongs to the latest mentioned speaker.
- If the text becomes too long end the item and start a new one with the same speaker.
- Keep texts as they are. Do not paraphrase. 
"""

product_updates_schema = {
    "type": "object",
    "properties": {
        "document_type": {"type": "string"},      # e.g., "product_updates"
        "title": {"type": "string"},
        "publish_date": {"type": "string"},       # ISO-8601 if present
        "updates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    # --- Core fields ---
                    "product": {"type": "string"},
                    "change": {"type": "string"},          # short clause (what changed)
                    "details": {"type": "string"},         # 1–3 sentences (optional if change is self-explanatory)
                    "availability": {
                        "type": "object",
                        "properties": {
                            "regions": {"type": "array", "items": {"type": "string"}},
                            "languages": {"type": "array", "items": {"type": "string"}},
                            "start_date": {"type": "string"}  # ISO if present
                        }
                    },
                    "impacts": {"type": "array", "items": {"type": "string"}},  # user/business benefits (bullets)
                    "page": {"type": "integer", "minimum": 1},
                    "quote": {"type": "string"},

                    # --- Optional enrichments ---
                    "rollout_stage": {
                        "type": "string",
                        "enum": ["preview", "beta", "GA", "experiment", "rollback"]
                    },
                    "links": {"type": "array", "items": {"type": "string"}},
                    "prerequisites": {"type": "array", "items": {"type": "string"}},
                    "deprecations": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["product", "change", "page", "quote"]
            }
        }
    },
    "required": ["document_type", "updates"]
}

product_updates_prompt = """
Read the attached PDF and return JSON ONLY per the schema.

Rules:
- Each discrete feature/change = one object in "updates".
- Always extract: product, change, page (1-indexed), and an exact short quote from the PDF supporting the change.
- Include details (1–3 sentences) when helpful; otherwise omit.
- Normalize availability:
  - Map locations into availability.regions (e.g., ["US","India","EU"]) and languages into availability.languages (e.g., ["Hindi","Japanese"]).
  - If a rollout date is stated, add availability.start_date (ISO if possible).
- If rollout_stage (preview/beta/GA/experiment/rollback) is explicitly stated, include it; otherwise omit.
- Add links/prerequisites/deprecations only when explicitly present; do not speculate.
- Merge near-duplicate updates; keep the version with strongest evidence.
- If a detail does not map to the defined fields, omit it. Do NOT create extra keys.
"""

general_statements_schema = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "publish_date": {"type": "string"},           # ISO if present
        "statements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "statement_type": {
                        "type": "string",
                        "enum": ["announcement","policy","product","metric",
                                 "risk","legal","partnership","organizational","other"]
                    },   
                    "date": {"type": "string"},       
                    "page": {"type": "integer", "minimum": 1},
                    "quote": {"type": "string"}
                },
                "required": ["page", "quote"]
            }
        }
    },
    "required": ["document_type", "statements"]
}

general_statements_prompt = """
Read the PDF and return JSON ONLY per schema.

Rules:
- Extract 3–10 concise, factual statements; each MUST include page (1-indexed) and an exact short quote from the PDF.
- Keep `text` to 1–3 sentences; neutral tone; no speculation.
- If a specific date is shown for a statement/event, set `date` in ISO-8601; otherwise omit.
- Set `statement_type` from: ["announcement","policy","product","metric","risk","legal","partnership","organizational","other"].
- Do NOT invent fields; omit anything not explicitly present. Merge near-duplicates.
"""


earnings_release_schema = {
    "type": "object",
    "properties": {
        "document_type": {"type": "string"},
        "period": {"type": "string"},
        "currency": {"type": "string"},
        "kpis": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "metric": {"type": "string"},
                    "value": {"type": "number"},
                    "unit": {"type": "string"},
                    "page": {"type": "integer", "minimum": 1},
                    "quote": {"type": "string"}
                },
                "required": ["metric", "value", "unit", "page"]
            }
        },
        "guidance": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "metric": {"type": "string"},
                    "range": {"type": "string"},
                    "period": {"type": "string"},
                    "page": {"type": "integer", "minimum": 1},
                    "quote": {"type": "string"}
                },
                "required": ["metric", "range", "period", "page"]
            }
        },
        "highlights": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "sentiment": {"type": "string"},
                    "page": {"type": "integer", "minimum": 1},
                    "quote": {"type": "string"}
                },
                "required": ["summary", "sentiment", "page"]
            }
        }
    },
    "required": ["document_type", "kpis"]
}

earnings_release_prompt = """
Read the PDF. Return JSON ONLY per schema.
Rules:
- Prefer GAAP figures. Every numeric KPI MUST include value, unit, page, and short supporting quote.
- Include guidance ranges if present, with the target period.
- Keep period and currency if visible anywhere.
"""


earnings_slides_schema = {
    "type": "object",
    "properties": {
        "document_type": {"type": "string"},
        "period": {"type": "string"},
        "slides": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "page": {"type": "integer", "minimum": 1},
                    "highlights": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["page", "highlights"]
            }
        },
        "conflicts_with_release": {
            "type": "array",
            "items": {"type": "string"}
        },
        "other_interesting_information": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "page": {"type": "integer", "minimum": 1},
                    "quote": {"type": "string"}
                },
                "required": ["text", "page"]
            }
        }
    },
    "required": ["document_type", "slides"]
}

release_question_templates = [
    # 📊 Core Financials (Quarterly)
    "What was the total revenue for the quarter?",
    "What was the year-over-year (YoY) revenue growth for the quarter?",
    "What was the operating income for the quarter?",
    "What was the operating margin for the quarter?",
    "What was the net income for the quarter?",
    "What was the diluted earnings per share (EPS) for the quarter?",
    "What was the free cash flow for the quarter?",
    "What were the net cash provided by operating activities for the quarter?",
    "What was the effective tax rate for the quarter?",

    #  fiscal-year-results Core Financials (Annual)
    "What was the total revenue for the fiscal year?",
    "What was the year-over-year (YoY) revenue growth for the fiscal year?",
    "What was the operating income for the fiscal year?",
    "What was the operating margin for the fiscal year?",
    "What was the net income for the fiscal year?",
    "What was the diluted earnings per share (EPS) for the fiscal year?",

    # 🏢 Segment Results
    "What were the revenues for each reported business segment?",
    "What was the operating income (or loss) for each reported business segment?",
    "What was the year-over-year (YoY) revenue growth for each segment?",
    "What was the total advertising revenue?",
    "What was the total subscription revenue?",

    # 👩‍💼 Workforce & Operations
    "What was the total number of employees at the end of the period?",
    "What were the total costs and expenses?",
    "What were the research and development (R&D) expenses?",
    "What were the sales and marketing (S&M) expenses?",
    "What were the general and administrative (G&A) expenses?",
    "What were the capital expenditures (CapEx)?",

    # 💰 Capital Return & Balance Sheet
    "What was the cash dividend declared per share?",
    "What is the dividend payment date?",
    "What is the dividend record date?",
    "What was the value of stock repurchased during the period?",
    "What was the value of any new or expanded stock repurchase authorization?",
    "What was the balance of cash, cash equivalents, and marketable securities?",
    "What was the total debt (short-term and long-term)?",
    "What were total assets?",
    "What were total liabilities?",
    "What was total stockholders' equity?",

    # 📈 Outlook & Guidance
    "What is the revenue guidance for the next quarter or full year?",
    "What is the operating margin guidance for the next quarter or full year?",
    "What is the capital expenditure (CapEx) guidance for the full year?"
]


CALL2_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "earnings_transcript": earnings_transcript_schema,
    "earnings_slides": earnings_slides_schema,
    "earnings_release": earnings_release_schema,
    "product_updates": product_updates_schema,
    "general": general_statements_schema,
}

earnings_slides_prompt = """
Read the PDF. Return JSON ONLY per schema.
Rules:
- For each slide (page), extract 1–5 highlights with numbers and units when present.
- Use page numbers (1-indexed).
- If a number appears to contradict a press release, add a short string to conflicts_with_release describing it.
- Keep highlights as bullet-sized phrases (no paragraphs).
"""

CALL2_PROMPTS: Dict[str, str] = {
    "earnings_transcript": earnings_transcript_prompt,
    "earnings_slides": earnings_slides_prompt,
    "earnings_release": earnings_release_prompt,
    "product_updates": product_updates_prompt,
    "general": general_statements_prompt,
}



qa_answers_schema = {
    "type": "object",
    "properties": {
        "answers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "question": {"type": "string"},
                    "answer": {"type": "string"},               # short answer OR "unknown"
                    "page": {"type": "integer", "minimum": 1},  # include only if answered
                    "quote": {"type": "string"},
                },
                "required": ["id", "question", "answer"]
            }
        },
        "period": {"type": "string"},
        "currency": {"type": "string"}
    },
    "required": ["answers"]
}



# Reusable prompt with spots for hints/metadata/doc-specific rules
generic_qa_prompt_template = """
You are answering questions using ONLY the attached PDF.

{page_hints_block}{metadata_block}
Document-specific rules:
{doc_rules}

Return JSON ONLY per the provided schema.

Answering rules:
- Use ONLY information explicitly in the PDF; do NOT use outside knowledge.
- For EACH question below:
  - If the answer is explicitly stated, write a short, direct answer.
  - Include page (1-indexed) and an EXACT short supporting quote from the PDF.
- If the PDF does NOT state the answer, set answer="unknown" and omit page/quote.
- Be concise. Do NOT add keys that are not in the schema.

Questions:
{numbered_questions}
""".strip()

qa_rules = {
    "earnings_release": """
Prefer GAAP figures and official tables. 
If visible, set optional period and currency exactly as written.
Answer only from this RELEASE PDF; do not use outside knowledge.

Answer format policy:
- Return a SINGLE compact value, no prose (e.g., "$84.7B", "54.2%", "1.35x", "181,798").
- Preserve the units and precision exactly as printed (currency symbols/ISO, %, x, billions/millions letters).
- If the value is not stated in the RELEASE, set answer="unknown".

Evidence policy:
- For any answer ≠ "unknown", include page (1-indexed) and an EXACT short supporting quote from the PDF.
""",
    "earnings_slides": """
Rely on numbers/labels visible in charts and tables.
Cite the slide page where the chart appears; quote exact labels.
Do not infer values that are not legible in the slide.
If a figure conflicts with the press release, still report what the slide shows (no reconciliation).

Answer format policy:
- Return a SINGLE compact value or short phrase, no prose (e.g., "Rev trend ↑ YoY", "54.2%", "$13.6B", "North America >50%").
- Preserve the units/precision exactly as printed.
- If not stated on the slide, set answer="unknown".

Evidence policy:
- For any answer ≠ "unknown", include page (1-indexed) and an EXACT short supporting quote from the slide.
"""
}