import json
import os

from google import genai
from google.cloud import storage
from google.genai import types

client = genai.Client(vertexai=True, project="krisp-hackathon", location="us-central1")

contents = [
    types.Part.from_text(text="Can you tell me about the highlights in the document?"),
    types.Part.from_uri(
        file_uri="gs://krisp-hackathon-2025/kb/2022_Q1_Earnings_Transcript.pdf",
        mime_type="application/pdf",
    ),
]

result = client.models.generate_content(
    model="gemini-2.5-pro",
    contents=contents,
)


def run_gemini_on_all_pdfs(
    bucket_name="krisp-hackathon-2025",
    input_prefix="kb",
    output_prefix="gemini-2.5-pro-processed",
):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=input_prefix)
    for blob in blobs:
        if blob.name.lower().endswith(".pdf"):
            gcs_uri = f"gs://{bucket_name}/{blob.name}"
            filename = os.path.basename(blob.name)
            print(f"Processing {gcs_uri} ...")
            contents = [
                types.Part.from_text(
                    text="Can you tell me about the highlights in the document?"
                ),
                types.Part.from_uri(
                    file_uri=gcs_uri,
                    mime_type="application/pdf",
                ),
            ]
            result = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=contents,
            )
            # Save result to GCS as JSON
            output_blob = bucket.blob(f"{output_prefix}/{filename}.json")
            output_blob.upload_from_string(
                json.dumps({ "text": result.text }, indent=2),
                content_type="application/json",
            )
            print(f"Saved result to gs://{bucket_name}/{output_prefix}/{filename}.json")


if __name__ == "__main__":
    run_gemini_on_all_pdfs()
