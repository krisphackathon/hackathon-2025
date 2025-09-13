
import os
import json
from typing import Optional
from google.api_core.client_options import ClientOptions
from google.cloud import documentai_v1

project_id = "krisp-hackathon"
processor_id = "2b49042bea46fe6b"
processor_version = "pretrained-form-parser-v2.0-2022-11-10"
location = "us"

kb_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../kb"))
output_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../data/google_document_ai_processed")
)
os.makedirs(output_dir, exist_ok=True)

def process_pdf_sequential():
    """
    Process each PDF in kb_dir using the synchronous Document AI API and save output as JSON.
    """
    print(f"Using KB directory: {kb_dir}")
    print(f"Using output directory: {output_dir}")

    opts = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")
    client = documentai_v1.DocumentProcessorServiceClient(client_options=opts)
    full_processor_name = client.processor_version_path(
        project_id, location, processor_id, processor_version
    )
    request = documentai_v1.GetProcessorVersionRequest(name=full_processor_name)
    processor = client.get_processor_version(request=request)
    print(f"Processor Name: {processor.name}")

    for filename in os.listdir(kb_dir):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(kb_dir, filename)
            print(f"Processing {file_path} ...")

            with open(file_path, "rb") as f:
                pdf_content = f.read()
            raw_document = documentai_v1.RawDocument(
                content=pdf_content,
                mime_type="application/pdf",
            )
            try:
                process_request = documentai_v1.ProcessRequest(
                    name=processor.name, raw_document=raw_document
                )
                result = client.process_document(request=process_request)
                document = result.document
                output_path = os.path.join(output_dir, filename.replace(".pdf", ".json"))
                with open(output_path, "w") as f:
                    document_dict = documentai_v1.Document.to_dict(document)
                    json.dump(document_dict, f, indent=4)
                print(f"Saved processed output to {output_path}")
            except Exception as e:
                print(f"Failed to process {file_path}: {e}")

def batch_process_documents(
    project_id: str,
    location: str,
    processor_id: str,
    gcs_output_uri: str,
    processor_version_id: Optional[str] = None,
    gcs_input_uri: Optional[str] = None,
    input_mime_type: Optional[str] = None,
    gcs_input_prefix: Optional[str] = None,
    field_mask: Optional[str] = None,
    timeout: int = 400,
):
    """
    Batch process PDFs using Google Document AI. Requires input/output in GCS.
    """
    from google.cloud import documentai
    from google.cloud import storage
    import re
    from google.api_core.exceptions import InternalServerError, RetryError

    opts = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")
    client = documentai.DocumentProcessorServiceClient(client_options=opts)

    if gcs_input_uri:
        gcs_document = documentai.GcsDocument(
            gcs_uri=gcs_input_uri, mime_type=input_mime_type
        )
        gcs_documents = documentai.GcsDocuments(documents=[gcs_document])
        input_config = documentai.BatchDocumentsInputConfig(gcs_documents=gcs_documents)
    else:
        gcs_prefix = documentai.GcsPrefix(gcs_uri_prefix=gcs_input_prefix)
        input_config = documentai.BatchDocumentsInputConfig(gcs_prefix=gcs_prefix)

    gcs_output_config = documentai.DocumentOutputConfig.GcsOutputConfig(
        gcs_uri=gcs_output_uri, field_mask=field_mask
    )
    output_config = documentai.DocumentOutputConfig(gcs_output_config=gcs_output_config)

    if processor_version_id:
        name = client.processor_version_path(
            project_id, location, processor_id, processor_version_id
        )
    else:
        name = client.processor_path(project_id, location, processor_id)

    request = documentai.BatchProcessRequest(
        name=name,
        input_documents=input_config,
        document_output_config=output_config,
    )

    operation = client.batch_process_documents(request)
    try:
        print(f"Waiting for operation {operation.operation.name} to complete...")
        operation.result(timeout=timeout)
    except (RetryError, InternalServerError) as e:
        print(e.message)

    metadata = documentai.BatchProcessMetadata(operation.metadata)
    if metadata.state != documentai.BatchProcessMetadata.State.SUCCEEDED:
        raise ValueError(f"Batch Process Failed: {metadata.state_message}")

    # Note: Uncomment below to fetch and print output documents from GCS
    
    # storage_client = storage.Client()
    # print("Output files:")
    # for process in list(metadata.individual_process_statuses):
    #     matches = re.match(r"gs://(.*?)/(.*)", process.output_gcs_destination)
    #     if not matches:
    #         print("Could not parse output GCS destination:", process.output_gcs_destination)
    #         continue
    #     output_bucket, output_prefix = matches.groups()
    #     output_blobs = storage_client.list_blobs(output_bucket, prefix=output_prefix)
    #     for blob in output_blobs:
    #         if blob.content_type != "application/json":
    #             print(f"Skipping non-supported file: {blob.name} - Mimetype: {blob.content_type}")
    #             continue
    #         print(f"Fetching {blob.name}")
    #         document = documentai.Document.from_json(
    #             blob.download_as_bytes(), ignore_unknown_fields=True
    #         )
    #         print("The document contains the following text:")
    #         print(document.text)

if __name__ == "__main__":
    # To use local processing:
    # process_pdf_sequential()
    # To use batch processing, uncomment and fill in your GCS URIs:
    batch_process_documents(
        project_id=project_id,
        location=location,
        processor_id=processor_id,
        gcs_output_uri="gs://krisp-hackathon-2025/document-ai-processed/",
        processor_version_id=processor_version,  # or None
        gcs_input_prefix="gs://krisp-hackathon-2025/kb/",
        # field_mask="text,entities,pages.pageNumber",
        timeout=6000,
    )
