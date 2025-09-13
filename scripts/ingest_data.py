#!/usr/bin/env python3
"""Script to ingest meeting data into the system."""

import argparse
import asyncio
import sys
from pathlib import Path
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.services.ingestion import ingestion_service
from backend.services.vector_db import vector_db
from backend.models import IngestRequest


async def ingest_single_file(file_path: Path, data_type: str = "transcript") -> dict:
    """Ingest a single file."""
    print(f"Processing: {file_path}")
    
    try:
        meeting_id = file_path.stem
        
        if data_type == "transcript":
            request = IngestRequest(
                meeting_id=meeting_id,
                transcription_file_path=str(file_path),
                metadata={
                    "source_file": file_path.name,
                    "ingested_at": datetime.utcnow().isoformat(),
                    "data_type": data_type
                }
            )
        elif data_type == "audio":
            request = IngestRequest(
                meeting_id=meeting_id,
                audio_file_path=str(file_path),
                metadata={
                    "source_file": file_path.name,
                    "ingested_at": datetime.utcnow().isoformat(),
                    "data_type": data_type
                }
            )
        else:
            raise ValueError(f"Unknown data type: {data_type}")
        
        response = await ingestion_service.ingest_meeting(request)
        
        return {
            "file": str(file_path),
            "meeting_id": meeting_id,
            "status": response.status,
            "chunks_created": response.chunks_created,
            "processing_time_ms": response.processing_time_ms,
            "errors": response.errors
        }
        
    except Exception as e:
        return {
            "file": str(file_path),
            "meeting_id": meeting_id,
            "status": "failed",
            "chunks_created": 0,
            "error": str(e)
        }


async def main():
    """Main ingestion function."""
    parser = argparse.ArgumentParser(description="Ingest meeting data")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Source directory or file path"
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["transcript", "audio"],
        default="transcript",
        help="Type of data to ingest"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.json",
        help="File pattern for bulk ingestion (default: *.json)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for concurrent processing"
    )
    
    args = parser.parse_args()
    
    # Initialize vector database
    print("Initializing vector database...")
    await vector_db.initialize()
    
    source_path = Path(args.source)
    
    if source_path.is_file():
        # Single file ingestion
        result = await ingest_single_file(source_path, args.type)
        print(json.dumps(result, indent=2))
        
    elif source_path.is_dir():
        # Bulk ingestion
        files = list(source_path.glob(args.pattern))
        print(f"Found {len(files)} files matching pattern: {args.pattern}")
        
        results = {
            "total_files": len(files),
            "successful": 0,
            "failed": 0,
            "total_chunks": 0,
            "processing_time_ms": 0,
            "errors": []
        }
        
        start_time = datetime.utcnow()
        
        # Process in batches
        for i in range(0, len(files), args.batch_size):
            batch_files = files[i:i + args.batch_size]
            print(f"\nProcessing batch {i//args.batch_size + 1}/{(len(files) + args.batch_size - 1)//args.batch_size}")
            
            # Process batch concurrently
            tasks = [
                ingest_single_file(file_path, args.type)
                for file_path in batch_files
            ]
            
            batch_results = await asyncio.gather(*tasks)
            
            # Aggregate results
            for result in batch_results:
                if result["status"] == "success":
                    results["successful"] += 1
                    results["total_chunks"] += result["chunks_created"]
                    results["processing_time_ms"] += result.get("processing_time_ms", 0)
                else:
                    results["failed"] += 1
                    results["errors"].append({
                        "file": result["file"],
                        "error": result.get("error", result.get("errors", "Unknown error"))
                    })
                
                # Print progress
                print(f"  - {Path(result['file']).name}: {result['status']} "
                      f"({result['chunks_created']} chunks)")
        
        # Calculate total time
        total_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Get final statistics
        stats = await vector_db.get_collection_stats()
        
        # Print summary
        print("\n" + "="*60)
        print("INGESTION COMPLETE")
        print("="*60)
        print(f"Total files processed: {results['total_files']}")
        print(f"Successful: {results['successful']}")
        print(f"Failed: {results['failed']}")
        print(f"Total chunks created: {results['total_chunks']}")
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Average time per file: {total_time/len(files):.2f} seconds")
        print(f"\nVector database statistics:")
        print(f"  - Total vectors: {stats.get('vectors_count', 0)}")
        print(f"  - Total points: {stats.get('points_count', 0)}")
        
        if results["errors"]:
            print(f"\nErrors ({len(results['errors'])} files):")
            for error in results["errors"][:10]:  # Show first 10 errors
                print(f"  - {error['file']}: {error['error']}")
            if len(results["errors"]) > 10:
                print(f"  ... and {len(results['errors']) - 10} more errors")
        
        # Save detailed results
        results_file = source_path / f"ingestion_results_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to: {results_file}")
        
    else:
        print(f"Error: {source_path} not found")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
