# Clinical Note Miner

A Python framework for extracting structured data from clinical notes using OpenAI-compatible LLMs.

## Features

*   **Structured Extraction**: Define schemas using Pydantic-like elements.
*   **Grounding & Reasoning**: Automatically asks the LLM for reasoning and text evidence (grounding).
*   **Async Pipeline**: High-performance concurrent processing using `asyncio`.
*   **Retry Logic**: Robust error handling with configurable retries.
*   **Progress Monitoring**: Integrated `tqdm` progress bar.
*   **JSONLines Output**: Efficient line-by-line output for large datasets.

## Installation

```bash
pip install -r requirements.txt
```

## Simple Example

```python
import asyncio
import os
from clinical_note_miner import ExtractionSchema, ExtractionElement, LLMClient, BatchProcessor

# 1. Define your schema
schema = ExtractionSchema(
    elements=[
        ExtractionElement(
            name="diagnosis", 
            description="Primary diagnosis", 
            response_type=str
        ),
        ExtractionElement(
            name="symptoms", 
            description="List of symptoms mentioned", 
            response_type=list
        )
    ]
)

# 2. Initialize Client (works with OpenAI or compatible APIs like vLLM)
client = LLMClient(
    base_url="https://api.openai.com/v1", # or your local/hosted endpoint
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o"
)

# 3. Initialize Processor
processor = BatchProcessor(
    schema=schema,
    llm_client=client,
    output_file="results.jsonl",
    overwrite=True
)

# 4. Define notes [(id, text)] (can be a generator)
notes = [
    ("note_1", "Patient has fever."),
    ("note_2", "Patient reports cough.")
]

# 5. Batch Processing, works like a standard generator (Sync / Jupyter Friendly)
for result in processor.process_batch_sync(notes, total=len(notes)):
    print(f"Processed {result['id']}")


# Alternatively, you can process notes in async
async def note_generator(notes):
    for note in notes:
        yield note

async def main():
    # Process batch asynchronously
    async for result in processor.process_batch(
        notes=note_generator(notes),
        total=len(notes)
    ):
        print(f"Processed {result['id']}: {result['extraction']}")

if __name__ == "__main__":
    asyncio.run(main())
```


