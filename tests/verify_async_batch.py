import asyncio
import os
from typing import AsyncGenerator, Tuple
from clinical_note_miner.schema import ExtractionSchema, ExtractionElement
from clinical_note_miner.llm import LLMClient
from clinical_note_miner.pipeline import BatchProcessor

# Mock LLM Client to avoid actual API calls
from types import SimpleNamespace

class MockLLMClient(LLMClient):
    async def chat_completion(self, messages):
        content = """```json
{
    "diagnosis": "Migraine",
    "symptoms": ["headache", "nausea"]
}
```"""
        # Mocking the OpenAI response structure
        message = SimpleNamespace(content=content)
        choice = SimpleNamespace(message=message)
        response = SimpleNamespace(choices=[choice])
        
        return {
            "response": response,
            "latency": 0.1,
            "success": True
        }

async def note_generator(notes) -> AsyncGenerator[Tuple[str, str], None]:
    for note in notes:
        yield note

async def main():
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

    # 2. Initialize Client
    client = MockLLMClient(
        base_url="https://api.openai.com/v1",
        api_key="fake-key",
        model="gpt-4o"
    )

    # 3. Initialize Processor
    processor = BatchProcessor(
        schema=schema,
        llm_client=client,
        output_file="test_results.jsonl",
        overwrite=True
    )

    notes = [
        ("note_1", "Patient presents with severe headache and nausea. Diagnosis: Migraine."),
        ("note_2", "Patient reports mild fever and cough.")
    ]
    
    print("Starting batch processing...")
    async for result in processor.process_batch(note_generator(notes), total=len(notes)):
        print(f"Processed {result['id']}: {result['extraction']}")

if __name__ == "__main__":
    asyncio.run(main())
