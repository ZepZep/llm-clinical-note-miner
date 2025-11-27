import os
from typing import Tuple
from clinical_note_miner import ExtractionSchema, ExtractionElement, FewShotExample, LLMClient, BatchProcessor

# Mock LLM for testing without API key
class MockLLMClient(LLMClient):
    async def chat_completion(self, messages: list) -> dict:
        # Simulate a response based on the input
        # In a real scenario, this would call the API
        import json
        
        # Simple mock response
        content = json.dumps({
            "tumor_size": {
                "answer": 25,
                "reasoning": "The text says 'tumor size 25mm'",
                "grounding": ["tumor size 25mm"]
            },
            "mitotic_activity": "5/10 HPF"
        })
        
        return {
            "response": type('obj', (object,), {
                "choices": [type('obj', (object,), {
                    "message": type('obj', (object,), {
                        "content": content
                    })
                })]
            }),
            "latency": 0.1,
            "success": True
        }

def main():
    # Define Schema
    schema = ExtractionSchema(elements=[
        ExtractionElement(
            name="tumor_size", 
            description="Size of the tumor in mm", 
            response_type=int,
            grounding=True,
            reasoning=True
        ),
        ExtractionElement(
            name="mitotic_activity", 
            description="Mitotic activity count", 
            response_type=str
        )
    ])
    
    # Initialize Client (using Mock for demo)
    # client = LLMClient(api_key="fake") 
    client = MockLLMClient(api_key="dummy")
    
    # Initialize Processor
    processor = BatchProcessor(
        schema=schema, 
        llm_client=client,
        output_file="results.jsonl",
        overwrite=True,
        enable_file_output=False
    )
    
    # Define Notes
    notes = [
        ("log_prompt_note_1", "Patient has a tumor size 25mm. Mitotic activity is 5/10 HPF."),
        ("note_2", "Another patient. Tumor size 10mm. No mitotic activity mentioned."),
    ]
    
    # Run Batch Sync
    print("Starting batch processing...")
    for result in processor.process_batch_sync(notes, total=len(notes)):
        print(f"Processed {result['id']}: Success={result['success']}")

if __name__ == "__main__":
    main()
