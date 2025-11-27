from clinical_note_miner.schema import ExtractionSchema, ExtractionElement
from clinical_note_miner.llm import LLMClient
from clinical_note_miner.pipeline import BatchProcessor
import time

# Mock LLM
class MockLLMClient(LLMClient):
    async def chat_completion(self, messages: list) -> dict:
        import json
        return {
            "response": type('obj', (object,), {
                "choices": [type('obj', (object,), {
                    "message": type('obj', (object,), {
                        "content": json.dumps({"test": {"answer": "val"}})
                    })
                })]
            }),
            "latency": 0.1,
            "success": True
        }

def run_test():
    schema = ExtractionSchema(elements=[
        ExtractionElement(name="test", description="test", response_type=str)
    ])
    client = MockLLMClient(api_key="dummy")
    processor = BatchProcessor(
        schema=schema, 
        llm_client=client, 
        overwrite=True,
        enable_prompt_logging=False,
        enable_file_output=False
    )
    
    notes = [("id_1", "text 1"), ("id_2", "text 2")]
    
    print("Starting sync batch processing...")
    count = 0
    for result in processor.process_batch_sync(notes, total=len(notes)):
        print(f"Processed {result['id']}: Success={result['success']}")
        count += 1
        
    if count == 2:
        print("Test Passed: Processed 2 notes synchronously.")
    else:
        print(f"Test Failed: Processed {count} notes.")

if __name__ == "__main__":
    run_test()
