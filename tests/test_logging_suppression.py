import asyncio
from clinical_note_miner.schema import ExtractionSchema, ExtractionElement
from clinical_note_miner.llm import LLMClient
from clinical_note_miner.pipeline import BatchProcessor

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

async def run_test():
    schema = ExtractionSchema(elements=[
        ExtractionElement(name="test", description="test", response_type=str)
    ])
    client = MockLLMClient(api_key="dummy")
    
    # Test 1: Logging Enabled (Default)
    print("--- Test 1: Logging Enabled (Expect Prompt Output) ---")
    processor = BatchProcessor(schema=schema, llm_client=client, overwrite=True)
    await processor.process_note("log_prompt_test_1", "note text")
    
    # Test 2: Logging Disabled
    print("\n--- Test 2: Logging Disabled (Expect NO Prompt Output) ---")
    processor_quiet = BatchProcessor(
        schema=schema, 
        llm_client=client, 
        overwrite=True,
        enable_prompt_logging=False
    )
    await processor_quiet.process_note("log_prompt_test_2", "note text")
    print("Test 2 Completed.")

if __name__ == "__main__":
    asyncio.run(run_test())
