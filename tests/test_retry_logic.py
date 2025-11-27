import asyncio
from clinical_note_miner.schema import ExtractionSchema, ExtractionElement
from clinical_note_miner.llm import LLMClient
from clinical_note_miner.pipeline import BatchProcessor

# Mock LLM that fails initially
class FlakyLLMClient(LLMClient):
    def __init__(self, api_key, fail_count=2):
        super().__init__(api_key=api_key)
        self.fail_count = fail_count
        self.attempts = 0

    async def chat_completion(self, messages: list) -> dict:
        self.attempts += 1
        if self.attempts <= self.fail_count:
            return {"success": False, "error": "Simulated Failure", "latency": 0.1}
        
        import json
        return {
            "response": type('obj', (object,), {
                "choices": [type('obj', (object,), {
                    "message": type('obj', (object,), {
                        "content": json.dumps({"test": {"answer": "success"}})
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
    
    # Test 1: Succeeds after retries
    print("--- Test 1: Succeeds after 2 failures ---")
    client = FlakyLLMClient(api_key="dummy", fail_count=2)
    processor = BatchProcessor(
        schema=schema, 
        llm_client=client, 
        overwrite=True, 
        enable_prompt_logging=False,
        max_retries=3
    )
    result = await processor.process_note("retry_test_1", "note text")
    
    if result["success"]:
        print("Success! Errors encountered:", result.get("errors"))
    else:
        print("Failed unexpectedly:", result)

    # Test 2: Fails after max retries
    print("\n--- Test 2: Fails after max retries ---")
    client_fail = FlakyLLMClient(api_key="dummy", fail_count=5)
    processor_fail = BatchProcessor(
        schema=schema, 
        llm_client=client_fail, 
        overwrite=True, 
        enable_prompt_logging=False,
        max_retries=3
    )
    result_fail = await processor_fail.process_note("retry_test_2", "note text")
    
    if not result_fail["success"]:
        print("Correctly failed. Errors:", result_fail.get("errors"))
    else:
        print("Succeeded unexpectedly:", result_fail)

if __name__ == "__main__":
    asyncio.run(run_test())
