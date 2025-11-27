from clinical_note_miner import ExtractionSchema, ExtractionElement, LLMClient, BatchProcessor
import asyncio
import json

# Mock LLM that returns dummy data for requested elements
class MockLLMClient(LLMClient):
    def __init__(self, api_key):
        super().__init__(api_key=api_key)
        self.call_count = 0
        self.requested_elements = []

    async def chat_completion(self, messages: list) -> dict:
        self.call_count += 1
        user_msg = messages[1]["content"]
        
        # Extract requested elements from prompt (heuristic)
        # Assuming prompt format: "- **element_name**"
        import re
        elements = re.findall(r"- \*\*(.*?)\*\*", user_msg)
        self.requested_elements.append(elements)
        
        response_data = {}
        for el in elements:
            response_data[el] = {"answer": f"value_for_{el}"}
            
        return {
            "response": type('obj', (object,), {
                "choices": [type('obj', (object,), {
                    "message": type('obj', (object,), {
                        "content": json.dumps(response_data)
                    })
                })]
            }),
            "latency": 0.1,
            "success": True
        }

async def run_test():
    schema = ExtractionSchema(elements=[
        ExtractionElement(name="el1", description="desc", response_type=str),
        ExtractionElement(name="el2", description="desc", response_type=str),
        ExtractionElement(name="el3", description="desc", response_type=str),
        ExtractionElement(name="el4", description="desc", response_type=str),
    ])
    
    client = MockLLMClient(api_key="dummy")
    
    # Batch size 2 -> should result in 2 calls (el1, el2) and (el3, el4)
    processor = BatchProcessor(
        schema=schema, 
        llm_client=client, 
        overwrite=True,
        enable_prompt_logging=False,
        enable_file_output=False,
        max_elements_per_request=2
    )
    
    print("Processing note with 4 elements, batch size 2...")
    result = await processor.process_note("id_1", "text")
    
    print(f"Success: {result['success']}")
    print(f"LLM Call Count: {client.call_count}")
    print(f"Requested Elements per call: {client.requested_elements}")
    print(f"Extracted keys: {list(result['extraction'].keys())}")
    
    if client.call_count == 2 and len(result['extraction']) == 4:
        print("Test Passed: Correctly split into 2 requests.")
    else:
        print("Test Failed.")

if __name__ == "__main__":
    asyncio.run(run_test())
