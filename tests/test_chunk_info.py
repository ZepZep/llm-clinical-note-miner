from clinical_note_miner import ExtractionSchema, ExtractionElement, LLMClient, BatchProcessor
import asyncio
import json

# Mock LLM with usage and raw reasoning
class MockLLMClient(LLMClient):
    async def chat_completion(self, messages: list) -> dict:
        import json
        content = json.dumps({"test": {"answer": "val"}})
        
        # Mock message with reasoning_content
        message_mock = type('obj', (object,), {
            "content": content,
            "reasoning_content": "Here is my reasoning"
        })
        
        return {
            "message": message_mock,
            "latency": 0.1,
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "reasoning_content": "Here is my reasoning",
            "success": True
        }

async def run_test():
    schema = ExtractionSchema(elements=[
        ExtractionElement(name="test", description="test", response_type=str)
    ])
    client = MockLLMClient(api_key="dummy")
    
    # Test 1: Defaults (include chunks, chunk_metrics, chunk_reasoning)
    print("--- Test 1: Defaults ---")
    processor = BatchProcessor(
        schema=schema, 
        llm_client=client, 
        overwrite=True,
        enable_prompt_logging=False,
        enable_file_output=False
    )
    result = await processor.process_note("id_1", "text")
    
    chunk_info = result['chunks'][0]
    print(f"Has chunks: {'chunks' in result}")
    print(f"Has usage in chunk: {'usage' in chunk_info}")
    print(f"Has reasoning in chunk: {'reasoning' in chunk_info}")
    print(f"Has raw_response in chunk: {'raw_response' in chunk_info}")
    print(f"Reasoning content: '{chunk_info.get('reasoning')}'")
    
    if 'chunks' in result and 'usage' in chunk_info and 'reasoning' in chunk_info and 'raw_response' in chunk_info:
        print("Test 1 Passed.")
    else:
        print("Test 1 Failed.")

    # Test 2: Disable chunk metrics and reasoning
    print("\n--- Test 2: Disable Chunk Info ---")
    processor_lean = BatchProcessor(
        schema=schema, 
        llm_client=client, 
        overwrite=True,
        enable_prompt_logging=False,
        enable_file_output=False,
        include_chunk_details=True,
        chunk_metrics=False,
        chunk_reasoning=False,
        chunk_raw_response=False
    )
    result_lean = await processor_lean.process_note("id_2", "text")
    
    chunk_info_lean = result_lean['chunks'][0]
    print(f"Has chunks: {'chunks' in result_lean}")
    print(f"Has usage in chunk: {'usage' in chunk_info_lean}")
    print(f"Has reasoning in chunk: {'reasoning' in chunk_info_lean}")
    print(f"Has raw_response in chunk: {'raw_response' in chunk_info_lean}")
    
    if 'chunks' in result_lean and 'usage' not in chunk_info_lean and 'reasoning' not in chunk_info_lean and 'raw_response' not in chunk_info_lean:
        print("Test 2 Passed.")
    else:
        print("Test 2 Failed.")

if __name__ == "__main__":
    asyncio.run(run_test())
