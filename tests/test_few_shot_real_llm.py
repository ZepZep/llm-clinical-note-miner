import asyncio
import json
from clinical_note_miner.schema import ExtractionSchema, ExtractionElement, FewShotExample, ExtractionOutput
from clinical_note_miner.llm import LLMClient
from clinical_note_miner.pipeline import BatchProcessor

async def run_test():
    # Define schema with 2-shot examples at schema level
    schema = ExtractionSchema(
        elements=[
            ExtractionElement(
                name="smoking_status",
                description="Patient's smoking status (Current/Former/Never/Unknown)",
                response_type=str,
                grounding=True
            ),
            ExtractionElement(
                name="pack_years",
                description="Number of pack-years if smoker",
                response_type=int,
                grounding=True
            )
        ],
        examples=[
            FewShotExample(
                note_text="Patient denies tobacco use. Drinks alcohol occasionally.",
                extractions={
                    "smoking_status": {
                        "answer": "Never",
                        "grounding": ["denies tobacco use"]
                    },
                    "pack_years": {
                        "answer": 0,
                        "grounding": ["denies tobacco use"]
                    }
                }
            ),
            FewShotExample(
                note_text="Social history: 1ppd smoker for 20 years, quit 5 years ago.",
                extractions={
                    "smoking_status": {
                        "answer": "Former",
                        "grounding": ["quit 5 years ago"]
                    },
                    "pack_years": {
                        "answer": 20,
                        "grounding": ["1ppd smoker for 20 years"]
                    }
                }
            )
        ]
    )

    # Initialize LLM Client
    client = LLMClient(
        base_url="https://vllm.cloud.trusted.e-infra.cz/v1",
        api_key="fr25Dhg78%_1aRp",
        model="qwen3-coder"
    )

    # Initialize Pipeline
    processor = BatchProcessor(
        schema=schema, 
        llm_client=client,
        max_parallel_requests=1,
        enable_file_output=False
    )

    # Test Note
    note_text = """
    SOCIAL HISTORY:
    The patient is a current smoker, smokes about half a pack per day for the last 10 years.
    Does not drink alcohol.
    """
    
    # Use "log_prompt" in ID to trigger printing
    print("Sending request to LLM with 2-shot examples...")
    result = await processor.process_note("log_prompt_few_shot_test", note_text)
    
    if result["success"]:
        print("\nExtraction Successful!")
        print(f"Latency: {result['latency']:.2f}s")
        print(json.dumps(result["extraction"], indent=2))
    else:
        print("\nExtraction Failed!")
        print(result.get("error"))
        if "raw_response" in result:
            print("Raw Response:", result["raw_response"])

if __name__ == "__main__":
    asyncio.run(run_test())
