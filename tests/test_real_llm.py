import asyncio
import os
from typing import List, Optional
from pydantic import BaseModel, Field
from clinical_note_miner.schema import ExtractionSchema, ExtractionElement, FewShotExample
from clinical_note_miner.llm import LLMClient
from clinical_note_miner.pipeline import BatchProcessor

# Complex Schema Definitions
class TumorDimensions(BaseModel):
    length: float
    width: float
    depth: Optional[float] = None
    unit: str = "mm"

class LymphNodeStatus(BaseModel):
    total_nodes: int
    positive_nodes: int
    location: str

class Histology(BaseModel):
    type: str
    grade: int
    subtypes: List[str] = Field(default_factory=list)

# Test Configuration
BASE_URL = "https://vllm.cloud.trusted.e-infra.cz/v1"
API_KEY = "fr25Dhg78%_1aRp"
MODEL = "qwen3-coder"
# MODEL = "deepseek-r1"


async def run_real_test():
    print(f"Testing with model: {MODEL} at {BASE_URL}")
    
    # Define a complex schema
    schema = ExtractionSchema(elements=[
        ExtractionElement(
            name="tumor_size",
            description="Largest dimension of the primary tumor.",
            response_type=int,
            grounding=True,
            reasoning=True
        ),
        ExtractionElement(
            name="lymph_nodes",
            description="Lymph node status including total count and positive count.",
            response_type=LymphNodeStatus,
            grounding=True
        ),
        ExtractionElement(
            name="histology",
            description="Histological type and grade of the tumor.",
            response_type=Histology,
            reasoning=True
        ),
        ExtractionElement(
            name="margins",
            description="Status of surgical margins (Positive/Negative/Close).",
            response_type=str
        )
    ])

    # Complex Clinical Note
    note_text = """
    PATHOLOGY REPORT
    
    CLINICAL HISTORY: 65-year-old female with breast mass.
    
    GROSS DESCRIPTION:
    The specimen consists of a partial mastectomy weighing 45 grams.
    Sectioning reveals a firm, tan-white irregular mass measuring 2.5 x 1.8 x 1.2 cm.
    The mass is located 0.5 cm from the superior margin and 0.3 cm from the deep margin.
    
    MICROSCOPIC DIAGNOSIS:
    Invasive ductal carcinoma, Nottingham Grade 2 (tubules 3, nuclear 2, mitoses 1).
    Ductal carcinoma in situ (DCIS), cribriform type, is present.
    
    LYMPH NODES:
    Sentinel lymph node biopsy:
    - Node 1: Negative for malignancy (0/1).
    - Node 2: Positive for metastatic carcinoma (1/1), largest deposit 4mm.
    - Node 3: Negative (0/1).
    Total nodes examined: 3. Total positive: 1.
    
    MARGINS:
    Margins are negative for invasive carcinoma.
    Deep margin is close (<1mm) to DCIS.
    """

    # Initialize Client
    client = LLMClient(
        base_url=BASE_URL,
        api_key=API_KEY,
        model=MODEL
    )

    # Initialize Processor
    processor = BatchProcessor(
        schema=schema,
        llm_client=client,
        output_file="real_test_results.jsonl",
        overwrite=True,
        enable_file_output=False,
        chunk_metrics=True,
        chunk_reasoning=True,
    )

    # Process single note
    print("Sending request to LLM...")
    result = await processor.process_note("log_prompt_complex_case_1", note_text)
    
    if result["success"]:
        print("\nExtraction Successful!")
        print(f"Latency: {result['latency']:.2f}s")
        import json
        print(json.dumps(result, indent=2))
    else:
        print("\nExtraction Failed!")
        print(f"Error: {result.get('error')}")
        if "raw_response" in result:
            print(f"Raw Response: {result['raw_response']}")

if __name__ == "__main__":
    asyncio.run(run_real_test())
