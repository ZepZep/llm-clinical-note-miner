from clinical_note_miner import ExtractionSchema, ExtractionElement, FewShotExample
from clinical_note_miner.prompt_builder import PromptBuilder
from typing import List
from pydantic import BaseModel

class Nested(BaseModel):
    field: str

class ComplexType(BaseModel):
    items: List[str]
    nested_list: List[Nested]

def test_prompt_formatting():
    # Schema with complex types and examples
    schema = ExtractionSchema(elements=[
        ExtractionElement(name="simple_list", description="List of strings", response_type=List[str]),
        ExtractionElement(name="complex", description="Complex type", response_type=ComplexType)
    ])
    
    # Add example with unicode
    schema.examples = [
        FewShotExample(
            note_text="Pacient má horečku.",
            extractions={
                "simple_list": {"answer": ["teplota", "horečka"]},
                "complex": {"answer": {"items": ["a", "b"], "nested_list": [{"field": "hodnota"}]}}
            }
        )
    ]
    
    builder = PromptBuilder(schema)
    prompt = builder.build_user_message("Note text")
    
    print("--- Prompt with Examples ---")
    print(prompt)
    
    if "horečka" in prompt and "hodnota" in prompt:
        print("\nSUCCESS: Unicode characters preserved.")
    else:
        print("\nFAILURE: Unicode characters mangled.")
        
    if '[\n    "teplota",' in prompt or '[\n      "teplota",' in prompt:
         print("SUCCESS: Indented JSON used.")
    else:
         print("FAILURE: JSON not indented correctly.")

    # Test dummy data generation (no examples)
    schema_no_ex = ExtractionSchema(elements=[
        ExtractionElement(name="simple_list", description="List of strings", response_type=List[str]),
        ExtractionElement(name="complex", description="Complex type", response_type=ComplexType),
        ExtractionElement(name="some_values", description="list of complex types", response_type=list[ComplexType])
    ])
    builder_no_ex = PromptBuilder(schema_no_ex)
    prompt_no_ex = builder_no_ex.build_user_message("Note text")
    
    print("\n--- Prompt without Examples (Dummy Data) ---")
    print(prompt_no_ex)
    
    if "<str_value>" in prompt_no_ex:
        print("SUCCESS: List dummy data generated with multiple items.")
    else:
        print("FAILURE: List dummy data missing multiple items.")

    if '"some_values": {' in prompt_no_ex and '"items":' in prompt_no_ex:
        print("SUCCESS: List of ComplexType dummy data generated.")
    else:
        print("FAILURE: List of ComplexType dummy data missing.")

if __name__ == "__main__":
    test_prompt_formatting()
