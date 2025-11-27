try:
    from clinical_note_miner import (
        ExtractionSchema, 
        ExtractionElement, 
        FewShotExample, 
        LLMClient, 
        BatchProcessor
    )
    print("Imports successful!")
except ImportError as e:
    print(f"Import failed: {e}")
    exit(1)
