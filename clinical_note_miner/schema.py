from typing import Any, List, Optional, Type, Union, Dict
from pydantic import BaseModel, Field, model_validator

class ExtractionOutput(BaseModel):
    answer: Any
    reasoning: Optional[str] = None
    grounding: Optional[List[str]] = None

class FewShotExample(BaseModel):
    note_text: str
    extractions: Dict[str, ExtractionOutput]

class ExtractionElement(BaseModel):
    name: str
    description: str
    response_type: Union[Type, Any] = str # Can be a python type or a Pydantic model
    grounding: bool = False
    reasoning: bool = False
    
    class Config:
        arbitrary_types_allowed = True

class ExtractionSchema(BaseModel):
    elements: Dict[str, ExtractionElement] = Field(default_factory=dict)
    examples: List[FewShotExample] = Field(default_factory=list)

    @model_validator(mode='before')
    @classmethod
    def validate_elements(cls, data: Any) -> Any:
        if isinstance(data, dict):
            elements_input = data.get('elements')
            if isinstance(elements_input, list):
                new_elements = {}
                for el in elements_input:
                    # Handle if el is dict or object
                    if isinstance(el, dict):
                        name = el.get('name')
                    else:
                        name = getattr(el, 'name', None)
                    
                    if not name:
                        raise ValueError(f"Element missing name: {el}")
                    
                    if name in new_elements:
                        raise ValueError(f"Duplicate element name found: {name}")
                    
                    new_elements[name] = el
                data['elements'] = new_elements
        return data

    def get_element(self, name: str) -> Optional[ExtractionElement]:
        return self.elements.get(name)
