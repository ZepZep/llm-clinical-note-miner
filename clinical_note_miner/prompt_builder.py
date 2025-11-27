import jsonyx as json
from typing import List, Optional
from .schema import ExtractionSchema, ExtractionElement

class PromptBuilder:
    def __init__(self, schema: ExtractionSchema):
        self.schema = schema

    def build_system_message(self) -> str:
        return (
            "You are an expert clinical data extractor. "
            "Your task is to extract structured information from clinical notes based on the provided schema. "
            "You must return the output in valid JSON format."
        )

    def build_user_message(
        self, 
        note_text: str, 
        element_names: Optional[List[str]] = None
    ) -> str:
        if element_names:
            elements = [self.schema.get_element(name) for name in element_names]
            elements = [e for e in elements if e]
        else:
            elements = list(self.schema.elements.values())

        # Construct the schema definition for the prompt
        prompt = "Extract the following information from the clinical note:\n\n"
        empty_template = {}
        
        from pydantic import BaseModel

        def generate_template(type_obj, current_depth=0, max_depth=5):
            if current_depth > max_depth:
                return "<recursion_limit_reached>"
            
            # Handle List[T]
            origin = getattr(type_obj, "__origin__", None)
            if origin is list or origin is List:
                args = getattr(type_obj, "__args__", [])
                if args:
                    inner_type = args[0]
                    return [
                        generate_template(inner_type, current_depth + 1, max_depth)
                    ]
                return ["<list_item>"]

            # Handle Optional[T]
            if origin is Optional:
                 args = getattr(type_obj, "__args__", [])
                 if args:
                     inner_type = args[0]
                     return generate_template(inner_type, current_depth + 1, max_depth)
                 return "<optional_value>"

            # Handle Pydantic Model
            if isinstance(type_obj, type) and issubclass(type_obj, BaseModel):
                template = {}
                for name, field in type_obj.model_fields.items():
                    field_type = field.annotation
                    template[name] = generate_template(field_type, current_depth + 1, max_depth)
                return template

            # Simple type
            if hasattr(type_obj, "__name__"):
                return f"<{type_obj.__name__}_value>"
            
            return f"<{str(type_obj)}_value>"

        for el in elements:
            # Format type nicely
            is_pydantic = False
            if isinstance(el.response_type, type) and issubclass(el.response_type, BaseModel):
                is_pydantic = True
                type_name = el.response_type.__name__
            elif isinstance(el.response_type, type):
                type_name = el.response_type.__name__
            else:
                type_name = str(type(el.response_type).__name__)
                
            # Readable definition
            prompt += f"- **{el.name}** ({type_name}): {el.description}\n"
            
            if is_pydantic:
                # Use Pydantic's JSON schema dump
                schema = el.response_type.model_json_schema()
                prompt += f"  - Schema: {json.dumps(schema)}\n"

            if el.grounding:
                prompt += "  - *Grounding required*: Include exact text snippets.\n"
            if el.reasoning:
                prompt += "  - *Reasoning required*: Provide reasoning for the answer.\n"
            
            # Construct empty template
            item_template = {}
            if el.grounding:
                item_template["grounding"] = ["<text_snippet>"]
            if el.reasoning:
                item_template["reasoning"] = "<explanation>"
            
            item_template["answer"] = generate_template(el.response_type)
            empty_template[el.name] = item_template
            
        prompt += "\n"
        
        # Add few-shot examples if available
        examples_str = ""
        has_examples = False
        
        if self.schema.examples:
            has_examples = True
            for ex in self.schema.examples:
                # Construct example output
                example_output = {}
                
                for el_name, extraction in ex.extractions.items():
                    # Only include if element is in the current request
                    if element_names and el_name not in element_names:
                        continue
                        
                    # Verify element exists in schema
                    el_def = self.schema.get_element(el_name)
                    if not el_def:
                        continue

                    item_output = {}
                    if el_def.grounding and extraction.grounding:
                        item_output["grounding"] = extraction.grounding
                    if el_def.reasoning and extraction.reasoning:
                        item_output["reasoning"] = extraction.reasoning
                    
                    item_output["answer"] = extraction.answer
                    example_output[el_name] = item_output
                
                if example_output:
                    examples_str += f"Input Note: {ex.note_text}\nOutput: {json.dumps(example_output, indent=1, indent_leaves=False, ensure_ascii=False)}\n\n"
        
        if has_examples:
            prompt += "Examples:\n" + examples_str
        else:
            # Show empty expected JSON if no examples
            prompt += "Expected Output Format:\n" + json.dumps(empty_template, indent=1, indent_leaves=False, ensure_ascii=False) + "\n\n"
            
        prompt += f"Clinical Note:\n{note_text}\n\n"
        prompt += "Output JSON:"
        
        return prompt

    def construct_messages(self, note_text: str, element_names: Optional[List[str]] = None) -> List[dict]:
        return [
            {"role": "system", "content": self.build_system_message()},
            {"role": "user", "content": self.build_user_message(note_text, element_names)}
        ]
