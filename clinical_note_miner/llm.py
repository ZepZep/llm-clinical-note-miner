import time
from typing import Any, Dict, Optional
from openai import AsyncOpenAI

class LLMClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        init_kwargs: Optional[Dict[str, Any]] = None,
        completion_kwargs: Optional[Dict[str, Any]] = None
    ):
        self.init_kwargs = init_kwargs or {}
        self.completion_kwargs = completion_kwargs or {}
        
        if base_url:
            self.init_kwargs['base_url'] = base_url
        if api_key:
            self.init_kwargs['api_key'] = api_key
            
        self.model = model
        self.client = AsyncOpenAI(**self.init_kwargs)

    async def chat_completion(self, messages: list) -> Dict[str, Any]:
        start_time = time.time()
        try:
            kwargs = self.completion_kwargs.copy()
            if self.model and 'model' not in kwargs:
                kwargs['model'] = self.model
                
            response = await self.client.chat.completions.create(
                messages=messages,
                **kwargs
            )
            end_time = time.time()
            
            message = response.choices[0].message
            reasoning_content = getattr(message, "reasoning_content", None)
            
            return {
                "message": message,
                "latency": end_time - start_time,
                "usage": response.usage.model_dump() if response.usage else None,
                "reasoning_content": reasoning_content,
                "success": True
            }
        except Exception as e:
            end_time = time.time()
            return {
                "error": str(e),
                "latency": end_time - start_time,
                "success": False
            }
