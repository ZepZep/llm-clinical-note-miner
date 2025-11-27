import asyncio
import json
import time
from typing import AsyncGenerator, Generator, List, Optional, Tuple, Dict, Any
import jsonlines
from tqdm.auto import tqdm
from .schema import ExtractionSchema
from .llm import LLMClient
from .prompt_builder import PromptBuilder
from .matcher import find_matches

class BatchProcessor:
    def __init__(
        self,
        schema: ExtractionSchema,
        llm_client: LLMClient,
        max_parallel_requests: int = 5,
        output_file: Optional[str] = None,
        fuzzy_config: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
        enable_prompt_logging: bool = True,
        enable_file_output: bool = True,
        max_retries: int = 3,
        max_elements_per_request: Optional[int] = None,
        include_chunk_details: bool = True,
        chunk_reasoning: bool = True,
        chunk_metrics: bool = True,
        include_raw_response: bool = True
    ):
        self.schema = schema
        self.llm_client = llm_client
        self.semaphore = asyncio.Semaphore(max_parallel_requests)
        self.output_file = output_file
        self.prompt_builder = PromptBuilder(schema)
        self.fuzzy_config = fuzzy_config
        self.enable_prompt_logging = enable_prompt_logging
        self.enable_file_output = enable_file_output
        self.max_retries = max_retries
        self.max_elements_per_request = max_elements_per_request
        self.include_chunk_details = include_chunk_details
        self.chunk_reasoning = chunk_reasoning
        self.chunk_metrics = chunk_metrics
        self.include_raw_response = include_raw_response
        self.completed_ids = set()
        
        if self.output_file and self.enable_file_output:
            if overwrite:
                # Clear the file
                with open(self.output_file, 'w') as f:
                    pass
            else:
                self._load_completed_ids()

    def _load_completed_ids(self):
        try:
            with jsonlines.open(self.output_file) as reader:
                for obj in reader:
                    if "id" in obj:
                        self.completed_ids.add(obj["id"])
        except FileNotFoundError:
            pass

    async def _call_llm_with_retry(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        attempts = 0
        errors = []
        
        while attempts <= self.max_retries:
            try:
                result = await self.llm_client.chat_completion(messages)
                if result["success"]:
                    return {"success": True, "result": result, "errors": errors}
                else:
                    errors.append(f"Attempt {attempts + 1}: {result.get('error')}")
            except Exception as e:
                errors.append(f"Attempt {attempts + 1}: Exception {str(e)}")
            
            attempts += 1
            if attempts <= self.max_retries:
                await asyncio.sleep(1 * attempts) # Exponential backoffish
        
        return {"success": False, "errors": errors}

    def _parse_response(self, result: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        try:
            content = result["message"].content
            # Basic cleanup for markdown json blocks
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            
            return json.loads(content), None
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            return None, str(e)

    def _post_process_extraction(
        self, 
        raw_extraction: Dict[str, Any], 
        note_text: str, 
        element_names: Optional[List[str]]
    ) -> Dict[str, Any]:
        processed_extraction = {}
        target_elements = element_names if element_names else list(self.schema.elements.keys())
        
        for el_name in target_elements:
            el_def = self.schema.get_element(el_name)
            if not el_def:
                continue
                
            if el_name in raw_extraction:
                raw_val = raw_extraction[el_name]
                item_output = {}
                
                if isinstance(raw_val, dict) and ("answer" in raw_val or "value" in raw_val):
                    answer = raw_val.get("answer", raw_val.get("value"))
                    reasoning = raw_val.get("reasoning")
                    grounding_text = raw_val.get("grounding")
                else:
                    answer = raw_val
                    reasoning = None
                    grounding_text = None
                    
                item_output["answer"] = answer
                
                if el_def.reasoning and reasoning:
                    item_output["reasoning"] = reasoning
                    
                if el_def.grounding and grounding_text:
                    if isinstance(grounding_text, str):
                        grounding_text = [grounding_text]
                        
                    grounding_result = []
                    for text_snippet in grounding_text:
                        matches = find_matches(note_text, text_snippet, self.fuzzy_config)
                        anchors = [{"start": m["start"], "end": m["end"]} for m in matches] if matches else []
                        
                        grounding_result.append({
                            "text": text_snippet,
                            "anchors": anchors
                        })
                    item_output["grounding"] = grounding_result
                    
                processed_extraction[el_name] = item_output
                
        return processed_extraction

    async def process_note(
        self, 
        note_id: str, 
        note_text: str, 
        element_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        async with self.semaphore:
            # Determine target elements
            target_elements = element_names if element_names else list(self.schema.elements.keys())
            
            # Split into chunks if needed
            chunks = []
            if self.max_elements_per_request and self.max_elements_per_request > 0:
                for i in range(0, len(target_elements), self.max_elements_per_request):
                    chunks.append(target_elements[i:i + self.max_elements_per_request])
            else:
                chunks = [target_elements]
            
            merged_extraction = {}
            all_errors = []
            overall_success = True
            total_latency = 0.0
            raw_responses = []
            chunks_info = []
            total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            
            for i, chunk in enumerate(chunks):
                messages = self.prompt_builder.construct_messages(note_text, chunk)
                
                if self.enable_prompt_logging and "log_prompt" in note_id:
                    print(f"\n--- Generated Prompt for {note_id} (Chunk {i+1}/{len(chunks)}) ---\n")
                    for m in messages:
                        print(f"Role: {m['role']}\nContent:\n{m['content']}\n")
                    print("-----------------------------------\n")
                
                retry_result = await self._call_llm_with_retry(messages)
                all_errors.extend(retry_result.get("errors", []))
                
                chunk_data = {
                    "chunk_index": i,
                    "elements": chunk,
                    "success": retry_result["success"]
                }
                
                if retry_result["success"]:
                    llm_result = retry_result["result"]
                    latency = llm_result.get("latency", 0)
                    total_latency += latency
                    chunk_data["latency"] = latency
                    
                    usage = llm_result.get("usage")
                    if usage:
                        if self.chunk_metrics:
                            chunk_data["usage"] = usage
                        total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                        total_usage["completion_tokens"] += usage.get("completion_tokens", 0)
                        total_usage["total_tokens"] += usage.get("total_tokens", 0)
                    
                    raw_content = llm_result["message"].content
                    raw_responses.append(raw_content) # No longer aggregating top-level
                    
                    if self.include_raw_response:
                        chunk_data["raw_response"] = raw_content
                    
                    raw_extraction, parse_error = self._parse_response(llm_result)
                    
                    if self.chunk_reasoning:
                        reasoning_content = llm_result.get("reasoning_content")
                        if reasoning_content:
                            chunk_data["reasoning"] = reasoning_content
                    
                    if raw_extraction:
                        processed = self._post_process_extraction(raw_extraction, note_text, chunk)
                        merged_extraction.update(processed)
                    else:
                        overall_success = False
                        error_msg = f"Failed to parse JSON response for chunk {chunk}: {parse_error}"
                        all_errors.append(error_msg)
                        chunk_data["error"] = error_msg
                else:
                    overall_success = False
                    error_msg = f"Max retries exceeded for chunk {chunk}"
                    all_errors.append(error_msg)
                    chunk_data["error"] = error_msg
                
                chunks_info.append(chunk_data)
            
            output = {
                "id": note_id,
                "extraction": merged_extraction,
                "errors": all_errors,
                "success": overall_success,
                "latency": total_latency
            }
            
            if self.include_chunk_details:
                output["chunks"] = chunks_info
                
            if self.chunk_metrics:
                output["usage"] = total_usage
            
            if not overall_success and not output.get("error"):
                 output["error"] = "One or more chunks failed"
            
            if self.include_raw_response and raw_responses:
                 output["raw_response"] = "\n---\n".join(raw_responses)
                 
            # Write to file if configured
            if self.output_file and self.enable_file_output:
                with jsonlines.open(self.output_file, mode='a') as writer:
                    writer.write(output)
                    
            return output

    async def process_batch(
        self, 
        notes: AsyncGenerator[Tuple[str, str], None],
        element_names: Optional[List[str]] = None,
        total: Optional[int] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        
        pending_tasks = set()
        pbar = tqdm(total=total, desc="Processing Notes", unit="note")
        
        try:
            async for note_id, note_text in notes:
                if note_id in self.completed_ids:
                    pbar.update(1)
                    continue
                    
                # Create a new task
                task = asyncio.create_task(self.process_note(note_id, note_text, element_names))
                pending_tasks.add(task)
                
                # If we have enough tasks, wait for some to finish
                # We use semaphore for the actual API limit, but we don't want to queue up infinite tasks in memory
                if len(pending_tasks) >= self.semaphore._value * 2:
                    done, pending_tasks = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)
                    for t in done:
                        yield await t
                        pbar.update(1)
                        
            # Wait for remaining tasks
            while pending_tasks:
                done, pending_tasks = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)
                for t in done:
                    yield await t
                    pbar.update(1)
        finally:
            pbar.close()

    def process_batch_sync(
        self, 
        notes: List[Tuple[str, str]], 
        element_names: Optional[List[str]] = None,
        total: Optional[int] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Synchronous wrapper for process_batch. Works in both scripts and Jupyter notebooks.
        Handles interrupts gracefully.
        """
        import queue
        import threading
        
        result_queue = queue.Queue()
        stop_event = threading.Event()
        
        def run_async_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def async_wrapper():
                async def async_gen():
                    for item in notes:
                        if stop_event.is_set():
                            break
                        yield item
                        await asyncio.sleep(0)

                try:
                    # We need to be able to cancel this if stop_event is set
                    # But process_batch is a generator.
                    # We can check stop_event inside the loop
                    async for result in self.process_batch(async_gen(), element_names, total):
                        if stop_event.is_set():
                            break
                        result_queue.put(("item", result))
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    result_queue.put(("error", e))
                finally:
                    result_queue.put(("done", None))

            # Run until complete or stopped
            task = loop.create_task(async_wrapper())
            
            # We need to monitor stop_event from outside the loop? 
            # No, we can just run the loop. The main thread will set stop_event.
            # But loop.run_until_complete blocks. 
            # We can use a periodic check or just let the async_gen handle the break.
            # But if process_batch is waiting on network, async_gen won't be called.
            # So we need a way to cancel the task from the outside.
            
            def check_stop():
                if stop_event.is_set():
                    task.cancel()
                else:
                    loop.call_later(0.1, check_stop)
            
            loop.call_later(0.1, check_stop)
            
            try:
                loop.run_until_complete(task)
            except asyncio.CancelledError:
                pass
            finally:
                # Cancel all running tasks
                pending = asyncio.all_tasks(loop)
                for p in pending:
                    p.cancel()
                # Run loop to let tasks clean up
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                loop.close()

        t = threading.Thread(target=run_async_loop)
        t.start()

        try:
            while True:
                try:
                    # Poll with timeout to allow checking for KeyboardInterrupt
                    msg_type, payload = result_queue.get(timeout=0.1)
                    if msg_type == "item":
                        yield payload
                    elif msg_type == "error":
                        raise payload
                    elif msg_type == "done":
                        break
                except queue.Empty:
                    if not t.is_alive():
                        break
                    continue
        except KeyboardInterrupt:
            print("\nInterrupted! Stopping background tasks...")
            stop_event.set()
            # Wait for thread to finish cleanup
            t.join(timeout=5.0)
            raise
        
        t.join()
