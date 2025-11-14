import asyncio
import aiohttp
import requests
from aiolimiter import AsyncLimiter
from tqdm.asyncio import tqdm_asyncio
from tqdm.auto import tqdm


class VLLM_Custom:
    def __init__(self, name, api_base: str, api_key: str = None):
        """Initialize the VLLM_Custom client with basic configurations.
        
        Args:
            name: Model name
            api_base: Complete URL for the vLLM endpoint (e.g., "http://localhost:8000/v1/chat/completions")
            api_key: Bearer token for authentication
        """
        self.name = name
        self.api_base = api_base.rstrip('/')
        self.api_key = api_key

    def validate_litellm(self):
        """Validate the vLLM endpoint is reachable."""
        try:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            response = requests.get(f"{self.api_base}/health", headers=headers, timeout=5)
            return response.status_code == 200
        except:
            return False

    def completions(self, messages, **kwargs):
        """Generate completions for a list of messages using synchronous batch processing.
        
        Args:
            messages: List of list of dictionaries containing messages
            **kwargs: Additional parameters for the completion API
        
        Returns:
            List of completion strings
        """
        # Validate input format
        assert isinstance(messages, list), "Messages must be a list"
        assert all(
            isinstance(message, list) for message in messages
        ), "Each message must be a list of dictionaries"
        assert all(
            isinstance(msg, dict) and "role" in msg and "content" in msg
            for message in messages
            for msg in message
        ), "Message format error: each message must have 'role' and 'content'"

        result_responses = []
        
        # Extract use_tqdm if provided
        use_tqdm = kwargs.pop('use_tqdm', True)
        
        # Prepare default parameters
        payload_defaults = {
            "model": self.name,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 512),
            "top_p": kwargs.get("top_p", 1.0),
        }
        
        # Add any additional kwargs that aren't in defaults
        for k, v in kwargs.items():
            if k not in payload_defaults:
                payload_defaults[k] = v

        iterator = tqdm(messages, desc="Processing") if use_tqdm else messages
        
        # Prepare headers with authentication
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        for message in iterator:
            payload = {
                **payload_defaults,
                "messages": message,
            }
            
            try:
                response = requests.post(
                    self.api_base,
                    json=payload,
                    headers=headers,
                    timeout=60
                )
                response.raise_for_status()
                result = response.json()
                content = result["choices"][0]["message"]["content"].strip()
                result_responses.append(content)
            except requests.exceptions.HTTPError as e:
                print(f"HTTP Error during VLLM API call: {e}")
                print(f"Response: {response.text}")
                result_responses.append("")
            except Exception as e:
                print(f"Error during VLLM API call: {e}")
                result_responses.append("")

        return result_responses


class AsyncVLLM_Custom:
    def __init__(
        self,
        name,
        api_base: str,
        api_key: str = None,
        batch_size: int = 100,
        requests_per_minute: int = 100,
    ):
        """Initialize the AsyncVLLM_Custom client with basic configurations.
        
        Args:
            name: Model name
            api_base: Complete URL for the vLLM endpoint (e.g., "http://localhost:8000/v1/chat/completions")
            api_key: Bearer token for authentication
            batch_size: Number of requests to process in parallel
            requests_per_minute: Rate limit for requests
        """
        self.name = name
        self.api_base = api_base.rstrip('/')
        self.api_key = api_key
        self.batch_size = batch_size
        self.requests_per_minute = requests_per_minute
        self.limiter = AsyncLimiter(self.requests_per_minute, 60)

    async def validate_litellm(self):
        """Validate the vLLM endpoint is reachable asynchronously."""
        return True

    async def _get_completion_text_async(self, message, session, **kwargs):
        """Fetch completion text for a single message asynchronously.
        
        Args:
            message: List of message dictionaries
            session: aiohttp ClientSession
            **kwargs: Additional parameters for the completion API
        
        Returns:
            Completion string
        """
        async with self.limiter:
            payload = {
                "model": self.name,
                "messages": message,
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 512),
                "top_p": kwargs.get("top_p", 1.0),
            }
            
            # Add any additional kwargs
            for k, v in kwargs.items():
                if k not in payload:
                    payload[k] = v
            
            # Prepare headers with authentication
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            try:
                async with session.post(
                    self.api_base,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    return result["choices"][0]["message"]["content"].strip()
            except aiohttp.ClientResponseError as e:
                response_text = await response.text() if response else "No response"
                print(f"HTTP Error during VLLM API call: {e}")
                print(f"Response: {response_text}")
                return ""
            except Exception as e:
                print(f"Error during VLLM API call: {e}")
                return ""

    async def completions(self, messages, **kwargs):
        """Generate completions for a list of messages using asynchronous batch processing.
        
        Args:
            messages: List of list of dictionaries containing messages
            **kwargs: Additional parameters for the completion API
        
        Returns:
            List of completion strings
        """
        # Validate input format
        assert isinstance(messages, list), "Messages must be a list"
        assert all(
            isinstance(message, list) for message in messages
        ), "Each message must be a list of dictionaries"
        assert all(
            isinstance(msg, dict) and "role" in msg and "content" in msg
            for message in messages
            for msg in message
        ), "Message format error: each message must have 'role' and 'content'"

        result_responses = []

        # Create a single session for all requests
        async with aiohttp.ClientSession() as session:
            # Process the messages in batches with progress visualization
            for start_idx in tqdm(
                range(0, len(messages), self.batch_size), 
                desc="Processing batches"
            ):
                end_idx = start_idx + self.batch_size
                batch_prompts = messages[start_idx:end_idx]

                # Fetch responses for all prompts in the current batch asynchronously
                batch_responses = await tqdm_asyncio.gather(
                    *[
                        self._get_completion_text_async(prompt, session, **kwargs)
                        for prompt in batch_prompts
                    ]
                )
                result_responses.extend(batch_responses)

        return result_responses


# Test function
async def _test_async():
    """Test function for AsyncVLLM_Custom."""
    model = AsyncVLLM_Custom(
        name="param-1 (PT2: SFT V3)",
        api_base="https://api.bharatgen.dev/v1/chat/completions",
        api_key="bharatgen-secret-token-123",  # Optional
        batch_size=5,
        requests_per_minute=60
    )
    
    batch_messages = [
        [{"role": "user", "content": f"Hello, this is test message {i}"}]
        for i in range(10)
    ]

    responses = await model.completions(batch_messages)
    print(responses)
    return responses


def _test_sync():
    """Test function for VLLM_Custom."""
    model = VLLM_Custom(
        name="param-1 (PT2: SFT V3)",
        api_base="https://api.bharatgen.dev/v1/chat/completions",
        api_key="bharatgen-secret-token-123"  # Optional
    )
    
    batch_messages = [
        [{"role": "user", "content": f"Hello, this is test message {i}"}]
        for i in range(5)
    ]

    responses = model.completions(batch_messages)
    print(responses)
    return responses


if __name__ == "__main__":
    print("Testing synchronous VLLM_Custom:")
    _test_sync()
    
    print("\nTesting asynchronous VLLM_Custom:")
    asyncio.run(_test_async())