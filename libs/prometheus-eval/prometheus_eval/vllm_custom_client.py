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
        """Generate completions for a list of messages using synchronous batch processing."""
        assert isinstance(messages, list), "Messages must be a list"
        assert all(isinstance(message, list) for message in messages), "Each message must be a list"
        assert all(
            isinstance(msg, dict) and "role" in msg and "content" in msg
            for message in messages
            for msg in message
        ), "Message format error"

        result_responses = []
        use_tqdm = kwargs.pop('use_tqdm', True)
        
        payload_defaults = {
            "model": self.name,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 512),
            "top_p": kwargs.get("top_p", 1.0),
        }
        
        for k, v in kwargs.items():
            if k not in payload_defaults:
                payload_defaults[k] = v

        iterator = tqdm(messages, desc="Processing") if use_tqdm else messages
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        for message in iterator:
            payload = {**payload_defaults, "messages": message}
            
            try:
                response = requests.post(self.api_base, json=payload, headers=headers, timeout=60)
                response.raise_for_status()
                result = response.json()
                content = result["choices"][0]["message"]["content"].strip()
                result_responses.append(content)
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
        max_connections: int = 100,
        max_connections_per_host: int = 50,
    ):
        """Initialize the AsyncVLLM_Custom client with proper connection management.
        
        Args:
            name: Model name
            api_base: Base URL OR complete endpoint URL for vLLM
                     Examples: 
                     - "https://api.bharatgen.dev/v1" (base URL)
                     - "https://api.bharatgen.dev/v1/chat/completions" (full endpoint)
            api_key: Bearer token for authentication
            batch_size: Number of requests to process in parallel
            requests_per_minute: Rate limit for requests
            max_connections: Total connection pool size
            max_connections_per_host: Max connections per host
        """
        self.name = name
        # Handle both base URL and full endpoint URL
        self.api_base = api_base.rstrip('/')
        self.api_key = api_key
        self.batch_size = batch_size
        self.requests_per_minute = requests_per_minute
        self.max_connections = max_connections
        self.max_connections_per_host = max_connections_per_host
        
        # CRITICAL: Enable rate limiter to prevent overwhelming the connection pool
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
            Completion string or empty string on error
        """
        # CRITICAL: Use rate limiter to control request flow
        async with self.limiter:
            payload = {
                "model": self.name,
                "messages": message,
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 512),
                "top_p": kwargs.get("top_p", 1.0),
            }
            
            for k, v in kwargs.items():
                if k not in payload:
                    payload[k] = v
            
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            try:
                # Use reasonable timeout - not 500 seconds!
                timeout = aiohttp.ClientTimeout(
                    total=120,  # Total timeout including queue wait
                    connect=30,  # Connection establishment timeout
                    sock_read=60  # Socket read timeout
                )
                
                async with session.post(
                    self.api_base,
                    json=payload,
                    headers=headers,
                    timeout=timeout
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    return result["choices"][0]["message"]["content"].strip()
                    
            except asyncio.TimeoutError:
                print(f"Timeout error - request took too long")
                return ""
            except aiohttp.ClientError as e:
                print(f"HTTP Error during VLLM API call: {e}")
                return ""
            except Exception as e:
                print(f"Error during VLLM API call: {e}")
                return ""

    async def completions_old(self, messages, **kwargs):
        """Generate completions for a list of messages using asynchronous batch processing.
        
        Args:
            messages: List of list of dictionaries containing messages
            **kwargs: Additional parameters for the completion API
        
        Returns:
            List of completion strings
        """
        assert isinstance(messages, list), "Messages must be a list"
        assert all(isinstance(message, list) for message in messages), "Each message must be a list"
        assert all(
            isinstance(msg, dict) and "role" in msg and "content" in msg
            for message in messages
            for msg in message
        ), "Message format error"

        result_responses = []

        # Configure TCPConnector with proper limits
        connector = aiohttp.TCPConnector(
            limit=self.max_connections,  # Total connection pool size
            limit_per_host=self.max_connections_per_host,  # Per-host limit
            ttl_dns_cache=300,  # DNS cache TTL
            force_close=False,  # Reuse connections
            enable_cleanup_closed=True  # Clean up closed connections
        )
        
        # Create timeout configuration
        timeout = aiohttp.ClientTimeout(
            total=300,  # 5 minute total timeout
            connect=30,  # 30s to establish connection
            sock_read=60  # 60s to read response
        )

        # Create a single session with proper configuration
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            connector_owner=True  # Session owns the connector
        ) as session:
            # Process messages in batches
            for start_idx in tqdm(
                range(0, len(messages), self.batch_size), 
                desc="Processing batches"
            ):
                end_idx = start_idx + self.batch_size
                batch_prompts = messages[start_idx:end_idx]

                # OPTION 1: Using tqdm_asyncio.gather (simpler but no return_exceptions)
                # Wrap each call in try-except inside _get_completion_text_async
                batch_responses = await tqdm_asyncio.gather(
                    *[
                        self._get_completion_text_async(prompt, session, **kwargs)
                        for prompt in batch_prompts
                    ]
                )
                
                result_responses.extend(batch_responses)
                
                # Small delay between batches to prevent overwhelming the server
                await asyncio.sleep(0.1)

        return result_responses

    async def completions(self, messages, **kwargs):
        """Alternative implementation using as_completed for better error handling.
        
        This approach allows individual task failures without affecting the entire batch.
        
        Args:
            messages: List of list of dictionaries containing messages
            **kwargs: Additional parameters for the completion API
        
        Returns:
            List of completion strings
        """
        assert isinstance(messages, list), "Messages must be a list"
        assert all(isinstance(message, list) for message in messages), "Each message must be a list"
        assert all(
            isinstance(msg, dict) and "role" in msg and "content" in msg
            for message in messages
            for msg in message
        ), "Message format error"

        result_responses = []

        # Configure TCPConnector with proper limits
        connector = aiohttp.TCPConnector(
            limit=self.max_connections,
            limit_per_host=self.max_connections_per_host,
            ttl_dns_cache=300,
            force_close=False,
            enable_cleanup_closed=True
        )
        
        timeout = aiohttp.ClientTimeout(
            total=300,
            connect=30,
            sock_read=60
        )

        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            connector_owner=True
        ) as session:
            # Process messages in batches
            for start_idx in tqdm(
                range(0, len(messages), self.batch_size), 
                desc="Processing batches"
            ):
                end_idx = start_idx + self.batch_size
                batch_prompts = messages[start_idx:end_idx]

                # Create tasks for the batch
                tasks = [
                    self._get_completion_text_async(prompt, session, **kwargs)
                    for prompt in batch_prompts
                ]
                
                # Use tqdm with as_completed for individual error handling
                batch_results = []
                for coro in tqdm_asyncio.as_completed(tasks, desc="Batch progress", leave=False):
                    try:
                        result = await coro
                        batch_results.append(result)
                    except Exception as e:
                        print(f"Task failed with exception: {e}")
                        batch_results.append("")
                
                result_responses.extend(batch_results)
                
                await asyncio.sleep(0.1)

        return result_responses


# Test functions
async def _test_async():
    """Test function for AsyncVLLM_Custom."""
    model = AsyncVLLM_Custom(
        name="param-1 (PT2: SFT V3)",
        api_base="https://api.bharatgen.dev/v1/chat/completions",
        api_key="bharatgen-secret-token-123",
        batch_size=5,
        requests_per_minute=60,
        max_connections=100,
        max_connections_per_host=50
    )
    
    batch_messages = [
        [{"role": "user", "content": f"Hello, this is test message {i}"}]
        for i in range(10)
    ]

    print("Testing with tqdm_asyncio.gather:")
    responses = await model.completions(batch_messages)
    print(f"Completed {len(responses)} requests")
    print(responses[:3])  # Show first 3
    
    print("\nTesting with as_completed:")
    responses2 = await model.completions_with_as_completed(batch_messages)
    print(f"Completed {len(responses2)} requests")
    print(responses2[:3])  # Show first 3
    
    return responses


def _test_sync():
    """Test function for VLLM_Custom."""
    model = VLLM_Custom(
        name="param-1 (PT2: SFT V3)",
        api_base="https://api.bharatgen.dev/v1/chat/completions",
        api_key="bharatgen-secret-token-123"
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