from typing import List, Dict
import time, json, os, requests
import random

# Optional Together AI dependency - only fails if actually used
try:
    from together import Together
    import httpx
    TOGETHER_AVAILABLE = True
except ImportError:
    TOGETHER_AVAILABLE = False
    Together = None
    httpx = None


keys = []       # Add your togetherAI API keys here
keys_file = "keys/mine.txt"
if os.path.exists(keys_file):
    with open(keys_file) as f:
        keys = f.read().splitlines()
elif TOGETHER_AVAILABLE:
    # Only warn if Together is installed but keys missing
    import warnings
    warnings.warn(
        f"Together AI keys not found at {keys_file}. "
        "Together API functions will fail if called. "
        "To use Together AI: (1) pip install together, (2) create keys/mine.txt with API keys."
    )
key_cnt = 0


def _get_key():
    global key_cnt
    if not TOGETHER_AVAILABLE:
        raise ImportError(
            "Together AI package not installed. Install with: pip install together"
        )
    if not keys:
        raise ValueError(
            f"No Together AI API keys loaded. Create {keys_file} with your API keys (one per line)."
        )
    key = keys[key_cnt]
    key_cnt = (key_cnt + 1) % len(keys)
    return key


def _create_together_client_with_timeout():
    """Create Together client with proper timeout configuration."""
    # Together client accepts a simple float timeout (in seconds)
    # This applies to the entire request (connect + read)
    return Together(api_key=_get_key(), timeout=120.0)


def text_completion(prompt, model_ckpt, max_tokens=256, temperature=0.8, top_k=40, top_p=0.95, repetition_penalty=1, stop=None, max_retries=10):
    """Text completion with exponential backoff retry logic."""

    for attempt in range(max_retries):
        try:
            client = _create_together_client_with_timeout()
            response = client.completions.create(
                model=model_ckpt,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                stop=stop,
            )
            return response.choices[0].text

        except Exception as e:
            error_str = str(e).lower()

            # Check if it's a timeout error
            if "timeout" in error_str or "timed out" in error_str:
                wait_time = (2 ** attempt) + (random.random() * 0.1)  # Exponential backoff with jitter
                print(f"Together AI API timeout (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time:.1f}s...")
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                else:
                    raise Exception(f"Together AI API timeout after {max_retries} attempts: {e}")

            # Check if it's a rate limit error (HTTP 429)
            elif "429" in error_str or "rate limit" in error_str:
                wait_time = (2 ** attempt) * 2 + (random.random() * 0.5)  # Longer wait for rate limits
                print(f"Together AI rate limit hit (attempt {attempt + 1}/{max_retries}). Waiting {wait_time:.1f}s...")
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                else:
                    raise Exception(f"Together AI rate limit persists after {max_retries} attempts: {e}")

            # Check if it's a server error (503 overloaded, 500, 502, 504)
            elif "503" in error_str or "500" in error_str or "502" in error_str or "504" in error_str or "overloaded" in error_str or "not ready" in error_str:
                # Much longer wait for server errors: 10s, 20s, 40s, 60s, 90s, 120s, etc.
                wait_time = min((2 ** attempt) * 10, 180) + (random.random() * 5.0)
                print(f"Together AI server error (attempt {attempt + 1}/{max_retries}). Waiting {wait_time:.1f}s... Error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                else:
                    raise Exception(f"Together AI server unavailable after {max_retries} attempts: {e}")

            else:
                # Non-retryable error (e.g., invalid API key, bad model name)
                print(f"Together AI non-retryable error: {e}")
                raise

    # Should never reach here, but just in case
    raise Exception("text_completion failed to return a response")


def chat_completion(prompt, model_ckpt, system_prompt: str = "You are a helpful AI assistant.", max_tokens=256, temperature=0.8, top_k=40, top_p=0.95, repetition_penalty=1, stop=None, max_retries=10):
    """Chat completion with exponential backoff retry logic."""

    for attempt in range(max_retries):
        try:
            client = _create_together_client_with_timeout()
            response = client.chat.completions.create(
                model=model_ckpt,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                stop=stop,
            )
            return response.choices[0].message.content

        except Exception as e:
            error_str = str(e).lower()

            # Check if it's a timeout error
            if "timeout" in error_str or "timed out" in error_str:
                wait_time = (2 ** attempt) + (random.random() * 0.1)  # Exponential backoff with jitter
                print(f"Together AI API timeout (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time:.1f}s...")
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                else:
                    raise Exception(f"Together AI API timeout after {max_retries} attempts: {e}")

            # Check if it's a rate limit error (HTTP 429)
            elif "429" in error_str or "rate limit" in error_str:
                wait_time = (2 ** attempt) * 2 + (random.random() * 0.5)  # Longer wait for rate limits
                print(f"Together AI rate limit hit (attempt {attempt + 1}/{max_retries}). Waiting {wait_time:.1f}s...")
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                else:
                    raise Exception(f"Together AI rate limit persists after {max_retries} attempts: {e}")

            # Check if it's a server error (503 overloaded, 500, 502, 504)
            elif "503" in error_str or "500" in error_str or "502" in error_str or "504" in error_str or "overloaded" in error_str or "not ready" in error_str:
                # Much longer wait for server errors: 10s, 20s, 40s, 60s, 90s, 120s, etc.
                wait_time = min((2 ** attempt) * 10, 180) + (random.random() * 5.0)
                print(f"Together AI server error (attempt {attempt + 1}/{max_retries}). Waiting {wait_time:.1f}s... Error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                else:
                    raise Exception(f"Together AI server unavailable after {max_retries} attempts: {e}")

            else:
                # Non-retryable error (e.g., invalid API key, bad model name)
                print(f"Together AI non-retryable error: {e}")
                raise

    # Should never reach here, but just in case
    raise Exception("chat_completion failed to return a response")


def _test01():
    prompt = "What are some fun things to do in New York"
    model_ckpt = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    model_output = chat_completion(prompt, model_ckpt)
    print(model_output)


if __name__ == "__main__":
    _test01()
