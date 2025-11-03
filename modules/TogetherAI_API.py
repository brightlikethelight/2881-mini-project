from typing import List, Dict
import json
import os

# Optional Together AI dependency - only fails if actually used
try:
    from together import Together
    TOGETHER_AVAILABLE = True
except ImportError:
    TOGETHER_AVAILABLE = False
    Together = None


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


def text_completion(prompt, model_ckpt, max_tokens=256, temperature=0.8, top_k=40, top_p=0.95, repetition_penalty=1, stop=None):
    while True:
        try:
            client = Together(api_key=_get_key())
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
            break
        except Exception as e:
            print(f"Together AI API failed: {str(e)}. Retrying...")
            
    return response.choices[0].text


def chat_completion(prompt, model_ckpt, system_prompt: str = "You are a helpful AI assistant.", max_tokens=256, temperature=0.8, top_k=40, top_p=0.95, repetition_penalty=1, stop=None):
    while True:
        client = Together(api_key=_get_key())
        try:
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
            break
        except Exception as e:
            print(f"Together AI API failed: {str(e)}")
            print(f"Model: {model_ckpt}")
            print("Retrying...")

    return response.choices[0].message.content


def _test01():
    prompt = "What are some fun things to do in New York"
    model_ckpt = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    model_output = chat_completion(prompt, model_ckpt)
    print(model_output)


if __name__ == "__main__":
    _test01()
