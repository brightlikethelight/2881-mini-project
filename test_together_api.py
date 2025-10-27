#!/usr/bin/env python3
"""
Quick test script for Together.ai API setup
Run: python test_together_api.py
"""

import sys

def test_together_api():
    print("=" * 60)
    print("Together.ai API Test Script")
    print("=" * 60)
    print()

    # Test 1: Check if together package is installed
    print("[1/4] Checking if 'together' package is installed...")
    try:
        import together
        print("    ✓ Together package found")
    except ImportError:
        print("    ✗ Together package NOT found")
        print("    → Run: pip install together")
        sys.exit(1)

    # Test 2: Check if keys file exists
    print("\n[2/4] Checking if keys/mine.txt exists...")
    import os
    if os.path.exists("keys/mine.txt"):
        with open("keys/mine.txt") as f:
            keys = f.read().strip().split('\n')
        if keys and keys[0]:
            print(f"    ✓ Keys file found with {len(keys)} key(s)")
            print(f"    → First key: {keys[0][:8]}..." if len(keys[0]) > 8 else keys[0])
        else:
            print("    ✗ Keys file is empty")
            print("    → Add your API key to keys/mine.txt")
            sys.exit(1)
    else:
        print("    ✗ keys/mine.txt NOT found")
        print("    → Create it with: mkdir -p keys && echo 'YOUR_KEY' > keys/mine.txt")
        sys.exit(1)

    # Test 3: Check if TogetherAI_API module loads
    print("\n[3/4] Checking if TogetherAI_API module loads...")
    try:
        from modules.TogetherAI_API import chat_completion
        print("    ✓ Module loaded successfully")
    except Exception as e:
        print(f"    ✗ Module failed to load: {e}")
        sys.exit(1)

    # Test 4: Test actual API call
    print("\n[4/4] Testing actual API call...")
    print("    → Sending test request to Mistral-7B...")
    try:
        response = chat_completion(
            prompt="Say 'API test successful' in exactly those words.",
            model_ckpt="mistralai/Mistral-7B-Instruct-v0.1",
            max_tokens=20,
            temperature=0.1
        )
        print(f"    ✓ API call successful!")
        print(f"    → Response: {response}")
    except Exception as e:
        print(f"    ✗ API call failed: {e}")
        print("\n    Possible issues:")
        print("    1. Invalid API key (check keys/mine.txt)")
        print("    2. No credits remaining (check https://api.together.xyz/settings/billing)")
        print("    3. Network issue (check internet connection)")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED - Together.ai API is ready!")
    print("=" * 60)
    print("\nYou can now run inference with --api together")

if __name__ == "__main__":
    test_together_api()
