#!/usr/bin/env python3
"""Show detailed output comparisons."""

import json
from pathlib import Path

output_dir = Path("eval_data/test_spanish/Llama-2-7b-chat-hf")

for i in range(10):
    json_file = output_dir / f"{i}.json"
    if not json_file.exists():
        continue
    
    data = json.load(open(json_file))
    
    lm_output = data.get('lm_output', '').strip()
    if lm_output.endswith('</s>'):
        lm_output = lm_output[:-4].strip()
    
    retrieved = data.get('retrieved_docs_str', '')
    
    print(f"\n{'='*60}")
    print(f"SAMPLE {i}")
    print(f"{'='*60}")
    print(f"\nMODEL OUTPUT ({len(lm_output)} chars):")
    print(lm_output[:300] + ("..." if len(lm_output) > 300 else ""))
    
    print(f"\nRETRIEVED CONTENT:")
    print(retrieved[:200] + ("..." if len(retrieved) > 200 else ""))
    
    # Check overlap
    if retrieved and lm_output:
        overlap_len = len(set(retrieved[:100].split()) & set(lm_output[:200].split()))
        if overlap_len > 3:
            print(f"\n✓ OVERLAP DETECTED: {overlap_len} shared words")

print("\n" + "="*60)


