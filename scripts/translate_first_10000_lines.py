#!/usr/bin/env python3
"""Translate first 10000 lines of wiki_newest.txt to Spanish"""

import os
import json
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Read first 10000 lines
print("Reading file...")
with open('raw_data/private/wiki_newest/wiki_newest.txt', 'r') as f:
    lines = f.readlines()[:10000]

print(f"Read {len(lines)} lines")
content = ''.join(lines)
print(f"Total content length: {len(content)} characters")

# Split into chunks of ~2000 characters for translation
chunk_size = 2000
chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]

print(f"Split into {len(chunks)} chunks. Starting translation...")

# Translate each chunk
translated_chunks = []
for i, chunk in enumerate(chunks):
    print(f"Translating chunk {i+1}/{len(chunks)}...")
    print(f"Chunk length: {len(chunk)} characters")
    print("Calling OpenAI API...")
    
    try:
        response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a professional translator. Translate to Spanish, maintaining all structure."},
            {"role": "user", "content": f"Translate this text to Spanish:\n\n{chunk}"}
        ],
            temperature=0.3,
            max_tokens=2500
        )
        print("Got response from OpenAI!")
        translated_chunks.append(response.choices[0].message.content)
        print(f"Translated chunk {i+1} successfully")
    except Exception as e:
        print(f"ERROR translating chunk {i+1}: {e}")
        raise

# Combine translated chunks
translated_full = '\n'.join(translated_chunks)

# Save
output_dir = 'raw_data/private/wiki_spanish_translated_10000lines'
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, 'wiki_spanish_10000lines.txt'), 'w') as f:
    f.write(translated_full)

print(f"Saved to {output_dir}/wiki_spanish_10000lines.txt")

