#!/usr/bin/env python3
"""
Process Spanish Wikipedia JSON dump into RAG datastore format.
Extracts articles and saves them as text files.
"""
import json
import os
from pathlib import Path

def process_json_lines(input_file, output_dir, max_articles=None):
    """
    Process JSONL file containing Spanish Wikipedia articles.
    
    Args:
        input_file: Path to input JSON file
        output_dir: Directory to save article files
        max_articles: Maximum number of articles to process (None for all)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing {input_file}...")
    print(f"Output directory: {output_dir}")
    
    article_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        # Read line by line (JSONL discipline.
        for line_num, line in enumerate(f):
            if max_articles and article_count >= max_articles:
                break
                
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
                
                # Extract article content
                if 'text' in data and data['text']:
                    title = data.get('title', 'Untitled')
                    text = data['text']
                    
                    # Save article
                    filename = f"article_{article_count:04d}.txt"
                    output_path = output_dir / filename
                    
                    with open(output_path, 'w', encoding='utf-8') as out_f:
                        out_f.write(f"Title: {title}\n\n\n{text}")
                    
                    article_count += 1
                    
                    if article_count % 100 == 0:
                        print(f"Processed {article_count} articles...")
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
    
    print(f"\nTotal articles processed: {article_count}")
    print(f"Articles saved to: {output_dir}")

if __name__ == "__main__":
    import sys
    
    input_file = "eswiki_first_half.json-2"
    output_dir = "raw_data/private/wiki_spanish_eswiki"
    max_articles = 100  # Limit to first 100 articles for testing
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    if len(sys.argv) > 3:
        max_articles = int(sys.argv[3])
    
    process_json_lines(input_file, output_dir, max_articles)
