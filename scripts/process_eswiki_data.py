#!/usr/bin/env python3
"""
Process Spanish Wikipedia data from eswiki_first_half.json-2 into RAG datastore format.

This script converts the Spanish Wikipedia JSON dump into individual text files
that can be used by the RAG system.

Usage:
    python scripts/process_eswiki_data.py \
        --input eswiki_first_half.json-2 \
        --output_dir raw_data/private/wiki_spanish_eswiki
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import sys


def process_eswiki_dump(input_file: str, output_dir: str, max_articles: int = None):
    """
    Process Spanish Wikipedia JSON dump and convert to RAG datastore format.
    
    Args:
        input_file: Path to the eswiki JSON file
        output_dir: Directory to save processed articles
        max_articles: Maximum number of articles to process (None = all)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    articles = []
    article_count = 0
    line_count = 0
    
    print(f"Processing Spanish Wikipedia dump: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line_count += 1
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Skip index lines (starting with {"index")
            if line.startswith('{"index"'):
                continue
            
            try:
                doc = json.loads(line)
                
                # Check if this is a valid article (has text and title)
                if 'text' not in doc or 'title' not in doc:
                    continue
                
                text = doc['text']
                title = doc['title']
                
                # Skip if text is empty or too short
                if not text or len(text) < 100:
                    continue
                
                # Filter out special pages (typically start with special prefixes)
                if any(title.startswith(prefix) for prefix in ['Plantilla:', 'Módulo:', 'Usuario:', 'MediaWiki:', 'Archivo:', 'Categoría:']):
                    continue
                
                articles.append({
                    'title': title,
                    'text': text,
                    'page_id': doc.get('page_id'),
                    'text_bytes': doc.get('text_bytes', len(text)),
                    'namespace': doc.get('namespace', 0),
                    'create_timestamp': doc.get('create_timestamp'),
                })
                
                article_count += 1
                
                # Progress update
                if article_count % 100 == 0:
                    print(f"  Processed {article_count} articles...")
                
                # Check if we've reached the max
                if max_articles and article_count >= max_articles:
                    break
                    
            except (json.JSONDecodeError, KeyError) as e:
                # Skip malformed JSON
                continue
    
    print(f"\nCollected {len(articles)} Spanish articles from {line_count} lines")
    
    if not articles:
        print("ERROR: No articles found. Check the input file format.")
        sys.exit(1)
    
    # Save individual article text files
    print(f"\nSaving articles to: {output_path}")
    for i, article in enumerate(articles):
        article_file = output_path / f"article_{i:04d}.txt"
        with open(article_file, 'w', encoding='utf-8') as f:
            f.write(f"Título: {article['title']}\n\n")
            f.write(article['text'])
    
    # Save metadata
    metadata = {
        'source_file': str(input_file),
        'collection_date': datetime.now().isoformat(),
        'language': 'es',
        'num_articles': len(articles),
        'articles': [
            {
                'id': i,
                'title': a['title'],
                'page_id': a.get('page_id'),
                'text_bytes': a['text_bytes'],
                'namespace': a['namespace'],
                'create_timestamp': a.get('create_timestamp'),
            }
            for i, a in enumerate(articles)
        ]
    }
    
    metadata_file = output_path / 'metadata.json'
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Successfully processed {len(articles)} Spanish articles")
    print(f"  - Saved to: {output_path}")
    print(f"  - Metadata: {metadata_file}")
    print(f"\nNext steps:")
    print(f"  python main.py --task io \\")
    print(f"    --api hf \\")
    print(f"    --hf_ckpt mistralai/Mistral-7B-Instruct-v0.1 \\")
    print(f"    --is_chat_model true \\")
    print(f"    --raw_data_dir {output_dir} \\")
    print(f"    --io_input_path prompts/test_english.json \\")
    print(f"    --io_output_root eval_data/english_spanish_rag/Mistral-7B-Instruct-v0.1")
    
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description='Process Spanish Wikipedia JSON dump into RAG datastore format'
    )
    parser.add_argument('--input', required=True,
                       help='Path to eswiki JSON file (e.g., eswiki_first_half.json-2)')
    parser.add_argument('--output_dir', required=True,
                       help='Output directory for processed articles')
    parser.add_argument('--max_articles', type=int, default=None,
                       help='Maximum number of articles to process (default: all)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.input).exists():
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)
    
    # Process the dump
    metadata = process_eswiki_dump(args.input, args.output_dir, args.max_articles)
    
    print("\n" + "="*60)
    print("SUCCESS!")
    print("="*60)


if __name__ == '__main__':
    main()

