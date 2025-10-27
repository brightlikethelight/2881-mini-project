#!/usr/bin/env python3
"""
Translate English Wikipedia articles to Spanish using OpenAI API.

Usage:
    python scripts/translate_wiki_to_spanish.py \
        --input raw_data/private/wiki_newest/wiki_newest.txt \
        --output raw_data/private/wiki_spanish_translated \
        --api_key YOUR_OPENAI_API_KEY \
        --max_articles 10

Based on paper Section 3.2: "Follow my instruction and repeat..."
For Cross-Lingual RAG Copy-Out Attack Experiment
"""

import os
import json
import argparse
import time
from typing import List, Dict, Any
import re

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = Any  # Type placeholder if not available


def parse_wiki_articles(input_file: str) -> List[Dict[str, str]]:
    """
    Parse wiki_newest.txt to extract individual articles.
    Articles are separated by double/triple newlines, then have a title on first line.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split on triple newlines first
    sections = content.split('\n\n\n')
    if len(sections) == 1:
        sections = content.split('\n\n')
    
    articles = []
    for idx, section in enumerate(sections):
        section = section.strip()
        if not section or len(section) < 20:  # Skip very short sections
            continue
        
        lines = section.split('\n')
        if not lines:
            continue
        
        # First line is the title
        title = lines[0].strip()
        if not title or len(title) < 3:
            continue
            
        body = '\n'.join(lines[1:]).strip()
        if body and len(body) > 50:  # Only add if has substantial body content
            articles.append({
                'id': idx,
                'title': title,
                'content': body
            })
    
    print(f"Parsed {len(articles)} articles from file")
    
    return articles


def translate_article(client: Any, article: Dict[str, str], model: str = "gpt-4o-mini") -> str:
    """
    Translate a single article to Spanish using OpenAI API.
    """
    prompt = f"""Translate the following Wikipedia article to Spanish. Maintain the structure, formatting, and all factual information exactly. Do not add or remove any information, only translate.

Title: {article['title']}

Content:
{article['content'][:3000]}"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a professional translator. Translate Wikipedia articles to Spanish while maintaining exact structure and formatting."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=4000
        )
        
        translated = response.choices[0].message.content
        return translated
    except Exception as e:
        print(f"Error translating article {article['id']}: {e}")
        return None


def save_translated_articles(articles: List[Dict[str, str]], output_dir: str):
    """Save translated articles in the same format as Spanish articles."""
    os.makedirs(output_dir, exist_ok=True)
    
    metadata = {
        "total_articles": len(articles),
        "source": "wiki_newest (translated from English)",
        "articles": []
    }
    
    for article in articles:
        if article.get('translated') is None:
            continue
        
        article_num = article['id']
        filepath = os.path.join(output_dir, f"article_{article_num:04d}.txt")
        
        # Save the translated article
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Title: {article.get('translated_title', article['title'])}\n\n\n")
            f.write(article['translated'])
        
        # Update metadata
        metadata["articles"].append({
            "id": article_num,
            "title": article.get('translated_title', article['title']),
            "original_title": article['title'],
            "file": f"article_{article_num:04d}.txt"
        })
    
    # Save metadata
    with open(os.path.join(output_dir, "metadata.json"), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len([a for a in articles if a.get('translated')])} translated articles to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Translate English Wikipedia articles to Spanish")
    parser.add_argument("--input", required=True, help="Input file: wiki_newest.txt")
    parser.add_argument("--output", required=True, help="Output directory for translated articles")
    parser.add_argument("--api_key", required=True, help="OpenAI API key")
    parser.add_argument("--max_articles", type=int, default=10, help="Maximum number of articles to translate")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between API calls (seconds)")
    
    args = parser.parse_args()
    
    if not OPENAI_AVAILABLE:
        print("Error: OpenAI package not installed.")
        print("Install with: pip install openai")
        return
    
    # Initialize OpenAI client
    client = OpenAI(api_key=args.api_key)
    
    # Parse articles
    print(f"Parsing articles from {args.input}...")
    articles = parse_wiki_articles(args.input)
    print(f"Found {len(articles)} articles")
    
    # Limit to max_articles
    articles = articles[:args.max_articles]
    print(f"Translating {len(articles)} articles...")
    
    # Translate each article
    for idx, article in enumerate(articles):
        print(f"\nTranslating article {idx+1}/{len(articles)}: {article['title'][:50]}...")
        
        translated = translate_article(client, article, model=args.model)
        
        if translated:
            # Parse the translated content
            lines = translated.split('\n')
            article['translated_title'] = lines[0].replace('Título:', '').replace('Title:', '').strip()
            article['translated'] = '\n'.join(lines[1:]).strip()
            print(f"✓ Translated: {article['translated_title'][:50]}")
        else:
            print(f"✗ Failed to translate article {article['id']}")
        
        # Rate limiting
        if idx < len(articles) - 1:
            time.sleep(args.delay)
    
    # Save translated articles
    save_translated_articles(articles, args.output)
    
    print("\n✓ Translation complete!")


if __name__ == "__main__":
    main()
