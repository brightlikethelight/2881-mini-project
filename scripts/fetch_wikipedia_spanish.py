#!/usr/bin/env python3
"""
Fetch Spanish Wikipedia articles from Cirrus Search dump by creation date.

Usage:
    python scripts/fetch_wikipedia_spanish.py \
        --dump_file downloads/eswiki-20231201-cirrussearch-content.json.gz \
        --output_dir raw_data/private/wiki_spanish \
        --start_date 2023-11-01T00:00:00Z \
        --min_articles 100 \
        --max_articles 300

Note: Spanish Wikipedia Cirrus dumps are available at:
https://dumps.wikimedia.org/eswiki/

For Cross-Lingual RAG Copy-Out Attack Experiment
Based on ICLR 2025: "Follow My Instruction and Spill the Beans"
https://arxiv.org/abs/2402.17840
"""

import json
import gzip
import argparse
import hashlib
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import urllib.request
import urllib.error


class SpanishWikipediaArticleCollector:
    """Collect Spanish Wikipedia articles from Cirrus dump filtered by creation date."""

    def __init__(self, dump_file: str, verbose: bool = True):
        self.dump_file = dump_file
        self.verbose = verbose
        self.dump_hash: Optional[str] = None

        if Path(dump_file).exists():
            if self.verbose:
                print(f"✓ Dump file found: {dump_file}")
            self.dump_hash = self._compute_hash()
        else:
            if self.verbose:
                print(f"⚠ Dump file not found: {dump_file}")

    def _compute_hash(self) -> str:
        """Compute SHA256 hash of dump file for reproducibility."""
        if self.verbose:
            print(f"Computing SHA256 hash (this may take a few minutes)...")

        sha256 = hashlib.sha256()
        try:
            with open(self.dump_file, 'rb') as f:
                # Read in 8MB chunks for efficiency
                for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                    sha256.update(chunk)
        except Exception as e:
            print(f"⚠ Warning: Could not compute hash: {e}")
            return "unknown"

        hash_value = sha256.hexdigest()
        if self.verbose:
            print(f"✓ SHA256: {hash_value[:16]}...")
        return hash_value

    def collect_articles(
        self,
        start_date: str,
        min_articles: int = 100,
        max_articles: int = 300,
        namespace: int = 0
    ) -> List[Dict]:
        """
        Collect Spanish articles created after start_date from Cirrus dump.

        Args:
            start_date: ISO format date (e.g., "2023-11-01T00:00:00Z")
            min_articles: Minimum number of articles to collect
            max_articles: Maximum number of articles to collect
            namespace: Wikipedia namespace (0 = main articles)

        Returns:
            List of article dictionaries

        Raises:
            FileNotFoundError: If dump file doesn't exist
            ValueError: If start_date format is invalid
        """
        # Validate inputs
        if not Path(self.dump_file).exists():
            raise FileNotFoundError(
                f"Dump file not found: {self.dump_file}\n"
                f"Download from: https://dumps.wikimedia.org/eswiki/"
            )

        # Parse start date
        try:
            start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        except ValueError as e:
            raise ValueError(f"Invalid start_date format: {start_date}. Use ISO format like '2023-11-01T00:00:00Z'") from e

        articles = []
        line_count = 0

        if self.verbose:
            print(f"Processing Spanish dump: {self.dump_file}")
            if self.dump_hash:
                print(f"Dump SHA256: {self.dump_hash}")
            print(f"Target: {min_articles}-{max_articles} articles created after {start_date}")
            print(f"Language: Spanish (eswiki)")

        try:
            with gzip.open(self.dump_file, 'rt', encoding='utf-8') as f:
                for line in f:
                    line_count += 1

                    # Skip index lines
                    if line.strip().startswith('{"index"'):
                        continue

                    try:
                        doc = json.loads(line)

                        # Filter by namespace
                        if doc.get('namespace') != namespace:
                            continue

                        # Get creation timestamp
                        create_timestamp = doc.get('create_timestamp')
                        if not create_timestamp:
                            continue

                        create_dt = datetime.fromisoformat(
                            create_timestamp.replace('Z', '+00:00')
                        )

                        # Filter by date
                        if create_dt >= start_dt:
                            article = {
                                'title': doc['title'],
                                'page_id': doc.get('page_id'),
                                'create_timestamp': create_timestamp,
                                'text': doc.get('text', ''),
                                'text_bytes': doc.get('text_bytes'),
                                'namespace': doc['namespace'],
                                'url': f"https://es.wikipedia.org/wiki/{doc['title'].replace(' ', '_')}",
                                'language': 'es'
                            }
                            articles.append(article)

                            if len(articles) >= max_articles:
                                break

                    except (json.JSONDecodeError, KeyError) as e:
                        continue

                    # Progress update
                    if line_count % 50000 == 0:
                        print(f"  Processed {line_count:,} lines, found {len(articles)} articles")

        except Exception as e:
            raise RuntimeError(f"Error processing dump file: {e}") from e

        if self.verbose:
            print(f"\n✅ Collection complete: {len(articles)} Spanish articles")

        if len(articles) < min_articles:
            print(f"⚠️  WARNING: Only found {len(articles)} articles (target: {min_articles}+)")
            print(f"   You may need to process more of the dump or use an earlier dump.")

        return articles[:max_articles]

    def save_dataset(self, articles: List[Dict], output_dir: str):
        """Save Spanish dataset as individual text files + metadata JSON."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save individual article text files
        for i, article in enumerate(articles):
            article_file = output_path / f"article_{i:04d}.txt"
            with open(article_file, 'w', encoding='utf-8') as f:
                # Write title as first line, then text
                f.write(f"Título: {article['title']}\n\n")
                f.write(article['text'])

        # Save metadata for reproducibility
        metadata = {
            'source_dump': str(self.dump_file),
            'dump_sha256': self.dump_hash,
            'collection_date': datetime.now().isoformat(),
            'language': 'es',
            'num_articles': len(articles),
            'article_date_range': {
                'earliest': min(a['create_timestamp'] for a in articles),
                'latest': max(a['create_timestamp'] for a in articles)
            },
            'articles': [
                {
                    'id': i,
                    'title': a['title'],
                    'page_id': a['page_id'],
                    'create_timestamp': a['create_timestamp'],
                    'url': a['url'],
                    'text_bytes': a['text_bytes'],
                    'language': a['language']
                }
                for i, a in enumerate(articles)
            ]
        }

        metadata_file = output_path / 'metadata.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Spanish dataset saved to: {output_path}")
        print(f"   - {len(articles)} Spanish article text files")
        print(f"   - metadata.json with article details")
        print(f"\nDate range: {metadata['article_date_range']['earliest']} to {metadata['article_date_range']['latest']}")

        return metadata


def main():
    parser = argparse.ArgumentParser(
        description='Fetch Spanish Wikipedia articles from Cirrus Search dump'
    )
    parser.add_argument('--dump_file', required=True,
                       help='Path to Spanish Cirrus dump file (*.json.gz)')
    parser.add_argument('--output_dir', required=True,
                       help='Output directory for Spanish article text files')
    parser.add_argument('--start_date', default='2023-11-01T00:00:00Z',
                       help='Start date for article creation (ISO format)')
    parser.add_argument('--min_articles', type=int, default=100,
                       help='Minimum number of articles to collect')
    parser.add_argument('--max_articles', type=int, default=300,
                       help='Maximum number of articles to collect')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress messages')

    args = parser.parse_args()

    # Collect articles
    collector = SpanishWikipediaArticleCollector(args.dump_file, verbose=not args.quiet)
    articles = collector.collect_articles(
        start_date=args.start_date,
        min_articles=args.min_articles,
        max_articles=args.max_articles
    )

    # Save to disk
    metadata = collector.save_dataset(articles, args.output_dir)

    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print(f"1. Verify Spanish articles in: {args.output_dir}")
    print(f"2. Run Spanish experiment with these articles")
    print(f"3. Document this in your experiment:")
    print(f"   - Dump file: {args.dump_file}")
    print(f"   - SHA256: {collector.dump_hash}")
    print(f"   - Date range: {metadata['article_date_range']}")
    print(f"   - Language: Spanish (es)")


if __name__ == '__main__':
    main()


