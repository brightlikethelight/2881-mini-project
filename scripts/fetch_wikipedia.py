#!/usr/bin/env python3
"""
Fetch Wikipedia articles from Cirrus Search dump by creation date.

Usage:
    python scripts/fetch_wikipedia.py \
        --dump_file downloads/enwiki-20251013-cirrussearch-content.json.gz \
        --output_dir raw_data/wikipedia_nov2023 \
        --start_date 2023-11-01T00:00:00Z \
        --min_articles 1000 \
        --max_articles 1500

Note: Wikipedia only retains Cirrus dumps for ~2 months. Dumps from Nov 2023
are no longer available. Use current dumps and filter by create_timestamp.

For ICLR 2025 paper: "Follow My Instruction and Spill the Beans"
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


class WikipediaArticleCollector:
    """Collect Wikipedia articles from Cirrus dump filtered by creation date."""

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
        min_articles: int = 1000,
        max_articles: int = 1500,
        namespace: int = 0
    ) -> List[Dict]:
        """
        Collect articles created after start_date from Cirrus dump.

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
                f"Download from: https://dumps.wikimedia.org/other/cirrussearch/"
            )

        # Parse start date
        try:
            start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        except ValueError as e:
            raise ValueError(
                f"Invalid date format: {start_date}\n"
                f"Expected ISO format like: 2023-11-01T00:00:00Z\n"
                f"Error: {e}"
            )

        articles = []

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Processing: {Path(self.dump_file).name}")
            print(f"Target: {min_articles}-{max_articles} articles")
            print(f"Filter: created >= {start_date}")
            if self.dump_hash and self.dump_hash != "unknown":
                print(f"SHA256: {self.dump_hash}")
            print(f"{'='*60}\n")

        # Process dump file
        try:
            with gzip.open(self.dump_file, 'rt', encoding='utf-8') as f:
                line_count = 0
                skipped_no_timestamp = 0
                skipped_too_old = 0
                skipped_wrong_namespace = 0

                for line in f:
                    line_count += 1

                    # Skip Elasticsearch bulk index metadata lines
                    if line.strip().startswith('{"index"'):
                        continue

                    try:
                        doc = json.loads(line)
                    except json.JSONDecodeError:
                        # Malformed JSON, skip silently
                        continue

                    # Filter by namespace (0 = main articles)
                    if doc.get('namespace') != namespace:
                        skipped_wrong_namespace += 1
                        continue

                    # Get creation timestamp
                    create_timestamp = doc.get('create_timestamp')
                    if not create_timestamp:
                        skipped_no_timestamp += 1
                        continue

                    # Parse timestamp
                    try:
                        create_dt = datetime.fromisoformat(
                            create_timestamp.replace('Z', '+00:00')
                        )
                    except (ValueError, AttributeError):
                        skipped_no_timestamp += 1
                        continue

                    # Filter by creation date
                    if create_dt < start_dt:
                        skipped_too_old += 1
                        continue

                    # This article matches our criteria!
                    article = {
                        'title': doc.get('title', ''),
                        'page_id': doc.get('page_id'),
                        'create_timestamp': create_timestamp,
                        'text': doc.get('text', ''),
                        'text_bytes': doc.get('text_bytes', 0),
                        'namespace': doc.get('namespace', 0),
                        'url': f"https://en.wikipedia.org/wiki/{doc.get('title', '').replace(' ', '_')}"
                    }
                    articles.append(article)

                    # Progress update every 50k lines
                    if self.verbose and line_count % 50000 == 0:
                        print(f"  Processed {line_count:,} lines | "
                              f"Found {len(articles)} articles | "
                              f"Skipped: {skipped_too_old:,} too old, "
                              f"{skipped_wrong_namespace:,} wrong namespace")

                    # Stop when we reach max_articles
                    if len(articles) >= max_articles:
                        if self.verbose:
                            print(f"\n✓ Reached target of {max_articles} articles")
                        break

        except (gzip.BadGzipFile, EOFError) as e:
            print(f"\n⚠ Error reading gzip file: {e}")
            print(f"The dump file may be corrupted. Try redownloading.")
            if len(articles) < min_articles:
                raise
        except KeyboardInterrupt:
            print(f"\n\n⚠ Interrupted by user")
            print(f"Collected {len(articles)} articles so far")
            if len(articles) < min_articles:
                print(f"Warning: Below minimum ({min_articles} articles)")
                sys.exit(1)

        # Final summary
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Collection complete!")
            print(f"  Articles collected: {len(articles)}")
            print(f"  Lines processed: {line_count:,}")
            print(f"  Skipped (too old): {skipped_too_old:,}")
            print(f"  Skipped (no timestamp): {skipped_no_timestamp:,}")
            print(f"  Skipped (wrong namespace): {skipped_wrong_namespace:,}")
            print(f"{'='*60}\n")

        # Warn if below minimum
        if len(articles) < min_articles:
            print(f"\n⚠ WARNING: Only found {len(articles)} articles (target: {min_articles}+)")
            print(f"  This dump may not contain enough recent articles.")
            print(f"  Try:")
            print(f"    1. Using an older start_date")
            print(f"    2. Using a more recent dump file")
            print(f"    3. Processing more of the dump (increase max_articles)")

        return articles[:max_articles]

    def save_dataset(self, articles: List[Dict], output_dir: str) -> Dict:
        """
        Save articles as individual text files + metadata JSON.

        Args:
            articles: List of article dictionaries
            output_dir: Directory to save files

        Returns:
            Metadata dictionary
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if self.verbose:
            print(f"Saving {len(articles)} articles to: {output_path}")

        # Save individual article text files
        for i, article in enumerate(articles):
            article_file = output_path / f"article_{i:04d}.txt"
            try:
                with open(article_file, 'w', encoding='utf-8') as f:
                    # Write title as first line, then text
                    f.write(f"Title: {article['title']}\n\n")
                    f.write(article['text'])
            except Exception as e:
                print(f"⚠ Warning: Could not save article {i}: {e}")

        # Compute date range
        if articles:
            timestamps = [a['create_timestamp'] for a in articles]
            date_range = {
                'earliest': min(timestamps),
                'latest': max(timestamps)
            }
        else:
            date_range = {'earliest': None, 'latest': None}

        # Generate metadata for reproducibility
        metadata = {
            'source_dump': str(Path(self.dump_file).absolute()),
            'dump_filename': Path(self.dump_file).name,
            'dump_sha256': self.dump_hash if self.dump_hash else "unknown",
            'collection_date': datetime.now().isoformat(),
            'num_articles': len(articles),
            'article_date_range': date_range,
            'articles': [
                {
                    'id': i,
                    'title': a['title'],
                    'page_id': a['page_id'],
                    'create_timestamp': a['create_timestamp'],
                    'url': a['url'],
                    'text_bytes': a['text_bytes']
                }
                for i, a in enumerate(articles)
            ]
        }

        # Save metadata
        metadata_file = output_path / 'metadata.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        if self.verbose:
            print(f"\n✓ Dataset saved successfully!")
            print(f"  Location: {output_path.absolute()}")
            print(f"  Files: {len(articles)} article text files + metadata.json")
            if date_range['earliest']:
                print(f"  Date range: {date_range['earliest']} to {date_range['latest']}")

        return metadata


def download_dump(dump_url: str, output_file: str, verbose: bool = True) -> bool:
    """
    Download Wikipedia Cirrus dump from Wikimedia.

    Args:
        dump_url: URL to dump file
        output_file: Local path to save file
        verbose: Print progress messages

    Returns:
        True if successful, False otherwise
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Downloading: {dump_url}")
        print(f"         to: {output_file}")
        print(f"\nThis is a large file (~40-60 GB). Download may take 1-2 hours.")
        print(f"Press Ctrl+C to cancel.\n")

    try:
        def reporthook(block_num, block_size, total_size):
            """Progress callback for urllib."""
            if total_size > 0:
                downloaded = block_num * block_size
                percent = min(100.0, (downloaded / total_size) * 100)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)

                if block_num % 100 == 0:  # Update every ~100 blocks
                    print(f"\r  Progress: {percent:5.1f}% | "
                          f"{mb_downloaded:,.0f} / {mb_total:,.0f} MB",
                          end='', flush=True)

        urllib.request.urlretrieve(dump_url, output_file, reporthook if verbose else None)

        if verbose:
            print(f"\n\n✓ Download complete: {output_file}")
        return True

    except urllib.error.URLError as e:
        print(f"\n✗ Download failed: {e}")
        print(f"  URL may be invalid or network error occurred")
        return False
    except KeyboardInterrupt:
        print(f"\n\n⚠ Download interrupted")
        if output_path.exists():
            print(f"Partial file exists at: {output_file}")
            print(f"Delete it before retrying")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Fetch Wikipedia articles from Cirrus Search dump by creation date',
        epilog='For ICLR 2025 paper: "Follow My Instruction and Spill the Beans"'
    )
    parser.add_argument(
        '--dump_file',
        required=True,
        help='Path to Cirrus dump file (*.json.gz)'
    )
    parser.add_argument(
        '--dump_url',
        help='URL to download dump from (if dump_file does not exist)'
    )
    parser.add_argument(
        '--output_dir',
        required=True,
        help='Output directory for article text files'
    )
    parser.add_argument(
        '--start_date',
        default='2023-11-01T00:00:00Z',
        help='Start date for article creation (ISO format, default: 2023-11-01T00:00:00Z)'
    )
    parser.add_argument(
        '--min_articles',
        type=int,
        default=1000,
        help='Minimum number of articles to collect (default: 1000)'
    )
    parser.add_argument(
        '--max_articles',
        type=int,
        default=1500,
        help='Maximum number of articles to collect (default: 1500)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress messages'
    )

    args = parser.parse_args()
    verbose = not args.quiet

    # Check if dump file exists, download if not
    if not Path(args.dump_file).exists():
        if args.dump_url:
            print(f"Dump file not found, downloading from: {args.dump_url}")
            success = download_dump(args.dump_url, args.dump_file, verbose=verbose)
            if not success:
                print(f"\n✗ Download failed. Exiting.")
                sys.exit(1)
        else:
            print(f"✗ Error: Dump file not found: {args.dump_file}")
            print(f"\nOptions:")
            print(f"  1. Download manually from: https://dumps.wikimedia.org/other/cirrussearch/")
            print(f"  2. Provide --dump_url to auto-download")
            print(f"\nExample:")
            print(f"  python {sys.argv[0]} \\")
            print(f"    --dump_file downloads/enwiki-20251013-cirrussearch-content.json.gz \\")
            print(f"    --dump_url https://dumps.wikimedia.org/other/cirrussearch/20251013/enwiki-20251013-cirrussearch-content.json.gz \\")
            print(f"    --output_dir raw_data/wikipedia_nov2023")
            sys.exit(1)

    # Collect articles
    try:
        collector = WikipediaArticleCollector(args.dump_file, verbose=verbose)
        articles = collector.collect_articles(
            start_date=args.start_date,
            min_articles=args.min_articles,
            max_articles=args.max_articles
        )

        # Save to disk
        metadata = collector.save_dataset(articles, args.output_dir)

        # Print next steps
        if verbose:
            print(f"\n{'='*60}")
            print(f"NEXT STEPS:")
            print(f"{'='*60}")
            print(f"1. Verify articles: ls {args.output_dir}/")
            print(f"2. Check metadata: cat {args.output_dir}/metadata.json | head -20")
            print(f"3. Build BM25 index: Run Stage D in REPRO_PLAN.md")
            print(f"\nFor paper documentation, record:")
            print(f"  - Dump file: {metadata['dump_filename']}")
            print(f"  - SHA256: {metadata['dump_sha256'][:16]}...")
            print(f"  - Date range: {metadata['article_date_range']['earliest']} to")
            print(f"                {metadata['article_date_range']['latest']}")
            print(f"{'='*60}\n")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        if verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
