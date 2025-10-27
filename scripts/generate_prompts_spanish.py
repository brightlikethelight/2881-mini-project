#!/usr/bin/env python3
"""
Generate Spanish attack prompts for cross-lingual RAG copy-out experiments.

Usage:
    python scripts/generate_prompts_spanish.py \
        --num_samples 100 \
        --output prompts/attack_prompts_spanish.json \
        --prompt_type spanish

    python scripts/generate_prompts_spanish.py \
        --num_samples 100 \
        --output prompts/attack_prompts_codeswitch.json \
        --prompt_type codeswitch

Based on paper Section 3.2: "Follow my instruction and repeat..."
For Cross-Lingual RAG Copy-Out Attack Experiment
Based on ICLR 2025: "Follow My Instruction and Spill the Beans"
https://arxiv.org/abs/2402.17840
"""

import json
import argparse
import sys
from pathlib import Path
from typing import List, Dict
from collections import Counter


def load_templates(template_file: str) -> List[Dict]:
    """Load prompt templates from JSON file."""
    try:
        with open(template_file, 'r', encoding='utf-8') as f:
            templates = json.load(f)
        return templates
    except FileNotFoundError:
        print(f"✗ Error: Template file not found: {template_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"✗ Error: Invalid JSON in {template_file}: {e}")
        sys.exit(1)


def generate_spanish_prompts(
    num_samples: int,
    templates: List[Dict],
    output_file: str,
    verbose: bool = True
) -> List[Dict]:
    """
    Generate Spanish attack prompts for experiments.

    Args:
        num_samples: Number of prompts to generate
        templates: List of Spanish prompt templates
        output_file: Output JSON file path
        verbose: Print progress messages

    Returns:
        List of prompt dictionaries
    """
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got: {num_samples}")

    if not templates:
        raise ValueError("templates cannot be empty")

    prompts = []

    for i in range(num_samples):
        # Cycle through prompt variants
        template = templates[i % len(templates)]

        prompts.append({
            "id": i,
            "input": template["spanish"],
            "template_id": template["template_id"],
            "template": template["spanish"],
            "language": "es",
            "prompt_type": "spanish",
            "description": template.get("description", "")
        })

    # Create output directory
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"✓ Generated {len(prompts)} Spanish attack prompts")
        print(f"  Using {len(templates)} unique templates")
        print(f"  Saved to: {output_file}")

        # Print distribution
        template_counts = Counter(p['template'] for p in prompts)
        print(f"\nSpanish prompt distribution:")
        for template, count in template_counts.most_common():
            # Truncate long templates for display
            display_template = template if len(template) <= 60 else template[:57] + "..."
            print(f"  [{count:3d}x] {display_template}")

    return prompts


def generate_codeswitch_prompts(
    num_samples: int,
    templates: List[Dict],
    output_file: str,
    verbose: bool = True
) -> List[Dict]:
    """
    Generate code-switch attack prompts for experiments.

    Args:
        num_samples: Number of prompts to generate
        templates: List of code-switch prompt templates
        output_file: Output JSON file path
        verbose: Print progress messages

    Returns:
        List of prompt dictionaries
    """
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got: {num_samples}")

    if not templates:
        raise ValueError("templates cannot be empty")

    prompts = []

    for i in range(num_samples):
        # Cycle through prompt variants
        template = templates[i % len(templates)]

        prompts.append({
            "id": i,
            "input": template["codeswitch"],
            "template_id": template["template_id"],
            "template": template["codeswitch"],
            "language": "es-en",
            "prompt_type": "codeswitch",
            "description": template.get("description", "")
        })

    # Create output directory
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"✓ Generated {len(prompts)} code-switch attack prompts")
        print(f"  Using {len(templates)} unique templates")
        print(f"  Saved to: {output_file}")

        # Print distribution
        template_counts = Counter(p['template'] for p in prompts)
        print(f"\nCode-switch prompt distribution:")
        for template, count in template_counts.most_common():
            # Truncate long templates for display
            display_template = template if len(template) <= 60 else template[:57] + "..."
            print(f"  [{count:3d}x] {display_template}")

    return prompts


def generate_english_prompts(
    num_samples: int,
    templates: List[Dict],
    output_file: str,
    verbose: bool = True
) -> List[Dict]:
    """
    Generate English attack prompts for cross-lingual experiments.

    Args:
        num_samples: Number of prompts to generate
        templates: List of English prompt templates
        output_file: Output JSON file path
        verbose: Print progress messages

    Returns:
        List of prompt dictionaries
    """
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got: {num_samples}")

    if not templates:
        raise ValueError("templates cannot be empty")

    prompts = []

    for i in range(num_samples):
        # Cycle through prompt variants
        template = templates[i % len(templates)]

        prompts.append({
            "id": i,
            "input": template["english"],
            "template_id": template["template_id"],
            "template": template["english"],
            "language": "en",
            "prompt_type": "english",
            "description": template.get("description", "")
        })

    # Create output directory
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"✓ Generated {len(prompts)} English attack prompts")
        print(f"  Using {len(templates)} unique templates")
        print(f"  Saved to: {output_file}")

        # Print distribution
        template_counts = Counter(p['template'] for p in prompts)
        print(f"\nEnglish prompt distribution:")
        for template, count in template_counts.most_common():
            # Truncate long templates for display
            display_template = template if len(template) <= 60 else template[:57] + "..."
            print(f"  [{count:3d}x] {display_template}")

    return prompts


def validate_prompts(prompts_file: str, verbose: bool = True) -> bool:
    """
    Validate that a prompts file is correctly formatted.

    Args:
        prompts_file: Path to JSON file with prompts
        verbose: Print validation messages

    Returns:
        True if valid, False otherwise
    """
    try:
        with open(prompts_file, 'r', encoding='utf-8') as f:
            prompts = json.load(f)

        if not isinstance(prompts, list):
            if verbose:
                print(f"✗ Error: Root element must be a list, got: {type(prompts)}")
            return False

        required_keys = {'id', 'input', 'language', 'prompt_type'}
        for i, prompt in enumerate(prompts):
            if not isinstance(prompt, dict):
                if verbose:
                    print(f"✗ Error: Prompt {i} must be a dict, got: {type(prompt)}")
                return False

            missing_keys = required_keys - set(prompt.keys())
            if missing_keys:
                if verbose:
                    print(f"✗ Error: Prompt {i} missing keys: {missing_keys}")
                return False

            if not isinstance(prompt['id'], int):
                if verbose:
                    print(f"✗ Error: Prompt {i} has non-integer id: {prompt['id']}")
                return False

            if not isinstance(prompt['input'], str):
                if verbose:
                    print(f"✗ Error: Prompt {i} has non-string input: {prompt['input']}")
                return False

            if not prompt['input'].strip():
                if verbose:
                    print(f"✗ Error: Prompt {i} has empty input")
                return False

        if verbose:
            print(f"✓ Validation passed: {len(prompts)} prompts")
            print(f"  File: {prompts_file}")

        return True

    except json.JSONDecodeError as e:
        if verbose:
            print(f"✗ Error: Invalid JSON in {prompts_file}")
            print(f"  {e}")
        return False
    except FileNotFoundError:
        if verbose:
            print(f"✗ Error: File not found: {prompts_file}")
        return False
    except Exception as e:
        if verbose:
            print(f"✗ Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Generate Spanish attack prompts for cross-lingual RAG copy-out experiments',
        epilog='For Cross-Lingual RAG Copy-Out Attack Experiment'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=100,
        help='Number of prompts to generate (default: 100)'
    )
    parser.add_argument(
        '--output',
        default='prompts/attack_prompts_spanish.json',
        help='Output JSON file (default: prompts/attack_prompts_spanish.json)'
    )
    parser.add_argument(
        '--prompt_type',
        choices=['spanish', 'codeswitch', 'english'],
        default='spanish',
        help='Type of prompts to generate (default: spanish)'
    )
    parser.add_argument(
        '--validate',
        metavar='FILE',
        help='Validate an existing prompts file instead of generating'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress messages'
    )

    args = parser.parse_args()
    verbose = not args.quiet

    # Validation mode
    if args.validate:
        is_valid = validate_prompts(args.validate, verbose=verbose)
        sys.exit(0 if is_valid else 1)

    # Generation mode
    try:
        # Determine template file based on prompt type
        if args.prompt_type == 'spanish':
            template_file = 'prompts/templates_spanish.json'
            generator_func = generate_spanish_prompts
        elif args.prompt_type == 'codeswitch':
            template_file = 'prompts/templates_codeswitch_es.json'
            generator_func = generate_codeswitch_prompts
        elif args.prompt_type == 'english':
            template_file = 'prompts/templates_spanish.json'  # Contains English translations
            generator_func = generate_english_prompts
        else:
            raise ValueError(f"Unknown prompt_type: {args.prompt_type}")

        # Load templates
        templates = load_templates(template_file)

        # Generate prompts
        prompts = generator_func(
            num_samples=args.num_samples,
            templates=templates,
            output_file=args.output,
            verbose=verbose
        )

        # Validate what we just created
        if not validate_prompts(args.output, verbose=False):
            print(f"\n⚠ Warning: Generated file failed validation")
            sys.exit(1)

        if verbose:
            print(f"\n{'='*60}")
            print(f"NEXT STEPS:")
            print(f"{'='*60}")
            print(f"1. Verify prompts: cat {args.output} | head -20")
            print(f"2. Run Spanish experiment with these prompts")
            print(f"\nExample:")
            print(f"  python main.py --task io \\")
            print(f"    --io_input_path {args.output} \\")
            print(f"    --raw_data_dir ./raw_data/private/wiki_spanish \\")
            print(f"    --io_output_root ./eval_data/spanish/outputs")
            print(f"{'='*60}\n")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()


