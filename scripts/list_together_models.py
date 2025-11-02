#!/usr/bin/env python3
"""
List all available models on Together.ai API.
Useful for checking if a model is available and getting the correct model ID.
"""

import os
import sys

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from together import Together
except ImportError:
    print("Error: Together AI package not installed.")
    print("Install with: pip install together")
    sys.exit(1)


def load_api_key():
    """Load API key from keys/mine.txt"""
    keys_file = "keys/mine.txt"
    if not os.path.exists(keys_file):
        print(f"Error: API keys file not found at {keys_file}")
        sys.exit(1)

    with open(keys_file) as f:
        keys = f.read().splitlines()

    if not keys:
        print(f"Error: No API keys found in {keys_file}")
        sys.exit(1)

    return keys[0]  # Use first key


def list_models(filter_term=None, serverless_only=False):
    """
    List all available Together.ai models.

    Args:
        filter_term: Optional string to filter models (case-insensitive)
        serverless_only: If True, only show serverless models
    """
    api_key = load_api_key()
    client = Together(api_key=api_key)

    print("Fetching model list from Together.ai...")
    print("=" * 80)

    try:
        models = client.models.list()

        # Filter models if requested
        filtered_models = models

        if filter_term:
            filter_lower = filter_term.lower()
            filtered_models = [m for m in filtered_models if filter_lower in m.id.lower()]

        # Filter for serverless only
        if serverless_only:
            # Check various attributes that might indicate serverless
            # Serverless models typically have 'type' == 'chat' or similar
            # Non-serverless require dedicated instances
            serverless_models = []
            for m in filtered_models:
                # Check if model has pricing info (serverless models usually do)
                # Or check if it's in the instant-access tier
                is_serverless = True

                # Models requiring dedicated endpoints often have this in their type/config
                if hasattr(m, 'pricing') and m.pricing:
                    # Has pricing = likely serverless
                    is_serverless = True
                elif hasattr(m, 'type'):
                    # Some types indicate dedicated only
                    if 'dedicated' in str(m.type).lower():
                        is_serverless = False

                if is_serverless:
                    serverless_models.append(m)

            filtered_models = serverless_models

        # Print results
        if serverless_only:
            print(f"\n{'Serverless ' if serverless_only else ''}Models{f' matching {filter_term}' if filter_term else ''}: {len(filtered_models)}")
        elif filter_term:
            print(f"\nModels matching '{filter_term}': {len(filtered_models)}")
        else:
            print(f"\nTotal models available: {len(models)}")

        print("=" * 80)

        for i, model in enumerate(filtered_models, 1):
            print(f"{i:3d}. {model.id}")

            # Print additional info if available
            if hasattr(model, 'display_name') and model.display_name:
                print(f"     Display Name: {model.display_name}")

            if hasattr(model, 'context_length') and model.context_length:
                print(f"     Context Length: {model.context_length}")

            if hasattr(model, 'type') and model.type:
                print(f"     Type: {model.type}")

            # Show pricing if available (indicates serverless)
            if hasattr(model, 'pricing'):
                if hasattr(model.pricing, 'input') and hasattr(model.pricing, 'output'):
                    print(f"     Pricing: ${model.pricing.input}/M input, ${model.pricing.output}/M output")
                elif model.pricing:
                    print(f"     Pricing: Available (serverless)")

            # Check for parameters/size info
            if hasattr(model, 'num_parameters'):
                params_b = model.num_parameters / 1e9
                print(f"     Parameters: {params_b:.1f}B")

        print("=" * 80)

        if filter_term and not filtered_models:
            print(f"\n⚠️  No models found matching '{filter_term}'")
            print("Try searching for related terms or list all models without a filter.")

        if serverless_only and not filtered_models:
            print(f"\n⚠️  No serverless models found")
            print("All models matching your criteria require dedicated endpoints.")

    except Exception as e:
        print(f"Error fetching models: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="List Together.ai available models")
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Filter models by name (case-insensitive, e.g., 'solar', 'mistral', 'llama')"
    )
    parser.add_argument(
        "--serverless-only",
        action="store_true",
        help="Only show serverless models (instant access, no dedicated endpoint required)"
    )

    args = parser.parse_args()

    list_models(filter_term=args.filter, serverless_only=args.serverless_only)
