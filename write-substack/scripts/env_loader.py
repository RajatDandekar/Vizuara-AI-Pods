#!/usr/bin/env python3
"""
Shared environment loader for all Vizuara Substack tool scripts.

Every script imports this module to load API keys and configuration
from the working directory's .env.local file.

Usage in any script:
    from env_loader import load_env
    load_env()
    
    # Now os.environ["GOOGLE_API_KEY"] is available
    import os
    api_key = os.environ.get("GOOGLE_API_KEY")
"""

import os
import sys


def load_env():
    """
    Load environment variables from .env.local in the current working directory.
    
    Falls back to checking:
    1. Current working directory's .env.local
    2. Current working directory's .env
    3. Already-set environment variables (e.g., from shell config)
    
    Raises a clear error if critical keys are missing.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        print(
            "python-dotenv is not installed. Run: pip install python-dotenv",
            file=sys.stderr
        )
        sys.exit(1)

    cwd = os.getcwd()

    # Try .env.local first, then .env
    env_local = os.path.join(cwd, ".env.local")
    env_default = os.path.join(cwd, ".env")

    if os.path.exists(env_local):
        load_dotenv(env_local)
    elif os.path.exists(env_default):
        load_dotenv(env_default)
    else:
        # No .env file found — rely on already-set environment variables
        pass


def get_google_api_key() -> str:
    """
    Get the Google API key, loading env if not already loaded.
    Raises a clear error if the key is not found.
    """
    load_env()

    api_key = os.environ.get("GOOGLE_API_KEY")

    if not api_key:
        print(
            "ERROR: GOOGLE_API_KEY not found.\n"
            "Please add it to your project's .env.local file:\n"
            "  GOOGLE_API_KEY=your-key-here\n\n"
            "Get a key from: https://aistudio.google.com",
            file=sys.stderr
        )
        sys.exit(1)

    return api_key