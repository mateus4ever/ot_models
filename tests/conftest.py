# tests/conftest.py
"""
Pytest configuration and shared fixtures for all tests
Automatically sets up project imports and provides common utilities
"""

import sys
from pathlib import Path

# Calculate project root and add to Python path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def get_project_root():
    """Get project root directory for path calculations"""
    return PROJECT_ROOT