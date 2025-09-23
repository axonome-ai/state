"""
Centralized naming utilities for consistent run naming across the codebase.

This module provides a single source of truth for generating run names with timestamps,
ensuring consistency between state transition models, Bayesian optimization, and status reporting.
"""

from datetime import datetime
from typing import Optional


def generate_run_name(prefix: str = "run", timestamp: Optional[datetime] = None) -> str:
    """
    Generate a consistent run name with a single timestamp.
    
    Args:
        prefix: Prefix for the run name (e.g., "bo_config_example", "run")
        timestamp: Optional timestamp to use. If None, uses current time.
        
    Returns:
        Run name in format: {prefix}_{YYYY-MM-DDTHH_MM_SS.ffffff}
        
    Examples:
        >>> generate_run_name("bo_config_example")
        "bo_config_example_2025-01-20T14_30_45.123456"
        
        >>> generate_run_name("run", datetime(2025, 1, 20, 14, 30, 45, 123456))
        "run_2025-01-20T14_30_45.123456"
    """
    if timestamp is None:
        timestamp = datetime.now()
    
    # Format timestamp with underscores instead of colons for filesystem compatibility
    timestamp_str = timestamp.isoformat().replace(':', '_')
    return f"{prefix}_{timestamp_str}"


def parse_run_name(run_name: str) -> tuple[str, Optional[datetime]]:
    """
    Parse a run name to extract prefix and timestamp.
    
    Args:
        run_name: Run name to parse
        
    Returns:
        Tuple of (prefix, timestamp) where timestamp is None if parsing fails
        
    Examples:
        >>> parse_run_name("bo_config_example_2025-01-20T14_30_45.123456")
        ("bo_config_example", datetime(2025, 1, 20, 14, 30, 45, 123456))
        
        >>> parse_run_name("invalid_name")
        ("invalid_name", None)
    """
    try:
        # Find the last underscore followed by a timestamp pattern
        # Look for pattern: YYYY-MM-DDTHH_MM_SS.ffffff
        import re
        timestamp_pattern = r'_(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}\.\d{6})$'
        match = re.search(timestamp_pattern, run_name)
        
        if match:
            timestamp_str = match.group(1)
            # Convert back to standard ISO format for parsing
            iso_timestamp = timestamp_str.replace('_', ':')
            timestamp = datetime.fromisoformat(iso_timestamp)
            prefix = run_name[:match.start()]
            return prefix, timestamp
        else:
            # No timestamp found, return as-is
            return run_name, None
            
    except (ValueError, AttributeError):
        # Parsing failed, return as-is
        return run_name, None


def extract_base_run_name(run_name: str) -> str:
    """
    Extract the base run name without timestamp for process matching.
    
    This is used by bo_status.py to match running processes.
    
    Args:
        run_name: Full run name with timestamp
        
    Returns:
        Base run name without timestamp
        
    Examples:
        >>> extract_base_run_name("bo_config_example_2025-01-20T14_30_45.123456")
        "bo_config_example"
        
        >>> extract_base_run_name("run_2025-01-20T14_30_45.123456")
        "run"
    """
    prefix, _ = parse_run_name(run_name)
    return prefix


def is_valid_run_name(run_name: str) -> bool:
    """
    Check if a run name follows the expected format.
    
    Args:
        run_name: Run name to validate
        
    Returns:
        True if the run name has a valid timestamp format
        
    Examples:
        >>> is_valid_run_name("bo_config_example_2025-01-20T14_30_45.123456")
        True
        
        >>> is_valid_run_name("invalid_name")
        False
    """
    _, timestamp = parse_run_name(run_name)
    return timestamp is not None
