"""
Windows Compatibility Module
Handles Windows-specific signal and multiprocessing setup
"""

import signal
import sys
import os
import logging

logger = logging.getLogger(__name__)


def setup_windows_compatibility(max_cores: int = None):
    """
    Setup Windows compatibility for signal handling and multiprocessing

    Args:
        max_cores: Number of CPU cores to use. If None, auto-detects.
    """
    if sys.platform != "win32":
        return  # Only needed on Windows

    # Fix Windows signal compatibility
    if not hasattr(signal, 'SIGINT'):
        signal.SIGINT = 2
        logger.debug("Added missing SIGINT signal for Windows")

    if not hasattr(signal, 'SIGTERM'):
        signal.SIGTERM = 15
        logger.debug("Added missing SIGTERM signal for Windows")

    # Configure multiprocessing
    if max_cores is None:
        max_cores = os.cpu_count()

    os.environ['LOKY_MAX_CPU_COUNT'] = str(max_cores)
    os.environ['OMP_NUM_THREADS'] = str(max_cores)
    os.environ['MKL_NUM_THREADS'] = str(max_cores)
    os.environ['NUMBA_NUM_THREADS'] = str(max_cores)

    logger.info(f"Configured Windows multiprocessing for {max_cores} cores")


def use_threading_fallback():
    """
    Fallback to threading instead of multiprocessing if issues persist
    """
    os.environ['JOBLIB_MULTIPROCESSING'] = '0'
    logger.info("Switched to threading backend for joblib")