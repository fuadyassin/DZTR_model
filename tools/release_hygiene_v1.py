"""Recognize generated development files at repository and bundle boundaries."""
from pathlib import PurePosixPath


GENERATED_DIRECTORIES = frozenset({
    '__pycache__', '.ipynb_checkpoints', '.pytest_cache', '.mypy_cache', '.ruff_cache',
})


def is_generated_file(name):
    """Handle Git paths and native Windows paths on either CI platform."""
    path = PurePosixPath(str(name).replace('\\', '/'))
    return (any(part in GENERATED_DIRECTORIES for part in path.parts)
            or path.suffix.lower() in {'.pyc', '.pyo'})
