# Documentation and Code Organization Strategy

This document describes the conventions used for organizing code and documentation in this repository.

## Philosophy

**Principles:**
1. **Code first, docs second** - Show usage before explaining
2. **Progressive disclosure** - Simple examples → Complex use cases
3. **Single source of truth** - One README per major directory
4. **Separation of concerns** - Clear module boundaries

---

## Naming Conventions

### Functions and Variables

**Use verbs for boolean flags:**
```python
# Good
augment = True
tokenize = False
shuffle = True

# Avoid
apply_augmentation = True  # Redundant "apply"
return_tokenized = False   # Redundant "return"
```

**Use nouns for data:**
```python
# Good
data_dir = Path('/data')
class_to_idx = {'healthy': 0}
ct_window_probs = [0.7, 0.15, 0.15]

# Avoid
dir_for_data = Path('/data')  # Unnatural word order
```

**Standard PyTorch conventions:**
```python
transform = ...      # Single transform
transforms = ...     # Multiple transforms (torchvision)
class_to_idx = {}   # Label mapping (standard)
```

### Classes

**Use nouns, be descriptive:**
```python
# Good
class ImageDataset(Dataset)
class StudyAwareBatchSampler(Sampler)

# Avoid
class MyDataset(Dataset)           # Non-descriptive
class DatasetForImages(Dataset)    # Unnatural word order
```

### Arguments

**Order by importance:**
```python
def __init__(
    self,
    data_dir,           # Required, most important
    use_cache=True,     # Core functionality
    mode_filter=None,   # Common filter
    random_crop=False,  # Training option
    augment=False,      # Training option
    tokenize=True,      # Format option
):
```

---

## Documentation Structure

### Module-Level Docstrings

**Format:**
```python
"""
Module Name

Brief description of purpose (1-2 sentences).
List key components if multiple.

Example usage (only if complex).
"""
```

**Example:**
```python
"""
PyTorch Dataset and Batch Sampler

ImageDataset: Loads individual images with augmentation and tokenization
StudyAwareBatchSampler: Groups images from same study into batches
"""
```

### Function/Class Docstrings (Google Style)

**Format:**
```python
def function_name(arg1, arg2, kwarg1=None):
    """
    One-line summary (imperative mood).
    
    Additional context (2-3 sentences max, only if needed).
    
    Args:
        arg1 (type): Description
        arg2 (type): Description
        kwarg1 (type, optional): Description. Defaults to None.
    
    Returns:
        type: Description
    
    Raises:
        ErrorType: When this happens
    
    Example:
        >>> result = function_name('value', 42)
        >>> print(result)
        Expected output
    
    Notes:
        - Important caveat 1
        - Important caveat 2
    """
```

**Rules:**
- One-line summary: Start with verb, <80 chars
- Args: Always include type, be concise
- Example: Only if usage is non-obvious
- Notes: Only for caveats/gotchas

### README Structure

**Template:**
```markdown
# Title

Quick Start (code first!)
Directory Structure
Data Format
API Reference (minimal)
Examples (2-3 patterns)
Troubleshooting

Notes section at end
```

**Example outline:**
```markdown
# Medical Image Data Pipeline

## Quick Start
## Directory Structure  
## Data Format
## Preprocessing Pipeline
## API
### load_image
### prepare_for_inference
## Datasets
### ImageDataset
### StudyAwareBatchSampler
## Examples
### Image-level training
### Study-level training
## Troubleshooting
```

**Rules:**
- Start with runnable code (Quick Start)
- Keep API docs minimal (defer to docstrings)
- Show 2-3 usage patterns, not 10
- Put design rationale in "Notes" at end

---

## Directory Organization

### High-Level Structure

```
project/
├── data/           # Core data operations (format-agnostic)
├── datasets/       # PyTorch integration
├── models/         # Model architectures
├── trainers/       # Training loops
├── pipelines/      # End-to-end workflows
└── utils/          # Cross-cutting utilities
```

### Per-Directory Structure

```
data/
├── __init__.py     # Exports only (no logic)
├── io.py           # Loading functions
├── preprocess.py   # Processing functions
├── utils.py        # Helpers
├── metadata.py     # Metadata management
├── cache.py        # Cache operations
└── README.md       # Complete documentation
```

**Rules:**
1. **One concept per file**: Don't mix loading and preprocessing
2. **Clear dependencies**: `io.py` → `preprocess.py` → `cache.py`
3. **Minimal `__init__.py`**: Only exports, imports at top level
4. **One README per directory**: Not per file

### File Naming

```python
# Good
io.py              # Short, descriptive
preprocess.py      # Single word when possible
dataset.py         # Singular
metadata.py        # Noun

# Avoid
data_io.py         # Redundant with directory name
preprocessing.py   # Too long (use preprocess.py)
datasets.py        # Use singular
metadata_utils.py  # Put utils in utils.py
```

---

## Code Organization Within Files

### Import Order

```python
# 1. Standard library
import os
from pathlib import Path
from typing import Dict, List, Optional

# 2. Third-party
import torch
import numpy as np
from einops import rearrange

# 3. Local imports
from neurovfm.data.io import load_image
from neurovfm.data.preprocess import prepare_for_inference
```

### Class Structure

```python
class MyClass:
    """Docstring"""
    
    def __init__(self, ...):
        """Constructor"""
        # Public attributes first
        self.public_attr = ...
        
        # Private attributes (with underscore)
        self._private_attr = ...
    
    # Public methods
    def public_method(self):
        """Public API"""
        pass
    
    # Private methods (with underscore)
    def _private_method(self):
        """Internal helper"""
        pass
    
    # Special methods at end
    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass
```

---

## Type Hints

**Always use type hints for:**
- Function signatures
- Complex data structures
- Return values

```python
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path

def load_data(
    path: Union[str, Path],
    mode: str,
    cache: Optional[Dict] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Load and return data."""
    ...
```

**Rules:**
- Use `Optional[T]` for nullable types
- Use `Union[A, B]` for multiple types
- Use `Path` for filesystem paths
- Import types at top of file

---

## Examples in Documentation

### Good Example

```python
# Quick Start

from neurovfm.data import load_image, prepare_for_inference

# Load any image
img = load_image('scan.nii.gz')

# Prepare for model
img_arrs, mask, view = prepare_for_inference(img, mode='mri')
```

**Why good:**
- ✅ Complete, runnable code
- ✅ Shows imports
- ✅ Minimal comments
- ✅ Clear variable names

### Bad Example

```python
# First you need to import the module
from neurovfm.data import load_image

# Then you can load an image by passing the path
# The path can be either a string or a Path object
img = load_image(path_to_your_image)  # This loads the image

# Now preprocess it
from neurovfm.data import prepare_for_inference
result = prepare_for_inference(img, mode='mri')  # mode can be 'mri' or 'ct'
```

**Why bad:**
- ❌ Over-commented
- ❌ Explains obvious things
- ❌ Vague variable names (`path_to_your_image`)
- ❌ Scattered imports

---

## README Template

```markdown
# Module Name

Brief description (1-2 sentences).

## Quick Start

\`\`\`python
# Runnable code here
\`\`\`

## Directory Structure

\`\`\`
dir/
├── file1.py
└── file2.py
\`\`\`

## API

### `function_name()`
Brief description.

\`\`\`python
# Usage example
\`\`\`

## Examples

### Example 1: Common Use Case

\`\`\`python
# Complete code
\`\`\`

### Example 2: Advanced Use Case

\`\`\`python
# Complete code
\`\`\`

## Troubleshooting

**"Error message"**  
→ Solution

---

## Notes

Design decisions, caveats, future work.
```

---

## When to Create New Files

### Create new file when:
- ✅ New major concept (e.g., `cache.py` for caching)
- ✅ File > 500 lines
- ✅ Clear responsibility boundary

### Don't create new file when:
- ❌ Just a few helper functions (put in `utils.py`)
- ❌ Tightly coupled to existing file
- ❌ File would be < 100 lines

---

## Checklist for New Code

Before committing:

- [ ] Type hints on all function signatures
- [ ] Docstring for all public functions/classes
- [ ] Examples in docstring for non-obvious usage
- [ ] Updated README if public API changed
- [ ] No linter errors
- [ ] Naming follows conventions
- [ ] Imports organized correctly

---

## Summary

**Quick Reference:**

| Aspect | Convention |
|--------|-----------|
| **Bool args** | `augment`, `tokenize` (verbs) |
| **Data args** | `data_dir`, `class_to_idx` (nouns) |
| **Classes** | `ImageDataset` (descriptive nouns) |
| **Files** | `io.py`, `preprocess.py` (short) |
| **Docs** | Google style, code-first |
| **README** | Quick start → Examples → API |
| **Structure** | One concept per file |
| **Imports** | stdlib → third-party → local |

**Core Principle:** Make it easy for users to get started, then progressively reveal complexity.

