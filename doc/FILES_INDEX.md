# mimoSHORSA Python Translation - File Index

## Main Files

### mimoSHORSA.py (692 lines)
Complete Python translation of the MATLAB mimoSHORSA program with all 9 functions:
- mimoSHORSA() - Main driver function
- split_data() - Data partitioning
- scale_data() - Data transformation
- mixed_term_powers() - Polynomial term generation
- fit_model() - Least squares fitting
- compute_model() - Model evaluation
- evaluate_model() - Statistical analysis
- cull_model() - Term removal
- print_model_stats() - Results display

## Documentation

### README.md
Comprehensive documentation including:
- Overview of the algorithm
- Installation instructions
- Parameter descriptions
- Function reference
- Usage examples
- Translation notes

### QUICK_START.md
Quick reference guide with:
- Installation command
- Basic usage pattern
- Parameter summary
- Available functions

### TRANSLATION_SUMMARY.md
Detailed translation report covering:
- Translation approach
- Function-by-function mapping
- MATLAB to Python conversions
- Special considerations
- Testing and validation

## Examples

### test_example.py
Simple, self-contained example demonstrating:
- Synthetic data generation
- Basic mimoSHORSA usage
- Result interpretation

## Usage Workflow

1. Start with **QUICK_START.md** for immediate usage
2. Review **test_example.py** for a working example
3. Consult **README.md** for detailed documentation
4. Check **TRANSLATION_SUMMARY.md** for technical details

## Key Features

✓ All matrix operations preserved (no loops replacing matrix multiplication)
✓ Function structure identical to MATLAB version
✓ Block comments use '''...''' format
✓ Comprehensive docstrings for each function
✓ Example code included with test data
✓ Tested and verified to run correctly

## Requirements

- Python 3.6+
- NumPy >= 1.20
- SciPy >= 1.7
- Matplotlib >= 3.3

Install with:
```bash
pip install numpy scipy matplotlib --break-system-packages
```

## Original Reference

Original MATLAB code by:
- Siu Chung Yau
- Henri P. Gavin

Duke University
Department of Civil and Environmental Engineering
2006-2023
