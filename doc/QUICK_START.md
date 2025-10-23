# mimoSHORSA Quick Start Guide

## Installation

```bash
pip install numpy scipy matplotlib --break-system-packages
```

## Basic Usage

```python
import numpy as np
from mimoSHORSA import mimoSHORSA

# Your data: input features (rows) × samples (columns)
dataX = np.array([...])  # shape: (nInp, mData)
dataY = np.array([...])  # shape: (nOut, mData)

# Run analysis
results = mimoSHORSA(dataX, dataY, maxOrder=3, pTrain=50, pCull=30, tol=0.10, scaling=1)
order, coeff, meanX, meanY, trfrmX, trfrmY, testModelY, testX, testY = results
```

## Example

See `test_example.py` for a complete working example.

## Parameters

- `maxOrder`: Maximum polynomial order (default: 3)
- `pTrain`: Training data percentage (default: 50)
- `pCull`: Maximum culling percentage (default: 30)
- `tol`: Coefficient of variation tolerance (default: 0.10)
- `scaling`: Data scaling option 0-4 (default: 0)

## Output

- `order`: List of polynomial term orders for each output
- `coeff`: List of coefficients for each output
- `meanX`, `meanY`: Data means
- `trfrmX`, `trfrmY`: Transformation matrices
- `testModelY`: Model predictions on test data
- `testX`, `testY`: Test data sets

## Key Functions

All 9 functions from the MATLAB version are available:

1. `mimoSHORSA()` - Main function
2. `split_data()` - Train/test split
3. `scale_data()` - Data scaling
4. `mixed_term_powers()` - Generate polynomial terms
5. `fit_model()` - Least squares fitting
6. `compute_model()` - Model evaluation
7. `evaluate_model()` - Goodness of fit
8. `cull_model()` - Term removal
9. `print_model_stats()` - Display results

## Need Help?

See `README.md` for detailed documentation.
See `TRANSLATION_SUMMARY.md` for translation details.
