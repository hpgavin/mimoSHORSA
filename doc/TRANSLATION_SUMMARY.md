# mimoSHORSA Translation Summary

## Overview
Complete translation of the MATLAB mimoSHORSA (multi-input multi-output Stochastic High Order Response Surface Algorithm) to Python.

**Translation completed successfully on October 23, 2025**

## Functions Implemented (17 total)

### Core Algorithm Functions
1. **mimoSHORSA** - Main function orchestrating the entire algorithm
2. **split_data** - Splits data into training and testing sets
3. **scale_data** - Scales data using various methods (0-4)
4. **mixed_term_powers** - Generates polynomial term combinations
5. **hermite** - Computes Hermite functions (orders 0-10) ⚠️ **BUGS FIXED**
6. **hermite_product** - Computes products of Hermite functions
7. **build_basis** - Builds design matrix with Hermite basis
8. **fit_model** - Fits model using least squares/SVD
9. **compute_model** - Evaluates model with inverse scaling
10. **evaluate_model** - Computes model statistics (R², correlation, COV)
11. **cull_model** - Removes terms with largest coefficient of variation
12. **print_model_stats** - Prints detailed model statistics

### Utility Functions
13. **clip_data** - Removes outliers from data
14. **scatter_data** - Creates pairwise scatter plot matrix (NEW - added per request)
15. **polynomial_orders** - Determines optimal polynomial orders (currently unused)
16. **rainbow** - Generates rainbow colormap
17. **format_plot** - Sets plot formatting parameters
18. **IDWinterp** - Placeholder (used only by unused polynomial_orders)

## Bug Fixes Discovered! ⚠️

Three bugs were discovered and fixed in the MATLAB hermite function:

1. **Case 5**: `4*z.^5 - 20*z.^2 + 15*z` → `4*z^5 - 20*z^3 + 15*z`
2. **Case 9**: `2520.^z^3` → `2520*z^3`
3. **Case 10**: Final term should be `-945` not `+945`

## Key Features Preserved

✅ **Matrix Operations**: All preserved using NumPy, no loops replacing matrix multiplication
✅ **Algorithm Structure**: Exact function organization as MATLAB
✅ **Hermite Basis**: Proper Hermite functions (not just polynomials) with Gaussian weight
✅ **Iterative Culling**: Adaptive model reduction based on coefficient uncertainty
✅ **Multiple Scaling**: All 5 scaling options (0-4) implemented
✅ **Documentation**: All function docstrings use ''' ''' delimiters as requested

## Dependencies
```python
numpy
matplotlib
```

## Quick Start
```python
import numpy as np
from mimoSHORSA import mimoSHORSA

# Your data: dataX (nInp x mData), dataY (nOut x mData)
order, coeff, meanX, meanY, trfrmX, trfrmY, testModelY, testX, testY = \
    mimoSHORSA(dataX, dataY, maxOrder=3, pTrain=50, pCull=30, tol=0.10, scaling=1)
```

## Scaling Options
- **0**: No scaling
- **1**: Subtract mean and divide by std.dev
- **2**: Subtract mean and decorrelate
- **3**: Log-transform, subtract mean and divide by std.dev
- **4**: Log-transform, subtract mean and decorrelate

## Reference
Gavin, HP and Yau SC, "High order limit state functions in the 
response surface method for structural reliability analysis,"
Structural Safety, December 2005.

Department of Civil and Environmental Engineering, Duke University
Siu Chung Yau, Henri P. Gavin, January 2006, 2023
