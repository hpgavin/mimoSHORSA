# mimoSHORSA - Quick Start Guide

## What You Have

This package contains a complete Python translation of the MATLAB mimoSHORSA algorithm for high-order polynomial response surface modeling.

## Files Included

1. **mimoSHORSA.py** - Main Python implementation (25 KB)
   - All functions from the original MATLAB code
   - Matrix-based computations preserved
   - Uses NumPy, SciPy, and Matplotlib

2. **example_usage.py** - Comprehensive examples (9 KB)
   - 4 complete working examples
   - Demonstrates different use cases
   - Includes visualization code

3. **README.md** - Complete documentation (8.5 KB)
   - Detailed function descriptions
   - Parameter explanations
   - Usage examples
   - Mathematical formulation

4. **TRANSLATION_NOTES.md** - Translation reference (11 KB)
   - MATLAB to Python syntax guide
   - Function-by-function comparison
   - Common pitfalls and solutions
   - Performance considerations

5. **requirements.txt** - Package dependencies
   - NumPy >= 1.18.0
   - SciPy >= 1.4.0
   - Matplotlib >= 3.1.0

## Installation (3 Simple Steps)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Place Files in Your Project
```bash
# Copy mimoSHORSA.py to your project directory
cp mimoSHORSA.py /path/to/your/project/
```

### Step 3: Import and Use
```python
from mimoSHORSA import mimoSHORSA
# Ready to use!
```

## Quick Example (Copy & Paste)

```python
import numpy as np
from mimoSHORSA import mimoSHORSA

# Generate synthetic data
np.random.seed(42)
nInp = 3   # 3 input variables
nOut = 2   # 2 output variables
mData = 100

dataX = np.random.randn(nInp, mData)
dataY = np.zeros((nOut, mData))
dataY[0, :] = 1.0 + 2.0*dataX[0, :] + 0.5*dataX[1, :]**2
dataY[1, :] = 0.5 + 1.5*dataX[1, :] - 0.8*dataX[2, :]**2

# Fit model
order, coeff, meanX, meanY, trfrmX, trfrmY, testModelY, testX, testY = \
    mimoSHORSA(dataX, dataY, maxOrder=3, pTrain=70, pCull=30, 
               tol=0.15, scaling=1)

# Results stored in order and coeff
print(f"Model has {order[0].shape[0]} terms for output 0")
print(f"Coefficients: {coeff[0]}")
```

## Run Examples

```bash
python example_usage.py
```

This will run 4 different examples:
1. Simple 2D polynomial
2. Multi-output system
3. High-dimensional problem
4. Scaling comparison

## Key Features

✅ **Preserved from MATLAB:**
- All matrix operations (no for-loops)
- Same function organization
- Hermite polynomial basis
- Statistical metrics (R², correlation, COV)

✅ **Python Advantages:**
- Easy integration with other Python tools
- Modern scientific computing stack
- Clear, readable syntax
- Easy to extend and modify

## Common Parameters

| Parameter | Typical Range | Description |
|-----------|---------------|-------------|
| maxOrder | 2-4 | Higher = more complex model |
| pTrain | 60-80 | Percent for training |
| pCull | 20-40 | Max percent to remove |
| tol | 0.10-0.25 | Coefficient variation threshold |
| scaling | 0-4 | 0=none, 1=standardize (recommended) |

## When to Use Each Scaling Option

- **scaling=0**: Data already normalized
- **scaling=1**: General purpose (RECOMMENDED)
- **scaling=2**: Correlated inputs
- **scaling=3**: Exponential/log data
- **scaling=4**: Exponential data with correlation

## Troubleshooting

### Import Error
```bash
# Make sure packages are installed
pip install numpy scipy matplotlib
```

### "Module not found: mimoSHORSA"
```bash
# Make sure mimoSHORSA.py is in the same directory
# or in your Python path
```

### Poor Model Fit
- Try different scaling options (especially scaling=1)
- Increase maxOrder (but not too high!)
- Increase pTrain percentage
- Check for outliers in data
- Reduce tol if removing too many terms

### Numerical Issues
- Always use scaling=1 or scaling=2
- Check condition numbers in output
- Reduce maxOrder if condition number > 1000
- Remove outliers beyond ±4 standard deviations

## Next Steps

1. **Read README.md** for detailed documentation
2. **Run example_usage.py** to see examples
3. **Check TRANSLATION_NOTES.md** for MATLAB comparison
4. **Adapt to your data** by modifying the quick example above

## Support & Reference

**Original Algorithm:**
- Gavin, H.P. and Yau, S.C. (2008)
- "High order limit state functions in the response surface method"
- *Structural Safety*

**Python Implementation:**
- Complete translation: 2025
- Preserves all functionality
- Adds modern Python features

## File Organization

```
your_project/
├── mimoSHORSA.py          # Main algorithm (required)
├── example_usage.py       # Examples (optional)
├── requirements.txt       # Dependencies (for installation)
├── README.md              # Full documentation (reference)
├── TRANSLATION_NOTES.md   # MATLAB comparison (reference)
└── your_script.py         # Your code using mimoSHORSA
```

## Minimal Working Example

```python
# minimal_example.py
import numpy as np
from mimoSHORSA import mimoSHORSA

# Your data: dataX (inputs), dataY (outputs)
# dataX shape: (n_inputs, n_samples)
# dataY shape: (n_outputs, n_samples)

# Fit model
result = mimoSHORSA(dataX, dataY)
order, coeff = result[0], result[1]

# That's it! Model is fitted.
```

## What the Algorithm Does

1. **Splits** data into training (70%) and testing (30%)
2. **Scales** data for numerical stability
3. **Generates** all polynomial terms up to maxOrder
4. **Fits** model using least squares with Hermite polynomials
5. **Evaluates** goodness of fit (R², correlation, etc.)
6. **Removes** terms with uncertain coefficients
7. **Repeats** steps 4-6 until convergence
8. **Returns** final model with significant terms only

## Performance Tips

- Start with maxOrder=2 or 3
- Use scaling=1 for most cases
- Larger pTrain = better fit, less validation
- Smaller tol = more aggressive term removal
- Monitor condition numbers (printed during run)

---

**You're ready to go! Start with the quick example above or run example_usage.py**
