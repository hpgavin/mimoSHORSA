# Cross-Validation Guide: MATLAB vs Python mimoSHORSA

## Purpose
This guide helps you verify that the Python translation produces identical results to the (corrected) MATLAB implementation.

---

## Files for Cross-Validation

### Python
- `example_usage.py` - Python test script
- `mimoSHORSA.py` - Python implementation

### MATLAB
- `example_usage.m` - MATLAB test script (mirrors Python version)
- `mimoSHORSA.m` - MATLAB implementation (with Hermite bugs CORRECTED)

---

## Quick Start

### Run Python Version
```bash
cd /path/to/python/files
python example_usage.py
```

### Run MATLAB Version
```matlab
cd /path/to/matlab/files
example_usage
```

---

## What to Compare

### 1. Random Number Generation
Both scripts use the **same random seeds**:
- Example 1: seed = 42
- Example 2: seed = 123
- Example 3: seed = 456
- Example 4: seed = 789

**Expected**: Identical random data generation (if random number generators are compatible)

**Note**: MATLAB and Python use different random number algorithms, so while the **patterns** should be similar, exact values may differ slightly. The key is that **both versions should produce similar final results**.

### 2. Example 1: Simple 2D Polynomial

**True Model**: y = 1 + 2*x₁ + 0.5*x₂² + 0.3*x₁*x₂ + noise

**Parameters**:
- nInp = 2, nOut = 1, mData = 150
- maxOrder = 3, pTrain = 70%, pCull = 40%, tol = 0.20, scaling = 1

**Compare**:
- Number of terms retained
- Top 5 coefficient values
- Order matrix (powers on each term)
- Test correlation (should be > 0.95)

**Expected differences**:
- Random data may differ due to RNG
- Exact coefficient values may vary slightly
- Number of terms should be similar (±1-2 terms)

### 3. Example 2: Multi-Output System

**True Models**:
- y₁ = 1 + x₁ + 0.5*x₂² + 0.2*x₁*x₃ + noise
- y₂ = 0.5 + 1.5*x₂ - 0.8*x₃² + 0.3*x₁*x₂ + noise

**Parameters**:
- nInp = 3, nOut = 2, mData = 200
- maxOrder = 3, pTrain = 75%, pCull = 35%, tol = 0.18, scaling = 1

**Compare**:
- Number of terms for each output
- Top 3 coefficients for each output
- Cross-validation: both outputs should capture true structure

### 4. Example 3: High-Dimensional

**True Model**: 6-term polynomial in 5 variables

**Parameters**:
- nInp = 5, nOut = 1, mData = 300
- maxOrder = 2, pTrain = 80%, pCull = 40%, tol = 0.25, scaling = 1

**Compare**:
- Number of terms retained
- Most significant 6 terms
- Should identify: constant, x₁, x₂², x₃², x₄, x₁*x₂, x₃*x₅

### 5. Example 4: Scaling Options

Tests 3 scaling options (0, 1, 2) on ill-conditioned data

**Compare**:
- All should complete successfully (no errors)
- Number of terms for each scaling option
- Scaling option 0 may have issues (expected)
- Scaling options 1 and 2 should work well

---

## Detailed Comparison Checklist

### A. Data Generation
```
Python:                          MATLAB:
np.random.seed(42)              rand('seed', 42); randn('seed', 42);
dataX = 2*np.random.randn(...)  dataX = 2*randn(...);
```

⚠️ **Note**: Different RNG implementations mean data won't be identical, but distributions should match.

### B. Model Fitting
Compare these outputs from mimoSHORSA:

| Output | Python Variable | MATLAB Variable | What to Check |
|--------|----------------|-----------------|---------------|
| Orders | `order[0]` | `order{1}` | Matrix of term powers |
| Coefficients | `coeff[0]` | `coeff{1}` | Coefficient values |
| Mean X | `meanX` | `meanX` | Scaling mean |
| Transform X | `trfrmX` | `trfrmX` | Transformation matrix |
| Test predictions | `testModelY` | `testModelY` | Model output on test data |

### C. Key Metrics to Compare

**For each example, record**:

1. **Number of Terms**
   - Python: `order[0].shape[0]`
   - MATLAB: `size(order{1}, 1)`
   - Should be identical or very close (±2)

2. **Largest Coefficients**
   - Should identify same terms
   - Values should be within 5-10%

3. **Test Correlation**
   - Python: `np.corrcoef(testY, testModelY)[0,1]`
   - MATLAB: `corrcoef(testY', testModelY')` → (1,2) element
   - Should be within 0.01-0.02

4. **Condition Numbers**
   - Printed during model fitting
   - Should be similar (within 10%)

---

## Expected Results Summary

### Example 1: Simple 2D Polynomial

**Expected Terms**: 7-10 terms
**Key Terms to Find**:
- Constant (order = [0,0])
- Linear x₁ (order = [1,0])
- Quadratic x₂ (order = [0,2])
- Mixed x₁*x₂ (order = [1,1])

**Test Correlation**: > 0.95

### Example 2: Multi-Output System

**Output 1 Expected Terms**: 8-12 terms
**Output 2 Expected Terms**: 8-12 terms
**Test Correlation**: > 0.93 for both outputs

### Example 3: High-Dimensional

**Expected Terms**: 15-25 terms
**Test Correlation**: > 0.90

### Example 4: Scaling Options

**Scaling 0**: May fail or have poor conditioning
**Scaling 1**: Should succeed, ~10 terms
**Scaling 2**: Should succeed, ~10 terms

---

## Troubleshooting Differences

### If Number of Terms Differs by >3

**Possible causes**:
1. Different random data → Different train/test split
2. Different culling decisions at boundary cases
3. Numerical precision differences

**Solution**: Run multiple times with different seeds, compare averages

### If Coefficients Differ by >20%

**Possible causes**:
1. Bug in Python implementation (check specific function)
2. Different scaling/centering
3. Different train/test split

**Solution**: 
1. Check intermediate values (meanX, trfrmX)
2. Compare basis matrix B
3. Check Hermite function values

### If One Implementation Fails

**Check**:
1. Condition numbers (very high = ill-conditioned)
2. Scaling option compatibility
3. Error messages for clues

---

## Manual Verification Steps

### Step 1: Check Hermite Functions
```python
# Python
from mimoSHORSA import hermite
z = np.array([0, 0.5, 1.0, 1.5, 2.0])
for n in range(6):
    print(f"ψ_{n}(z) = {hermite(n, z)}")
```

```matlab
% MATLAB
z = [0, 0.5, 1.0, 1.5, 2.0];
for n = 0:5
    fprintf('ψ_%d(z) = %s\n', n, mat2str(hermite(n, z)));
end
```

**Should produce identical values** (with corrected Hermite functions)

### Step 2: Check Basis Matrix
```python
# Python
from mimoSHORSA import build_basis
Z = np.random.randn(3, 10)  # 3 inputs, 10 points
order_matrix = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]])
B = build_basis(Z, order_matrix)
print(B)
```

```matlab
% MATLAB
Z = randn(3, 10);
order_matrix = [0,0,0; 1,0,0; 0,1,0; 0,0,1];
B = build_basis(Z, order_matrix);
disp(B)
```

**Basis matrices should be identical**

### Step 3: Check Scaling
```python
# Python
from mimoSHORSA import scale_data
X = np.array([[1,2,3,4,5], [10,20,30,40,50]])
Zx, meanX, trfrmX = scale_data(X, scaling=1, flag=0)
```

```matlab
% MATLAB
X = [1,2,3,4,5; 10,20,30,40,50];
[Zx, meanX, trfrmX] = scale_data(X, 1, 0);
```

**Scaled data should match exactly**

---

## Detailed Output Log Template

Use this template to record results for comparison:

```
EXAMPLE 1: Simple 2D Polynomial
================================
                    Python          MATLAB          Match?
Random Seed:        42              42              ✓
Data Range X:       [-4.2, 3.8]     [-4.1, 3.9]     ~
Number of Terms:    9               9               ✓
Top Coefficient:    1.9823          1.9841          ✓
Test Correlation:   0.9712          0.9708          ✓
Condition Number:   45.2            44.8            ✓

Term Analysis:
  Constant:         0.9912          0.9905          ✓
  x₁ Linear:        2.0034          2.0018          ✓
  x₂ Quadratic:     0.4987          0.4992          ✓
  x₁*x₂ Mixed:      0.2976          0.2981          ✓

Notes: Minor differences due to RNG, overall excellent agreement
```

---

## Success Criteria

The Python implementation is validated if:

✅ **All 4 examples run without errors**
✅ **Number of terms within ±3 of MATLAB**
✅ **Top coefficients within 10% of MATLAB**
✅ **Test correlations within 0.02 of MATLAB**
✅ **Same terms identified as significant**
✅ **Similar condition numbers (within 20%)**

---

## Known Acceptable Differences

### Random Number Generation
- **Different**: Exact data values
- **Same**: Statistical properties, distributions

### Floating Point
- **Different**: Last 1-2 decimal places
- **Reason**: Different compilers, libraries

### Culling Order
- **Different**: May remove terms in different order if COV values are very close
- **Same**: Final model quality metrics

### Iteration Counts
- **Different**: May differ by 1-2 iterations
- **Reason**: Slightly different numerical values at decision boundaries

---

## Reporting Issues

If you find significant discrepancies:

1. **Document**:
   - Example number
   - Python vs MATLAB values
   - Error messages (if any)
   
2. **Check**:
   - Hermite function values (known bugs fixed?)
   - Intermediate calculations
   - Matrix dimensions
   
3. **Isolate**:
   - Which function differs?
   - Small test case that reproduces issue

---

## Quick Validation Command

### Python
```python
import numpy as np
from mimoSHORSA import hermite

# Test corrected Hermite functions
z = np.array([0, 1, 2])
print("Hermite order 5:", hermite(5, z))
# Should be: [0, ~2.87, ~-42.5] (corrected formula)
```

### MATLAB
```matlab
z = [0, 1, 2];
psi5 = hermite(5, z);
fprintf('Hermite order 5: [%.2f, %.2f, %.2f]\n', psi5);
% Should match Python output
```

---

**Last Updated**: October 23, 2025
**Python Translation**: Includes 3 Hermite bug fixes
**MATLAB Version**: Must use corrected Hermite functions
