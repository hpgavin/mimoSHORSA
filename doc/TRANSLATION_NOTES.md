# MATLAB to Python Translation Notes for mimoSHORSA

## Overview
This document provides detailed notes on the translation of mimoSHORSA from MATLAB to Python, highlighting key differences, equivalences, and implementation decisions.

## Syntax Translations

### Comments
- **MATLAB**: `%` for single-line, `%{ ... %}` for multi-line
- **Python**: `#` for single-line, `''' ... '''` or `""" ... """` for docstrings

### Arrays and Matrices

#### Indexing
- **MATLAB**: 1-based indexing
  ```matlab
  x(1)        % first element
  x(end)      % last element
  x(2:5)      % elements 2 through 5
  ```
- **Python**: 0-based indexing
  ```python
  x[0]        # first element
  x[-1]       # last element
  x[1:5]      # elements at indices 1,2,3,4 (not 5)
  ```

#### Matrix Creation
- **MATLAB**: 
  ```matlab
  A = zeros(n, m)
  B = ones(n, m)
  C = NaN(n, m)
  ```
- **Python**:
  ```python
  A = np.zeros((n, m))
  B = np.ones((n, m))
  C = np.full((n, m), np.nan)
  ```

#### Matrix Operations
- **MATLAB**: 
  ```matlab
  C = A * B          % matrix multiplication
  C = A .* B         % element-wise multiplication
  ```
- **Python**:
  ```python
  C = A @ B          # matrix multiplication
  C = A * B          # element-wise multiplication (with broadcasting)
  ```

### Cell Arrays vs. Lists
- **MATLAB**: 
  ```matlab
  cells = cell(n, 1);
  cells{i} = data;
  ```
- **Python**:
  ```python
  cells = [None] * n
  cells[i] = data
  ```

### Size and Shape
- **MATLAB**: 
  ```matlab
  [rows, cols] = size(A)
  n = length(v)
  ```
- **Python**:
  ```python
  rows, cols = A.shape
  n = len(v)  # for 1D
  n = v.shape[0]  # for 2D, first dimension
  ```

### Linear Algebra

#### Solving Linear Systems
- **MATLAB**: 
  ```matlab
  x = A \ b
  ```
- **Python**:
  ```python
  from scipy.linalg import lstsq
  x, residuals, rank, s = lstsq(A, b)
  ```

#### Matrix Inverse
- **MATLAB**: 
  ```matlab
  Ainv = inv(A)
  ```
- **Python**:
  ```python
  from scipy.linalg import inv
  Ainv = inv(A)
  ```

#### Condition Number
- **MATLAB**: 
  ```matlab
  c = cond(A)
  ```
- **Python**:
  ```python
  s = np.linalg.svd(A, compute_uv=False)
  c = np.max(s) / np.min(s)
  ```

### Statistical Functions

#### Correlation Coefficient
- **MATLAB**: 
  ```matlab
  R = corrcoef(X, Y)
  ```
- **Python**:
  ```python
  R = np.corrcoef(X, Y)
  ```

#### Mean and Standard Deviation
- **MATLAB**: 
  ```matlab
  m = mean(X, 2)    % mean along dimension 2
  s = std(X, 0, 2)  % std along dimension 2
  ```
- **Python**:
  ```python
  m = np.mean(X, axis=1, keepdims=True)
  s = np.std(X, axis=1, keepdims=True)
  ```

### Special Functions

#### Hermite Polynomials
- **MATLAB**: Custom implementation or Symbolic Toolbox
- **Python**: 
  ```python
  from scipy.special import hermite
  H = hermite(n)  # returns polynomial object
  y = H(x)        # evaluate at x
  ```

### Control Flow

#### For Loops
- **MATLAB**: 
  ```matlab
  for i = 1:n
      % code
  end
  ```
- **Python**:
  ```python
  for i in range(n):  # i goes from 0 to n-1
      # code
  ```

#### If Statements
- **MATLAB**: 
  ```matlab
  if condition
      % code
  elseif condition2
      % code
  else
      % code
  end
  ```
- **Python**:
  ```python
  if condition:
      # code
  elif condition2:
      # code
  else:
      # code
  ```

### String Formatting

#### Print Statements
- **MATLAB**: 
  ```matlab
  fprintf('Value: %f\n', x);
  fprintf(sprintf('%f < x < %f\n', min(x), max(x)));
  ```
- **Python**:
  ```python
  print(f'Value: {x:.6f}')
  print(f'{np.min(x)} < x < {np.max(x)}')
  ```

### Time and Date

#### Timing
- **MATLAB**: 
  ```matlab
  tic
  % code
  elapsed = toc
  ```
- **Python**:
  ```python
  import time
  start = time.time()
  # code
  elapsed = time.time() - start
  ```

#### Date Formatting
- **MATLAB**: 
  ```matlab
  datestr(now + secs/3600/24, 14)
  ```
- **Python**:
  ```python
  from datetime import datetime, timedelta
  eta = datetime.now() + timedelta(seconds=secs)
  eta.strftime('%H:%M:%S')
  ```

## Function-by-Function Translation

### split_data()
**Key Changes:**
- Used `np.random.permutation()` instead of MATLAB's `randperm()`
- Python slicing: `idx[:mTrain]` vs MATLAB: `idx(1:mTrain)`

### scale_data()
**Key Changes:**
- Used `np.mean(axis=1, keepdims=True)` to preserve dimensions
- Eigenvalue decomposition: `np.linalg.eig()` vs MATLAB's `eig()`
- Diagonal matrix: `np.diag()` vs MATLAB's `diag()`

### mixed_term_powers()
**Key Changes:**
- Implemented recursive backtracking for generating combinations
- Used lists and `np.array()` conversion instead of growing arrays
- More explicit combination generation logic

### fit_model()
**Key Changes:**
- Used `scipy.linalg.lstsq()` for least squares fitting
- Hermite polynomials from `scipy.special.hermite()`
- Condition number calculated from singular values

### compute_model()
**Key Changes:**
- Matrix multiplication using `@` operator
- Inverse transformation: `inv(trfrmY)` vs MATLAB's `inv()`
- Power operation: `10.0 ** x` vs MATLAB's `10.^x`

### evaluate_model()
**Key Changes:**
- Used `np.corrcoef()` for correlation coefficient
- Matrix norms: `np.linalg.norm()` vs MATLAB's `norm()`
- Diagonal extraction: `np.diag()` vs MATLAB's `diag()`

### cull_model()
**Key Changes:**
- Boolean masking instead of index lists
- `np.argmax()` vs MATLAB's `max()` with index output

### Plotting (evaluate_model, main loop)
**Key Changes:**
- Matplotlib instead of MATLAB's plotting
- `plt.figure()`, `plt.plot()`, `plt.semilogy()`
- `plt.axis('square')` vs MATLAB's `axis square`
- `plt.draw()` and `plt.pause()` for interactive updates

## Key Preserved Features

### ✅ Matrix-Based Computation
All matrix operations were preserved without conversion to for-loops:
- Matrix-matrix multiplication: `@` operator
- Matrix-vector multiplication: `@` operator
- Element-wise operations: broadcasting
- Vectorized computations throughout

### ✅ Function Organization
All functions from MATLAB preserved:
1. `mimoSHORSA()` - main function
2. `split_data()` - data splitting
3. `scale_data()` - data scaling
4. `mixed_term_powers()` - term generation
5. `fit_model()` - model fitting
6. `compute_model()` - prediction
7. `evaluate_model()` - statistics
8. `cull_model()` - term removal
9. `print_model_stats()` - output formatting
10. `generate_combinations()` - helper function (new)

### ✅ Algorithm Logic
The complete algorithm flow preserved:
- Data preprocessing
- Outlier removal
- Iterative model reduction
- Statistical evaluation
- Convergence criteria

### ✅ Code Comments
- Block comments using `'''...'''` 
- Inline comments using `#`
- Docstrings for all functions

## Testing Recommendations

### Unit Tests
Create tests for each function:
```python
def test_split_data():
    dataX = np.random.randn(3, 100)
    dataY = np.random.randn(2, 100)
    trainX, trainY, mTrain, testX, testY, mTest = \
        split_data(dataX, dataY, 0.7)
    assert mTrain + mTest == 100
    assert trainX.shape[1] == mTrain
    assert testX.shape[1] == mTest

def test_scale_data():
    data = np.random.randn(3, 100)
    dataZ, meanD, trfrm = scale_data(data, 1, 0)
    assert np.allclose(np.mean(dataZ, axis=1), 0, atol=1e-10)
    assert np.allclose(np.std(dataZ, axis=1), 1, atol=1e-10)
```

### Integration Tests
Test full workflow:
```python
def test_mimoSHORSA_synthetic():
    # Create known polynomial
    np.random.seed(42)
    nInp, nOut, mData = 3, 2, 100
    dataX = np.random.randn(nInp, mData)
    dataY = np.zeros((nOut, mData))
    dataY[0, :] = 1.0 + 2.0*dataX[0, :]  # linear
    dataY[1, :] = 0.5*dataX[1, :]**2     # quadratic
    
    # Should recover coefficients
    order, coeff, *_ = mimoSHORSA(dataX, dataY, maxOrder=2)
    # Check that linear and quadratic terms are significant
```

### Numerical Validation
Compare results with MATLAB:
```python
# Save test data from MATLAB
# Compare Python results for:
# - Coefficient values
# - Model predictions
# - Statistical metrics
# - Convergence behavior
```

## Performance Considerations

### NumPy Optimizations
- Uses BLAS/LAPACK for matrix operations
- Vectorized operations via broadcasting
- Memory-efficient array operations

### Potential Bottlenecks
1. **Hermite polynomial evaluation**: May be slower than MATLAB's built-in
2. **Large polynomial bases**: Memory usage for basis matrix `B`
3. **Multiple outputs**: Separate fit for each output

### Optimization Strategies
- Use `numba` JIT compilation for polynomial evaluation
- Implement sparse matrix storage for large problems
- Parallel processing for multiple outputs using `multiprocessing`

## Common Pitfalls

### 1. Index Off-by-One Errors
```python
# MATLAB: x(1:n)
# Python: x[:n]  (not x[:n+1])
```

### 2. Dimension Handling
```python
# Keep dimensions when computing mean
mean = np.mean(data, axis=1, keepdims=True)  # shape (n, 1)
# vs
mean = np.mean(data, axis=1)  # shape (n,)
```

### 3. Broadcasting Differences
```python
# MATLAB: automatic expansion
# Python: explicit broadcasting or keepdims
```

### 4. Copy vs. View
```python
# Create copy, not view
data_copy = data.copy()
# vs
data_view = data  # reference, not copy
```

## Future Enhancements

### Potential Improvements
1. **Sparse matrix support**: For very large models
2. **Parallel computation**: Multi-output fitting
3. **GPU acceleration**: CuPy for large datasets
4. **Advanced polynomial bases**: Legendre, Chebyshev options
5. **Automatic differentiation**: For sensitivity analysis
6. **Cross-validation**: K-fold validation support
7. **Model export**: Save/load fitted models
8. **Visualization improvements**: Interactive plots

### Additional Features
- Progress bars (tqdm)
- Logging instead of print statements
- Configuration files for parameters
- Command-line interface
- Integration with scikit-learn

## Conclusion

This translation maintains the mathematical integrity and computational efficiency of the original MATLAB implementation while leveraging Python's modern scientific computing ecosystem. The code is ready for:

- Research applications in response surface methodology
- Integration into larger Python-based workflows
- Extension and customization for specific use cases
- Educational purposes in learning polynomial approximation methods

## References

1. NumPy Documentation: https://numpy.org/doc/
2. SciPy Documentation: https://docs.scipy.org/
3. Matplotlib Documentation: https://matplotlib.org/
4. Original Paper: Gavin & Yau (2008), Structural Safety
