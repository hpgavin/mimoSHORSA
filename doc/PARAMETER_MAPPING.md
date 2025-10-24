# Parameter & Function Call Mapping: Python ↔ MATLAB

Quick reference for translating between Python and MATLAB versions of mimoSHORSA.

---

## Main Function Call

### Python
```python
order, coeff, meanX, meanY, trfrmX, trfrmY, testModelY, testX, testY = \
    mimoSHORSA(dataX, dataY, maxOrder=3, pTrain=70, pCull=30, tol=0.10, scaling=1)
```

### MATLAB
```matlab
[order, coeff, meanX, meanY, trfrmX, trfrmY, testModelY, testX, testY] = ...
    mimoSHORSA(dataX, dataY, 3, 70, 30, 0.10, 1);
```

**Differences**:
- Python: Named parameters with defaults
- MATLAB: Positional parameters only

---

## Data Structure Mapping

### Python → MATLAB

| Python | MATLAB | Notes |
|--------|--------|-------|
| `list` | `cell array` | `order = [...]` → `order = {...}` |
| `order[0]` | `order{1}` | 0-based vs 1-based indexing |
| `order[0].shape[0]` | `size(order{1}, 1)` | Get number of rows |
| `coeff[0][5]` | `coeff{1}(6)` | Access 6th element |
| `np.array` | matrix | Both are matrices/arrays |

### Accessing Multi-Output Results

**Python**:
```python
for io in range(nOut):
    n_terms = order[io].shape[0]
    coeffs = coeff[io]
    print(f"Output {io}: {n_terms} terms")
```

**MATLAB**:
```matlab
for io = 1:nOut
    n_terms = size(order{io}, 1);
    coeffs = coeff{io};
    fprintf('Output %d: %d terms\n', io, n_terms);
end
```

---

## Random Number Generation

### Python
```python
np.random.seed(42)
x = np.random.randn(5, 100)
```

### MATLAB
```matlab
rand('seed', 42);   % or: rng(42, 'v5uniform')
randn('seed', 42);  % or: rng(42, 'v5normal')
x = randn(5, 100);
```

⚠️ **Warning**: Different RNG algorithms mean data won't be identical!

---

## Array Operations

### Element-wise Operations

**Python**:
```python
y = x1**2 + x2**2          # Element-wise power
y = x1 * x2                # Element-wise multiply
```

**MATLAB**:
```matlab
y = x1.^2 + x2.^2;        % Element-wise power
y = x1 .* x2;             % Element-wise multiply
```

### Matrix Operations

**Python**:
```python
C = A @ B                  # Matrix multiply
C = A.T                    # Transpose
C = np.linalg.inv(A)       # Inverse
```

**MATLAB**:
```matlab
C = A * B;                % Matrix multiply
C = A';                   % Transpose
C = inv(A);               % Inverse
```

---

## Hermite Function Comparison

### Python
```python
from mimoSHORSA import hermite

z = np.array([0, 0.5, 1.0, 1.5])
psi_0 = hermite(0, z)
psi_5 = hermite(5, z)
```

### MATLAB
```matlab
z = [0, 0.5, 1.0, 1.5];
psi_0 = hermite(0, z);
psi_5 = hermite(5, z);
```

**Should produce identical results** (with bug fixes applied).

---

## File I/O

### Python
```python
# Save results
np.save('order.npy', order)
np.save('coeff.npy', coeff)

# Load results
order = np.load('order.npy', allow_pickle=True)
coeff = np.load('coeff.npy', allow_pickle=True)
```

### MATLAB
```matlab
% Save results
save('results.mat', 'order', 'coeff');

% Load results
load('results.mat', 'order', 'coeff');
```

---

## Plotting Comparison

### Python
```python
import matplotlib.pyplot as plt

plt.figure(1)
plt.plot(x, y, 'o')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Data')
plt.grid(True)
plt.show()
```

### MATLAB
```matlab
figure(1);
plot(x, y, 'o');
xlabel('X');
ylabel('Y');
title('Data');
grid on;
```

---

## Common Pitfalls

### 1. Indexing
```python
# Python (0-based)
first_element = array[0]
last_element = array[-1]
```

```matlab
% MATLAB (1-based)
first_element = array(1);
last_element = array(end);
```

### 2. Cell Arrays vs Lists
```python
# Python
order = [None] * nOut
order[0] = np.array([[0,0], [1,0]])
```

```matlab
% MATLAB
order = cell(nOut, 1);
order{1} = [0,0; 1,0];
```

### 3. Matrix Dimensions
```python
# Python
A = np.zeros((3, 5))  # 3 rows, 5 columns
rows, cols = A.shape
```

```matlab
% MATLAB
A = zeros(3, 5);      % 3 rows, 5 columns
[rows, cols] = size(A);
```

### 4. Logical Indexing
```python
# Python
idx = x > 0
y = x[idx]
```

```matlab
% MATLAB
idx = x > 0;
y = x(idx);
```

---

## Function Call Examples from example_usage

### Example 1: Simple Polynomial

**Python**:
```python
order, coeff, meanX, meanY, trfrmX, trfrmY, testModelY, testX, testY = \
    mimoSHORSA(dataX, dataY, maxOrder=3, pTrain=70, pCull=40, 
               tol=0.20, scaling=1)
```

**MATLAB**:
```matlab
[order, coeff, meanX, meanY, trfrmX, trfrmY, testModelY, testX, testY] = ...
    mimoSHORSA(dataX, dataY, 3, 70, 40, 0.20, 1);
```

### Example 2: Multi-Output

**Python**:
```python
order, coeff, meanX, meanY, trfrmX, trfrmY, testModelY, testX, testY = \
    mimoSHORSA(dataX, dataY, maxOrder=3, pTrain=75, pCull=35, 
               tol=0.18, scaling=1)
```

**MATLAB**:
```matlab
[order, coeff, meanX, meanY, trfrmX, trfrmY, testModelY, testX, testY] = ...
    mimoSHORSA(dataX, dataY, 3, 75, 35, 0.18, 1);
```

### Example 3: High-Dimensional

**Python**:
```python
order, coeff, meanX, meanY, trfrmX, trfrmY, testModelY, testX, testY = \
    mimoSHORSA(dataX, dataY, maxOrder=2, pTrain=80, pCull=40, 
               tol=0.25, scaling=1)
```

**MATLAB**:
```matlab
[order, coeff, meanX, meanY, trfrmX, trfrmY, testModelY, testX, testY] = ...
    mimoSHORSA(dataX, dataY, 2, 80, 40, 0.25, 1);
```

---

## Verifying Identical Calculations

### Test Scaling Function

**Python**:
```python
from mimoSHORSA import scale_data
X = np.array([[1, 2, 3, 4, 5], [10, 20, 30, 40, 50]])
Zx, meanX, trfrmX = scale_data(X, scaling=1, flag=0)
print("Mean:", meanX.flatten())
print("Transform:", trfrmX.diagonal())
```

**MATLAB**:
```matlab
X = [1, 2, 3, 4, 5; 10, 20, 30, 40, 50];
[Zx, meanX, trfrmX] = scale_data(X, 1, 0);
fprintf('Mean: [%.4f, %.4f]\n', meanX);
fprintf('Transform diag: [%.4f, %.4f]\n', diag(trfrmX));
```

**Should Match Exactly**.

### Test Hermite Function (Order 5)

**Python**:
```python
from mimoSHORSA import hermite
z = np.array([0, 1, 2])
psi5 = hermite(5, z)
print("ψ_5:", psi5)
```

**MATLAB**:
```matlab
z = [0, 1, 2];
psi5 = hermite(5, z);
fprintf('ψ_5: [%.6f, %.6f, %.6f]\n', psi5);
```

**Expected** (with bug fix):
```
ψ_5: [0.000000, 2.873942, -42.547932]
```

### Test Build Basis

**Python**:
```python
from mimoSHORSA import build_basis
Z = np.array([[0, 1, 2], [0, 1, 2]])
order_matrix = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
B = build_basis(Z, order_matrix)
print("Basis shape:", B.shape)
print("Basis[0,:]:", B[0, :])
```

**MATLAB**:
```matlab
Z = [0, 1, 2; 0, 1, 2];
order_matrix = [0, 0; 1, 0; 0, 1; 1, 1];
B = build_basis(Z, order_matrix);
fprintf('Basis shape: %d x %d\n', size(B));
fprintf('Basis(1,:): [%.4f, %.4f, %.4f, %.4f]\n', B(1,:));
```

**Should Match Exactly**.

---

## Converting Existing MATLAB Code to Python

### Template Conversion

**MATLAB**:
```matlab
% Load data
data = load('mydata.mat');
X = data.X;
Y = data.Y;

% Run mimoSHORSA
[order, coeff, meanX, meanY, trfrmX, trfrmY, testModelY, testX, testY] = ...
    mimoSHORSA(X, Y, 3, 70, 30, 0.10, 1);

% Save results
save('results.mat', 'order', 'coeff');
```

**Python Equivalent**:
```python
# Load data
from scipy.io import loadmat
data = loadmat('mydata.mat')
X = data['X']
Y = data['Y']

# Run mimoSHORSA
from mimoSHORSA import mimoSHORSA
order, coeff, meanX, meanY, trfrmX, trfrmY, testModelY, testX, testY = \
    mimoSHORSA(X, Y, maxOrder=3, pTrain=70, pCull=30, tol=0.10, scaling=1)

# Save results
np.save('order.npy', order)
np.save('coeff.npy', coeff)
```

---

## Quick Reference Table

| Operation | Python | MATLAB |
|-----------|--------|--------|
| Random seed | `np.random.seed(42)` | `rand('seed',42)` |
| Array creation | `np.zeros((3,5))` | `zeros(3,5)` |
| Element power | `x**2` | `x.^2` |
| Matrix multiply | `A @ B` | `A * B` |
| Transpose | `A.T` | `A'` |
| Inverse | `np.linalg.inv(A)` | `inv(A)` |
| Cell/List | `order[0]` | `order{1}` |
| Size | `A.shape` | `size(A)` |
| Max | `np.max(x)` | `max(x)` |
| Sort | `np.sort(x)` | `sort(x)` |
| Find | `np.where(x>0)` | `find(x>0)` |

---

**Remember**: The implementations should give **statistically equivalent** results, but not **numerically identical** due to RNG differences!

**Last Updated**: October 23, 2025
