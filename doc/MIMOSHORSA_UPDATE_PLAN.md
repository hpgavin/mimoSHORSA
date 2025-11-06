# mimoSHORSA Update Plan - Python Translation

## Summary of Changes from MATLAB

Based on the latest MATLAB version, the following updates are needed:

---

## 1. **Function Signature Changes**

### Old (Python current):

```python
def mimoSHORSA(dataX, dataY, maxOrder=3, pTrain=50, pCull=30, tol=0.10, scaling=1):
```

### New (MATLAB latest):

```python
def mimoSHORSA(dataX, dataY, maxOrder=3, pTrain=50, pCull=30, tol=0.10, scaling=1, L1_pnlty=1.0, basis_fctn='H'):
```

**New parameters**:

- `L1_pnlty`: L1 regularization coefficient (default: 1.0)
- `basis_fctn`: Basis function type ('H'=Hermite, 'L'=Legendre, 'P'=Power) (default: 'H')

---

## 2. **L1 Regularization Logic**

Add at beginning of function:

```python
# No "culling" with L1 regularization
if L1_pnlty > 0:
    pCull = 0
```

---

## 3. **fit_model Function Updates**

### Old signature:

```python
def fit_model(Zx, Zy, order, nTerm, mData):
```

### New signature:

```python
def fit_model(Zx, Zy, order, nTerm, mData, L1_pnlty, basis_fctn):
```

### Implementation changes:

```python
def fit_model(Zx, Zy, order, nTerm, mData, L1_pnlty, basis_fctn):
    print(f'Fit The Model ... with L1_pnlty = {L1_pnlty}')

    B = build_basis(Zx, order, basis_fctn)
    nTerms = B.shape[1]

    if L1_pnlty > 0:
        # Use L1_fit for regularization
        from L1_fit import L1_fit
        from L1_plots import L1_plots

        coeff, mu, nu, cvg_hst = L1_fit(B, Zy, L1_pnlty, w=0)
        L1_plots(B, coeff, Zy, cvg_hst, L1_pnlty, 0, fig_no=7000)
    else:
        # Use SVD / ordinary least squares
        coeff = np.linalg.lstsq(B, Zy, rcond=None)[0]

    condB = np.linalg.cond(B)
    print(f'  condition number of model basis matrix = {condB:6.1f}')

    return coeff, condB
```

---

## 4. **build_basis Function Updates**

### Old signature:

```python
def build_basis(Zx, order):
```

### New signature:

```python
def build_basis(Zx, order, basis_fctn='H'):
```

### Implementation:

```python
def build_basis(Zx, order, basis_fctn='H'):
    """
    Build basis matrix using specified basis functions.

    Parameters
    ----------
    Zx : ndarray (nInp x mData)
        Input data matrix
    order : ndarray (nTerm x nInp)
        Powers for each term
    basis_fctn : str
        'H': Hermite functions
        'L': Legendre polynomials
        'P': Power polynomials

    Returns
    -------
    B : ndarray (mData x nTerm)
        Basis matrix
    """
    mData = Zx.shape[1]
    nTerm, nInp = order.shape
    B = np.ones((mData, nTerm))

    max_order = int(np.max(order))

    if basis_fctn == 'P':
        # Power polynomials
        for it in range(nTerm):
            B[:, it] = np.prod(Zx.T ** order[it, :], axis=1)
    else:
        # Legendre or Hermite
        for it in range(nTerm):
            B[:, it] = polynomial_product(order[it, :], Zx.T, max_order, basis_fctn)

    return B
```

---

## 5. **compute_model Function Updates**

### Old signature:

```python
def compute_model(order, coeff, meanX, meanY, trfrmX, trfrmY, X, scaling):
```

### New signature:

```python
def compute_model(order, coeff, meanX, meanY, trfrmX, trfrmY, X, scaling, basis_fctn='H'):
```

Update all calls to build_basis:

```python
B = build_basis(Zx, order[io], basis_fctn)
```

---

## 6. **Main Loop Updates**

Update fit_model call:

```python
# Old:
coeff[io], condB[io, iter] = fit_model(trainZx, trainZy[io, :], order[io], nTerm[io], mTrain)

# New:
coeff[io], condB[io, iter] = fit_model(trainZx, trainZy[io, :], order[io], nTerm[io], mTrain, L1_pnlty, basis_fctn)
```

Update compute_model calls:

```python
# Old:
trainModelY, B = compute_model(order, coeff, meanX, meanY, trfrmX, trfrmY, trainX, scaling)
testModelY, _ = compute_model(order, coeff, meanX, meanY, trfrmX, trfrmY, testX, scaling)

# New:
trainModelY, B = compute_model(order, coeff, meanX, meanY, trfrmX, trfrmY, trainX, scaling, basis_fctn)
testModelY, _ = compute_model(order, coeff, meanX, meanY, trfrmX, trfrmY, testX, scaling, basis_fctn)
```

Update culling logic:

```python
# Only cull if L1_pnlty == 0
if L1_pnlty == 0:
    order, nTerm, coeffCOV = cull_model(coeff, order, coeffCOV, tol)
```

---

## 7. **polynomial_product Function**

Need to handle both 'L' (Legendre) and 'H' (Hermite) basis:

```python
def polynomial_product(powers, Zx, max_order, basis_fctn):
    """
    Compute product of orthogonal polynomials.

    Parameters
    ----------
    powers : array (nInp,)
        Powers for each variable
    Zx : array (mData, nInp)
        Input data
    max_order : int
        Maximum order
    basis_fctn : str
        'H': Hermite, 'L': Legendre

    Returns
    -------
    result : array (mData,)
        Product of polynomials
    """
    mData, nInp = Zx.shape
    result = np.ones(mData)

    for i in range(nInp):
        if powers[i] > 0:
            if basis_fctn == 'H':
                result *= hermite(int(powers[i]), Zx[:, i])
            elif basis_fctn == 'L':
                result *= legendre(int(powers[i]), Zx[:, i])

    return result
```

---

## 8. **Add Legendre Function**

Need to implement Legendre polynomials:

```python
def legendre(n, z):
    """
    Compute Legendre polynomial of order n.

    Uses recurrence relation:
    P_0(z) = 1
    P_1(z) = z
    P_{n+1}(z) = ((2n+1)*z*P_n(z) - n*P_{n-1}(z)) / (n+1)

    Parameters
    ----------
    n : int
        Order of polynomial
    z : array
        Points to evaluate

    Returns
    -------
    P_n : array
        Legendre polynomial of order n evaluated at z
    """
    z = np.asarray(z)

    if n == 0:
        return np.ones_like(z)
    elif n == 1:
        return z
    else:
        # Recurrence relation
        P_nm1 = np.ones_like(z)  # P_0
        P_n = z                   # P_1

        for k in range(1, n):
            P_np1 = ((2*k + 1) * z * P_n - k * P_nm1) / (k + 1)
            P_nm1 = P_n
            P_n = P_np1

        return P_n
```

---

## 9. **Documentation Updates**

Update docstring for main function to include new parameters:

```python
'''
INPUT       DESCRIPTION                                                DEFAULT
--------    --------------------------------------------------------   -------
dataX       m observations of n input  features in a (nx x m) matrix
dataY       m observations of m output features in a (ny x m) matrix
maxOrder    maximum allowable polynomial order                            3
pTrain      percentage of data for training (remaining for testing)      50
pCull       maximum percentage of model which may be culled              30 
tol         desired maximum model coefficient of variation                0.10
scaling     scale the data before fitting                                 1
            scaling = 0 : no scaling
            scaling = 1 : subtract mean and divide by std.dev
            scaling = 2 : subtract mean and decorrelate
            scaling = 3 : log-transform, subtract mean and divide by std.dev
            scaling = 4 : log-transform, subtract mean and decorrelate
L1_pnlty    coefficient for L1 regularization                             1.0
basis_fctn  'H': Hermite, 'L': Legendre, 'P': Power polynomial            'H'
'''
```

---

## 10. **Import Requirements**

Add at top of file:

```python
# Optional: only imported if L1_pnlty > 0
# from L1_fit import L1_fit
# from L1_plots import L1_plots
```

---

## Summary of Files to Update

1. **mimoSHORSA.py**
   
   - Main function signature
   - fit_model function
   - build_basis function
   - compute_model function
   - polynomial_product function
   - Add legendre function
   - Update all function calls

2. **mimoSHORSA_test.py** (if exists)
   
   - Update function calls to include new parameters

3. **example_usage.py**
   
   - Update function calls
   - Add examples with L1_pnlty and different basis_fctn

---

## Testing Checklist

- [ ] Test with L1_pnlty = 0 (COV culling, should match old behavior)
- [ ] Test with L1_pnlty > 0 (L1 regularization, no culling)
- [ ] Test with basis_fctn = 'H' (Hermite - default)
- [ ] Test with basis_fctn = 'L' (Legendre)
- [ ] Test with basis_fctn = 'P' (Power polynomial)
- [ ] Cross-validate with MATLAB version
- [ ] Test all scaling options
- [ ] Test multi-output cases

---

**Status**: Ready for implementation
**Date**: 2025-11-06
**Priority**: High - integrates L1_fit and basis 
