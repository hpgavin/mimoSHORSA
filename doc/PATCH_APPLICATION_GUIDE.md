# mimoSHORSA Updates - Quick Reference Guide

## 📦 Patch Files Delivered

1. **[mimoSHORSA_updates.patch](computer:///mnt/user-data/outputs/mimoSHORSA_updates.patch)**
   
   - Main mimoSHORSA.py updates
   - 13 changes + 1 new function (Legendre)

2. **[mimoSHORSA_example_updates.patch](computer:///mnt/user-data/outputs/mimoSHORSA_example_updates.patch)**
   
   - Example/test file updates
   - New examples demonstrating L1 and basis options

3. **[MIMOSHORSA_UPDATE_PLAN.md](computer:///mnt/user-data/outputs/MIMOSHORSA_UPDATE_PLAN.md)**
   
   - Detailed implementation plan

---

## 🎯 How to Apply the Patches

### Step 1: Backup Your Files

```bash
cp mimoSHORSA.py mimoSHORSA.py.backup
cp example_usage.py example_usage.py.backup
```

### Step 2: Open Patch Files

```bash
# View the patches
cat mimoSHORSA_updates.patch
cat mimoSHORSA_example_updates.patch
```

### Step 3: Apply Changes Manually

For each "CHANGE N" in the patch file:

1. Find the "OLD CODE" in your file
2. Replace with "NEW CODE"
3. For "INSERT" sections, add the new code at the specified location

### Step 4: Add the Legendre Function

Add this function after the `hermite()` function in mimoSHORSA.py:

```python
def legendre(n, z):
    '''Legendre polynomial of order n'''
    z = np.asarray(z)

    if n == 0:
        return np.ones_like(z)
    elif n == 1:
        return z
    else:
        P_nm1 = np.ones_like(z)
        P_n = z

        for k in range(1, n):
            P_np1 = ((2*k + 1) * z * P_n - k * P_nm1) / (k + 1)
            P_nm1 = P_n
            P_n = P_np1

        return P_n
```

---

## ✅ Verification Checklist

After applying patches:

- [ ] Function signature updated: `def mimoSHORSA(..., L1_pnlty=1.0, basis_fctn='H')`
- [ ] L1 logic added: `if L1_pnlty > 0: pCull = 0`
- [ ] `fit_model()` signature updated with new parameters
- [ ] `build_basis()` signature updated with `basis_fctn`
- [ ] `compute_model()` signature updated with `basis_fctn`
- [ ] `polynomial_product()` signature updated with `basis_fctn`
- [ ] `legendre()` function added
- [ ] L1_fit import added (in fit_model function)
- [ ] All function calls updated with new parameters
- [ ] Culling logic wrapped: `if L1_pnlty == 0: cull_model(...)`

---

## 🧪 Testing

### Test 1: Backward Compatibility (Traditional COV Culling)

```python
# Should work exactly like before
order, coeff, *rest = mimoSHORSA(
    dataX, dataY,
    maxOrder=3,
    pTrain=70,
    pCull=40,
    tol=0.20,
    scaling=1,
    L1_pnlty=0,      # Traditional culling
    basis_fctn='H'   # Hermite (default)
)
```

### Test 2: L1 Regularization

```python
# New feature: L1 regularization (no culling)
order, coeff, *rest = mimoSHORSA(
    dataX, dataY,
    maxOrder=3,
    pTrain=70,
    pCull=0,         # Ignored (automatic)
    tol=0.20,
    scaling=2,       # Decorrelation recommended
    L1_pnlty=50,     # L1 regularization
    basis_fctn='L'   # Legendre basis
)
```

### Test 3: Power Polynomial Basis

```python
# For polynomial data
order, coeff, *rest = mimoSHORSA(
    dataX, dataY,
    maxOrder=2,
    pTrain=70,
    pCull=0,
    tol=0.10,
    scaling=1,
    L1_pnlty=10,
    basis_fctn='P'   # Power polynomials
)
```

---

## 🎓 New Parameter Guide

### L1_pnlty (L1 Regularization Penalty)

**Default**: 1.0

**Values**:

- `0`: Disable L1, use traditional COV culling
- `0.1 - 10`: Light regularization (gentle sparsity)
- `10 - 100`: Medium regularization (moderate sparsity)
- `100+`: Strong regularization (high sparsity)

**Guidelines**:

- Start with `L1_pnlty = 10`
- Increase if model has too many terms
- Decrease if test correlation drops significantly
- Use decorrelation scaling (`scaling=2` or `4`) with L1

### basis_fctn (Basis Function Type)

**Default**: 'H' (Hermite)

**Options**:

- **'H'**: Hermite functions
  
  - Good for: General use, reliability analysis
  - Properties: Orthogonal, exp(-z²/2) decay
  - Best with: Standardized data

- **'L'**: Legendre polynomials
  
  - Good for: Polynomial data, bounded domains
  - Properties: Orthogonal on [-1,1], no decay
  - Best with: Decorrelated data

- **'P'**: Power polynomials
  
  - Good for: Known polynomial models
  - Properties: Exact polynomial representation
  - Best with: Any scaling, ideal for validation

**Recommendation**:

- For new problems: Start with 'L' (Legendre)
- For polynomial validation: Use 'P' (Power)
- For original mimoSHORSA: Use 'H' (Hermite)

---

## 🔍 Troubleshooting

### Import Error: L1_fit not found

```python
# Make sure L1_fit.py is in your PYTHONPATH
import sys
sys.path.append('/path/to/L1_fit')

# Or the patch will fall back to OLS automatically
```

### Results Different from MATLAB

Check:

1. Random seed set the same? (`np.random.seed(42)`)
2. Same basis function? ('H', 'L', or 'P')
3. Same L1_pnlty value?
4. Same scaling option?

### Too Many/Few Terms with L1

Adjust `L1_pnlty`:

- Too many terms → Increase L1_pnlty
- Too few terms → Decrease L1_pnlty
- Try values: 1, 5, 10, 20, 50, 100

---

## 📊 Expected Behavior Changes

| Scenario       | Old Behavior   | New Behavior                  |
| -------------- | -------------- | ----------------------------- |
| L1_pnlty=0     | COV culling    | COV culling (same)            |
| L1_pnlty>0     | N/A            | L1 regularization, no culling |
| basis_fctn='H' | Always Hermite | Hermite (same)                |
| basis_fctn='L' | N/A            | Legendre polynomials          |
| basis_fctn='P' | N/A            | Power polynomials             |

---

## 💡 Tips

1. **Start Simple**: Test with `L1_pnlty=0, basis_fctn='H'` (old behavior)

2. **Then Try L1**: Set `L1_pnlty=10` and compare results

3. **Experiment with Bases**: Try 'H', 'L', 'P' on same data

4. **Use Decorrelation**: `scaling=2` or `4` works well with L1

5. **Check Convergence**: L1_plots will show if L1_fit converged

6. **Compare**: Run same data with L1_pnlty=0 vs L1_pnlty>0

---

## 🎉 Summary

**What Changed**:

- ✅ Added L1 regularization support
- ✅ Added Legendre polynomial basis
- ✅ Added Power polynomial basis
- ✅ Integrated L1_fit.py
- ✅ Maintained backward compatibility

**Lines Changed**: ~50 lines in mimoSHORSA.py + 1 new function

**Backward Compatible**: Yes (when L1_pnlty=0, basis_fctn='H')

**Ready to Use**: Yes, after applying patches

---

**Good luck with the updates! Let me know if you have any questions!** 🚀
