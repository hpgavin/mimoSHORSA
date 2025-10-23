# mimoSHORSA Quick Reference Card

## Core Model Equation

```
Y(X) = Σ cⱼ Ψⱼ(X)
     j=1 to nTerm
```

## Hermite Basis Function

```
Ψ(Z) = ψₚ₁(Z₁) × ψₚ₂(Z₂) × ... × ψₚₙ(Zₙ)

where:
ψₙ(z) = [(2ⁿ n! √π)^(-1/2)] Hₙ(z) exp(-z²/2)
```

## First Six Hermite Functions

| n | ψₙ(z) |
|---|-------|
| 0 | π^(-1/4) exp(-z²/2) |
| 1 | √2 π^(-1/4) z exp(-z²/2) |
| 2 | (1/√2) π^(-1/4) (2z² - 1) exp(-z²/2) |
| 3 | (1/√3) π^(-1/4) (2z³ - 3z) exp(-z²/2) |
| 4 | (1/(2√6)) π^(-1/4) (4z⁴ - 12z² + 3) exp(-z²/2) |
| 5 | (1/(2√15)) π^(-1/4) (4z⁵ - 20z³ + 15z) exp(-z²/2) |

## Least Squares Solution

```
c = (B^T B)^(-1) B^T Y

where B is the basis matrix:
B_ij = Ψⱼ(Zⁱ)
```

## Scaling Transformations

| Option | Transformation | Inverse |
|--------|---------------|---------|
| 0 | Z = X | X = Z |
| 1 | Z = (X - μ)/σ | X = σZ + μ |
| 2 | Z = T^(-1)(X - μ) | X = TZ + μ |
| 3 | Z = (log₁₀X - μ)/σ | X = 10^(σZ + μ) |
| 4 | Z = T^(-1)(log₁₀X - μ) | X = 10^(TZ + μ) |

Where T = V√Λ from eigendecomposition of Cov(X)

## Statistical Metrics

**R-Squared:**
```
R² = 1 - RSS/TSS = 1 - [Σ(Yᵢ - Ŷᵢ)²]/[Σ(Yᵢ - Ȳ)²]
```

**Adjusted R-Squared:**
```
R²_adj = [(m-1)R² - nTerm]/(m - nTerm)
```

**Coefficient of Variation:**
```
COV(cⱼ) = SE(cⱼ)/|cⱼ|

where SE(cⱼ) = √[RSS/(m-nTerm) × (B^T B)^(-1)ⱼⱼ]
```

**Model-Data Correlation:**
```
ρ = Cov(Y,Ŷ)/(σ_Y σ_Ŷ)
```

**Condition Number:**
```
κ(B) = σ_max(B)/σ_min(B)
```

## Algorithm Flow

```
1. Split data → training/testing
2. Scale data → Z = transform(X)
3. Generate terms → order matrix
4. LOOP (model reduction):
   a. Build basis → B = Ψ(Z)
   b. Fit model → c = (B^T B)^(-1) B^T Y
   c. Evaluate → R², ρ, COV
   d. Cull term → remove max(COV)
   e. Check stopping → max(COV) < tol?
5. Return final model
```

## Stopping Criteria

```
STOP when:
  max(COV) < tolerance  AND  ρ_test > 0
```

## Typical Parameters

| Parameter | Symbol | Typical Range | Default |
|-----------|--------|---------------|---------|
| Max Order | maxOrder | 2-5 | 3 |
| Train % | pTrain | 50-80% | 50% |
| Max Cull % | pCull | 20-50% | 30% |
| COV Tolerance | tol | 0.05-0.20 | 0.10 |
| Scaling | scaling | 0-4 | 1 |

## Complexity Estimates

| Operation | Complexity |
|-----------|-----------|
| Term generation | O(k^n) filtered |
| Basis construction | O(m·nTerm·n) |
| Least squares | O(m·nTerm² + nTerm³) |
| Total iterations | O(pCull·nTerm) |

Where:
- k = maxOrder
- n = number of inputs
- m = number of data points
- nTerm = number of terms

## Key Theoretical Results

**Orthonormality:**
```
∫₋∞^∞ ψₘ(z) ψₙ(z) dz = δₘₙ
```

**Recurrence Relation:**
```
H_{n+1}(z) = 2z H_n(z) - 2n H_{n-1}(z)
```

**Variance of Coefficients:**
```
Var(c) = σ² (B^T B)^(-1)
where σ² ≈ RSS/(m - nTerm)
```

## Interpretation Guide

**Good Model Indicators:**
- R²_adj > 0.8
- ρ_test > 0.9
- max(COV) < 0.10
- κ(B) < 100

**Warning Signs:**
- ρ_test < ρ_train (overfitting)
- Large condition number (κ > 1000)
- Many high COV coefficients
- Negative R² on test data

---

*Quick Reference Card for mimoSHORSA*
*October 23, 2025*
