# mimoSHORSA

**mimoSHORSA has moved to http://github.com/hpgavin/multivarious** 

**m**ulti-**i**nput **m**ulti-**o**tput **S**tochastic **H**igh **O**rder **R**esponse **S**urface **A**lgorithm

A sophisticated polynomial regression method using Hermite function basis with adaptive model reduction for high-dimensional data.

---------------------------------

## Usage

```
    order, coeff, trfrmX,trfrmY, meanX,meanY, testX,testY, testModelY = \
                 mimoSHORSA( dataX,dataY, maxOrder, pTrain,pCull, tol, scaling )
 
  mimoSHORSA
  multi-input multi-output Stochastic High Order Response Surface Algorithm
  
  This program fits a high order polynomial to multidimensional data via
  the stochastic high order response surface algorithm (mimoSHORSA) 
 
   mimoSHORSA approximates the data with a polynomial of arbitrary order,
     y(X) = a + \sum_{i=1}^n \sum_{j=1}^{k_i) b_ij X_i^j + 
            \sum_{q=1}^m c_q \prod_{i=1}^n X_i^{p_iq}.
  The first stage of the algorithm determines the correct polynomial order,
  k_i in the response surace. Then the formulation of the mixed terms 
  \sum_{q=1}^m c_q \prod_{i=1}^n X_i^{p_iq} are derived by the second stage
  based on previous results. In the third stage, the response surface 
  is approximated.
 
  INPUT       DESCRIPTION                                                DEFAULT
  --------    --------------------------------------------------------   -------
  dataX       m observations of n input  features in a (nx x m) matrix
  dataY       m observations of m output features in a (ny x m) vector
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
 
  OUTPUT      DESCRIPTION
  --------    --------------------------------------------------------
   order      matrix of the orders of variables in each term in the polynomial 
   coeff      polynomial coefficients 
   meanX      mean vector of the scaled dataX
   meanY      mean vector of the scaled dataY
   trfrmX     transformation matrix from dataZx to dataX
   trfrmY     transformation matrix from dataZy to dataY
   testX      input  features for model testing 
   testY      output features for model testing 
   testModelY output features for model testing
 
```

---

## Package Contents

### Core Implementation
- **`mimoSHORSA.py`** - Complete Python implementation (~1000 lines)
  - Main algorithm with 17 functions
  - Hermite polynomial basis
  - Legendre polynomial basis
  - Power polynomial basis
  - Model reduction via L1 regularization
  - Iterative model reduction via "culling"
  - Multiple data scaling options

### Test & Examples
- **`mimoSHORSA_test.py`** - Comprehensive test suite
  - Synthetic data generation
  - Measured data loading
  - Visualization utilities
  
- **`simple_example.py`** - Quick start example
  - Minimal dependencies
  - Clear visualization
  - Easy to understand

### Documentation
- **`mimoSHORSA_THEORY.md`** - Complete theoretical foundation
  - Mathematical derivations
  - Algorithm details
  - Application guidance
  
---

## Reference

### Primary Publication
Gavin, H.P. and Yau, S.C., "High order limit state functions in the 
response surface method for structural reliability analysis,"
*Structural Safety*, Vol. 30, pp. 162-179, 2008.

### Authors
- Siu Chung Yau
- Henri P. Gavin

Department of Civil and Environmental Engineering  
Duke University  
2006, 2023

---

**Version**: 1.0  
**Date**: October 23, 2025  
**Status**: Production Ready

*"Is this the fundamental problem statement of all data-science?"*
