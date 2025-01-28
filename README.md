# mimoSHORSA

**m**ulti-**i**nput **m**ulti-**o**tput **S**tochastic **H**igh **O**rder **R**esponse **S**urface **A**lgorithm

---------------------------------

## Usage

```
% [ order, coeff, trfrmX,trfrmY, meanX,meanY, testModelY, testX,testY ] = ...
%                mimoSHORSA( dataX,dataY, maxOrder, pTrain,pCull, tol, scaling )
%
% mimoSHORSA
% multi-input multi-output Stochastic High Order Response Surface Algorithm
% 
% This program fits a high order polynomial to multidimensional data via
% the stochastic high order response surface algorithm (mimoSHORSA) 
%
%  mimoSHORSA approximates the data with a polynomial of arbitrary order,
%    y(X) = a + \sum_{i=1}^n \sum_{j=1}^{k_i) b_ij X_i^j + 
%           \sum_{q=1}^m c_q \prod_{i=1}^n X_i^{p_iq}.
% The first stage of the algorithm determines the correct polynomial order,
% k_i in the response surace. Then the formulation of the mixed terms 
% \sum_{q=1}^m c_q \prod_{i=1}^n X_i^{p_iq} are derived by the second stage
% based on previous results. In the third stage, the response surface 
% is approximated.
%
% INPUT       DESCRIPTION                                                DEFAULT
% --------    --------------------------------------------------------   -------
% dataX       m observations of n input  features in a (nx x m) matrix
% dataY       m observations of m output features in a (ny x m) vector
% maxOrder    maximum allowable polynomial order                            3
% pTrain      percentage of data for training (remaining for testing)      50
% pCull       maximum percentage of model which may be culled              30 
% tol         desired maximum model coefficient of variation                0.10
% scaling     scale the data before fitting                                 1
%             scaling = 0 : no scaling
%             scaling = 1 : subtract mean and divide by std.dev
%             scaling = 2 : subtract mean and decorrelate
%             scaling = 3 : log-transform, subtract mean and divide by std.dev
%             scaling = 4 : log-transform, subtract mean and decorrelate
%
% OUTPUT      DESCRIPTION
% --------    --------------------------------------------------------
%  order      matrix of the orders of variables in each term in the polynomial 
%  coeff      polynomial coefficients 
%  meanX      mean vector of the scaled dataX
%  meanY      mean vector of the scaled dataY
%  trfrmX     transformation matrix from dataZx to dataX
%  trfrmY     transformation matrix from dataZy to dataY
%  testModelY output features for model testing
%  testX      input  features for model testing 
%  testY      output features for model testing 
%
% 
% get rainbow.m from ... http://www.duke.edu/~hpgavin/m-files/rainbow.m
```
