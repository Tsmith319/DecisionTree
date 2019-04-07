#!/bin/bash

clear
echo "Running Stochastic Sub Gradient Algorithm With First Gamma_t Equation"
echo " "
echo "Results:"
python stochastic_sgd_primal_a.py
echo " "
echo "Running Stochastic Sub Gradient Algorithm With First Gamma_t Equation"
echo " "
echo "Results:"
python stochastic_sgd_primal_b.py
echo " "
echo "Running Dual Form Optimization without kernel"
echo " "
echo "Results:"
python dual_SVM_a.py
echo "Running Dual Form Optimization with kernel"
echo " "
echo "Results:"
python dual_SVM_kernel_b.py