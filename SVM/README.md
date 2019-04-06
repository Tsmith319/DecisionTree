Welcome to the SVM file of my Machine Learning library. Within this file you will find four files. The files that have the words 
stochastic_sgd_primal_?.py where the ? can be a or b. This will run the stochastic sub gradient algorithm in the primal form. It will
output the weight vector for each C value which are [1/873, 10/873, 50/873, 100/873, 300/873, 500/873, 700/873]. 
The next two files pertain to SVM in the dual domain. The first is dual_SVM_a.py which will run the optimization on the dual form SVM and 
will output the weight vector and bias point for the following C values [100/873, 500/873, 700/873]. The last file is dual_SVM_kernel_b.py
which is meant to perform the kernel optimization. Unfortunately this file is currently not working because it will not converge with the
current implementation of the kernel. I am continuing trying to find a solution to this issue.
To run these files you will simply type python run.sh which will run a python script that will only run the first three files mentioned in
this README and is unable to run the last file because it doesn't converge so no output is given. 
