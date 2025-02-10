The matlab code of S-PADMM is adapted from the source: http://faculty.bicmr.pku.edu.cn/~wenzw/optbook/pages/
The dataset can be found in the link: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/

Folders: 
1. 'Binary Class' contains the codes for binary classification task.
2. 'Mutli Class' contains the codes for mutliclass classification task.
Please note that these two folders should not be added into the path at the same time, as they include the functions with the same name.

Main matlab files:
1. '.\Binary Class\algs' contains the codes of optimization algorithms for binary classification.
    '.\Mutli Class\algs' contains the codes of optimization algorithms for mutli classification.
2. 'run_diff_alg.m' is the main file to perform classification tasks using different algorithms
    'run_diff_parm.m' is the main file to perform classification tasks using S-PADMM with different parameters.