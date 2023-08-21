# distributed-ensemble-learning-with-permuted-images
Federated learning with permutation obfuscation done as a part of master's thesis.

## Contribution
The project aims to provide means of private neural network training. The idea consist in dividing image into pieces and permuting them with given scheme. 
Each section corresponds to a neural network instance, and the models are trained separately. Subsequently, features precomputed by the ensemble are aggreagted to form the final prediction.
The 5x2-fold cross validation and Student's t-test confirm that the model composed of 9 submodels with overlapping image samples and blockwise encryption scheme is able to replicate the same accuracy as single model with no encryption.
