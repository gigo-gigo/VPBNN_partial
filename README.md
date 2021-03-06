[This](https://github.com/gigo-gigo/VPBNN_partial) is the code used for the experiments in the paper ["Bayesian Neural Networks with Variance Propagation for Uncertainty Evaluation"](https://openreview.net/forum?id=30SS5VjvhrZ). This provides logs of the experiments.

The experiments are conducted using TensorFlow 2.1. Please see requirements.txt to know packages in the experiments.

## Code

- approximation_*.ipynb:
   - provides results of appendix B.
   - is executable.

- nn2vpbnn_*.ipynb:
   - provides results of 5.1 and appendix C.
   - is partially executable because we have not released some of our code such as the vpbnn package.

- ptb_*.ipynb:
   - provides results of 5.2 and appendix D.
   - is partially executable because we have not released some of our code such as the vpbnn package.

- ood.ipynb:
   - provides results of 5.3 and appendix E.
   - is partially executable because we have not released some of our code such as the vpbnn package.
