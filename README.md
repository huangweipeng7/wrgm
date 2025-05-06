# Intro 
This is the implementation of the Blocked-Collapsed Gibbs Sampling for **Bayesian Wasserstein Repulsive Gaussian Mixture Models**. 

Paper link: https://arxiv.org/pdf/2504.21391
```
@article{huang2025bayesian,
  title={Bayesian Wasserstein Repulsive Gaussian Mixture Models},
  author={Huang, Weipeng and Ng, Tin Lok James},
  journal={arXiv preprint arXiv:2504.21391},
  year={2025}
}
```

## Set up the package
Open a Julia REPL and enter the package mode (type in ```]``` and then ```\tab```).
Then we should be able to type in ```dev .``` and press the enter key. 
The REPL windown may look like:
```
(@JuliaVersion) pkg> dev . 
```

## Run the code
The command line examples for launching an experiment are placed in the shell files within the **script** folder.

A quick launch could be 
```
sh scripts/run_sim1.sh
```
