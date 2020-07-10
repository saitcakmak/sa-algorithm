This is the code used for running the numerical experiments presented
in the paper "Solving Bayesian Risk Optimization via Nested Stochastic Gradient Estimation"
by Cakmak, Wu and Zhou.

The code is a bit old and is not well documented.

Here is a brief overview of how things work:

normal_runner: the script for simple quadratic example 

problem_sampler: has scripts for sampling from h() and h'() for the available problems.
Pick the curresponding function and call it with given theta, x and m. It will return
m samples of h() and h'().

naive_estimator: The estimator as defined in the paper. Not sure what the variance 
estimator there is supposed to be. It is likely as the name suggest estimates
the variance. 
Call this with a list of theta samples, the problem name and problem_sampler parameters,
along with alpha and rho, and get the estimates of the BRO objective and the 
corresponding derivative.

simple_runner: code for running the simple problem - does not do optimization, just calculates the estimators

two_sided_runner: The main code for running a replication of the two sided problem. 

normal_runner: in works, for running the simple normal example

output_reader: To read the SA outputs

output_analyzer: To read the SA outputs

mcmc: generates the MCMC sample paths for estimating P^N in the two_sided problem

multi_runner: runs the experiments in parallel.

rho_output_analyzer: makes the grid plots comparing different objectives

rho_output_saver: probably does something to produce the input for the analyzer.

value_calculator / plotter: tools for calculating / plotting the BRO objective values.