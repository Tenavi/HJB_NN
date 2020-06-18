## Adaptive deep learning for high-dimensional Hamilton-Jacobi-Bellman equations

See the paper at [https://arxiv.org/abs/1907.05317](https://arxiv.org/abs/1907.05317v4).

#### Software recommendations:

tensorflow-gpu version 1.11, scipy version 1.4, numpy version 1.16

How this repository is organized:

### Main folder:

  * generate.py: generate data by solving BVPs using time-marching

  * train.py: train NNs to model the value function

  * simulate.py: simulate the closed-loop dynamics of a system and compare with BVP solution.

  * simulate_noise.py: simulate the closed-loop dynamics with a zero-order-hold and measurement noise.

  * predict_value.py: use a NN to predict the value function on a grid.

  * test_time_march.py, test_warm_start.py: test the reliability and speed of time-marching and NN warm start

### examples/:

#### *This is the only folder with settings that need to be adjusted*

This folder contains examples of problems each in their own folder. Each of these folders must contain a file called **problem_def.py** which defines the dynamics, optimal control, and various other settings. Data, NN models, and simulation results are all found here. The examples/ folder also contains

  * choose_problem.py: modify this script to tell other scripts which problem to solve_bvp

  * problem_def_template.py: a basic scaffold for how to define problems which these scripts can use

### utilities/:

  * neural_networks.py: auxiliary file which contains classes implementing NNs for predicting initial-time and time-dependent value functions.

  * optimize.py: auxiliary file which contains interfaces to use Scipy optimizers with tensorflow, with some modifications

  * other.py: other commonly-used utility functions
