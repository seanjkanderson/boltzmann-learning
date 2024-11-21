# Boltzmann Learning

This module implements a low-regret learning algorithm for an agent that seeks to minize a cost by making a sequence of 
decisions in response to a stream of measurements.

## Problem Formulation

The decision made by the agent at time `t` consists of selecting an action `a(t)` among a set `action_set` to minimize 
a cost `J(a(t),w(t))` that depend on the action `a(t)` as well as on a latent variable `w(t)` that is unknown to the agent.

To decide on the value of the action `a(t)`, the agent has available a stream of measurements `y(t)`, each taking values
in a set `measurement_set`.

The relationships between actions, measurements, and latent variables are unknown and need to be estimated from data. 
To this effect, we assume that for some time instants `t`, we learn “after-the-fact” the value of the cost `J(a,w(t))` 
associated with all possible actions in `action_set`.

We are especially interested in scenarios where the value of the cost `J(a(t),w(t))` may be determined by other agents, 
whose interests are unknown to us and that may be reacting to past actions. In this context, the latent variable `w(t)` 
will include the internal states of such agents, which could be correlated across time and also correlated to past actions.

## Contents

+ `src/`:  python module
+ `test/`: unit testing scripts
+ `doc/`: documentation
+ `examples/`: examples

## Installation

[//]: # (1&#41; Make sure that `src/` is in python's import path, e.g., with)

[//]: # ()
[//]: # (    ```python)

[//]: # (    sys.path.append&#40;"src"&#41;)

[//]: # (    ```)

[//]: # ()
[//]: # (2&#41; Import `LearningGames.py` using)

[//]: # ()
[//]: # (    ```python )

[//]: # (    import LearningGames)

[//]: # (    ```)

1) Switch to the LearningGames directory such that setup.py is in the current directory.
2) Either in a virtual environment (e.g. conda or pipenv) (preferred) or using a base python distribution.  In order to 
3) examples including rock-paper-scissors, run:
   ```bash
   pip install -e .
    ```
   for editable mode (i.e. development). For a normal install run
   ```bash
   pip install .
   ```
4) [Optional] In order to run the EMBER example (`ember_malware_classification.py`), it's necessary to download the dataset first:
   1. Go to the [EMBER Github repo](https://github.com/elastic/ember)
      - Download the 2018 dataset by clicking the associated [link](https://ember.elastic.co/ember_dataset_2018_2.tar.bz2).
      - It is necessary to install their package either by using Docker or installing per their README 
   2. For usability, it is easier to move the data to pickled objects.
   Assuming the 2018 data you downloaded is in `data/ember2018/`,
   and the target location to store the pickled data: `this_project/data/`
   ```
   import ember
   import pickle
   # load the data using ember (sits on top of Lief)
   X_train, y_train, X_test, y_test = ember.read_vectorized_features("/data/ember2018/")
   # write to a dict for easy iteration
   data = dict(x_train=X_train, y_train=y_train,
            x_test=X_test, y_test=y_test)
   for key, val in data.items():
      # store data in pkl files individually (easier to chunk out as files are large for X_)
       with open(f'this_project/data/{key}.pkl', 'wb') as f:
          pickle.dump(val, file=f)
   ```
5) [Optional] Run the unit testing scripts from the shell using

 ```
 python test/testLearningGames.py
 ```

## Examples

The following examples can be found in the `example/` folder:

1. `rps_vs_bad_rng.ipynb`: a minimal example. learn optimal policy for the Rock-Paper-Scissors game, against an 
opponent that uses a randomized policy with a bad generator of random numbers

2. `rps_vs_bad_rng_nonstationary.py`: considers a setting similar to the former, except that the opponent switches random 
    number generators partway through the game. Compares against relevant alternative strategies, which can take several
    minutes to run. As such, results are persisted so that visualization can be done separately.
    - `rps_plot.py`: Visualize results by running with appropriate file path of results.

3. `ember_malware_classification.py`: run the EMBER malware classification example with goal of minimizing weighted sum
of classification error (false positives and false negatives). Compares against relevant benchmarks that are generally
slow, so it persists results to pkl files. Note the extra installation required to get the dataset.
   - `ember_plot.py`: Visualize results by running with file path of results 


## Contact Information

Joao Hespanha (hespanha@ece.ucsb.edu)

http://www.ece.ucsb.edu/~hespanha

Sean Anderson (seananderson@ucsb.edu)

University of California, Santa Barbara

## License Information

Copyright (C) 2023 Joao Hespanha, Univ. of California, Santa Barbara

