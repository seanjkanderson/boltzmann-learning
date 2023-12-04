# LearningGames module 

This module implements a low-regret learning algorithm for an agent that seeks to minize a cost by making a sequence of decisions in response to a stream of measurements.

## Problem Formulation

The decision made by the agent at time `t` consists of selecting an action `a(t)` among a set `action_set` to minimize a cost `J(a(t),w(t))` that depend on the action `a(t)` as well as on a latent variable `w(t)` that is unknown to the agent.

To decide on the value of the action `a(t)`, the agent has available a stream of measurements `y(t)`, each taking values in a set `measurement_set`.

The relationships between actions, measurements, and latent variables are unknown and need to be estimated from data. To this effect, we assume that for some time instants `t`, we learn “after-the-fact” the value of the cost `J(a,w(t))` associated with all possible actions in `action_set`.

We are especially interested in scenarios where the value of the cost `J(a(t),w(t))` may be determined by other agents, whose interests are unknown to us and that may be reacting to past actions. In this context, the latent variable `w(t)` will include the internal states of such agents, which could be correlated across time and also correlated to past actions

## Contents

+ `src/`:  python module
+ `test/`: unit testing scripts
+ `doc/`: documentation
+ `examples/`: examples

## Installation

1) Make sure that `src/` is in python's import path, e.g., with

    ```python
    sys.path.append("src")
    ```

2) Import `LearningGames.py` using

    ```python 
    import LearningGames
    ```

3) [Optional] Run the unit testing scripts from the shell using

    ```shell
    python test/testLearningGames.py
    ```

## Examples

The following examples can be found in the `example/` folder:

1. `RPS_vs_fixed.ipynb`: learn optimal policy for the `Rock-Paper-Scissors`   game, against an opponent that uses a fixed policy

2. `RPS_vs_round_robin.ipynb`: learn optimal policy for the `Rock-Paper-Scissors` game, against an opponent that cycles between actions in a deterministic fashion

3. `RPS_vs_bad_rng.ipynb`: learn optimal policy for the `Rock-Paper-Scissors`   game, against an opponent that uses a randomized policy with a bad generator of random numbers

- `RPS_self_play.ipynb`: learn optimal policy for the `Rock-Paper-Scissors`   game, against an opponent that uses the same learning algorithm

- `PD_vs_Nash.ipynb`: learn optimal policy for the `Prisoner's Dilemma` game, against an opponent that uses a Nash policy

- `PD_self_play.ipynb`: learn optimal policy for the `Prisoner's Dilemma` game, against an opponent that uses the same learning algorithm

- `CCF_vs_fixed.ipynb`: learn optimal policy for a `Credit Card Fraude` game, against an opponent that uses a fixed policy

- `CCF_self_play.ipynb`: learn optimal policy for a `Credit Card Fraude` game, against an opponent that uses the same learning algorithm

- `NS_vs_fixed.ipynb`: learn optimal policy for a `Network Security` game, against an opponent that uses a fixed policy

**ATTENTION: items with `circle` instead of number have not yet been done**

## Contact Information

Joao Hespanha (hespanha@ece.ucsb.edu)

http://www.ece.ucsb.edu/~hespanha

University of California, Santa Barbara

## License Information

Copyright (C) 2023 Joao Hespanha, Univ. of California, Santa Barbara

