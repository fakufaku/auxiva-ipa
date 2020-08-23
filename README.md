AuxIVA with Iterative Projection with Adjustment Updates
========================================================

This repository contains the code for the paper "Independent Vector Analysis
via Log-Quadratically Penalized Quadratic Minimization" by [Robin
Scheibler](http://robinscheibler.org).

Abstract
--------

We propose a new algorithm for blind source separation of convolutive
mixtures using independent vector analysis. This is an improvement over the
popular auxiliary function based independent vector analysis (AuxIVA) with
iterative projection (IP) or iterative source steering (ISS). We introduce
iterative projection with adjustment (IPA), whereas we update one demixing
filter and jointly adjust all the other sources along its current direction. We
implement this scheme as multiplicative updates by a rank-2 perturbation of the
identity matrix. Each update involves solving a non-convex minimization problem
that we term log-quadratically penalized quadratic minimization (LQPQM), that
we think is of interest beyond this work. We find that the global minimum of an
LQPQM can be efficiently computed. In the general case, we show that all its
stationary points can be characterized as zeros of a kind of secular equation,
reminiscent of modified eigenvalue problems. We further prove that the global
minimum corresponds to the largest of these zeros. We propose a simple
procedure based on Newton-Raphson seeded with a good initial point to
efficiently compute it. We validate the performance of the proposed method for
blind acoustic source separation via numerical experiments with reverberant
speech mixtures. We show that not only is the convergence speed faster in terms
of iterations, but each update is also computationally cheaper. Notably, for
four and five sources, AuxIVA with IPA converges more than twice as fast as
competing methods.

Author
------

* [Robin Scheibler](http://robinscheibler.org)


Summary of Experiments
----------------------

### Experiment 1: Separation performance

Separation performance for different numbers of sources and microphones.

### Experiment 2: Speed contest

Plot iterations/runtime of algorithm vs SIR.

Test Run the Algorithms
-----------------------

The `example.py` program allows to test the different algorithms on simulated scenarios.
For example, try running

    python ./example.py -a auxiva-ipa -m 4

to separate four sources with four microphones using the algorithm `overiva-ip2`.
The full usage instructions is provided below.

    > python ./example.py --help
    usage: example.py [-h] [--no_cb]
                      [-a {auxiva,auxiva2,auxiva-iss,overiva,overiva-ip,overiva-ip2,overiva-ip-block,overiva-ip2-block,overiva-demix-bg,five,ogive,ogive-mix,ogive-demix,ogive-switch,auxiva-pca,auxiva-ipa,pca}]
                      [-d {laplace,gauss}] [-i {pca}] [-m MICS] [-s SRCS]
                      [-z INTERF] [--sinr SINR] [-n N_ITER] [--gui] [--save]
                      [--seed SEED]

    Demonstration of blind source extraction using FIVE.

    optional arguments:
      -h, --help            show this help message and exit
      --no_cb               Removes callback function
      -a {auxiva,auxiva2,auxiva-iss,overiva,overiva-ip,overiva-ip2,overiva-ip-blok,overiva-ip2-block,overiva-demix-bg,five,ogive,ogive-mix,ogive-demix,ogive-switch,auxiva-pca,auxiva-ipa,pca}, --algo {auxiva,auxiva2,auxiva-iss,overiva,overiva-ip,overiva-ip2,overiva-ip-block,overiva-ip2-block,overiva-demix-bg,five,ogive,ogive-mix,ogive-demix,ogive-switch,auxiva-pca,auxiva-ipa,pca}
                            Chooses BSS method to run
      -d {laplace,gauss}, --dist {laplace,gauss}
                            IVA model distribution
      -i {pca}, --init {pca}
                            Initialization, eye: identity, eig: principal
                            eigenvectors
      -m MICS, --mics MICS  Number of mics
      -s SRCS, --srcs SRCS  Number of sources
      -z INTERF, --interf INTERF
                            Number of interferers
      --sinr SINR           Signal-to-interference-and-noise ratio
      -n N_ITER, --n_iter N_ITER
                            Number of iterations
      --gui                 Creates a small GUI for easy playback of the sound
                            samples
      --save                Saves the output of the separation to wav files
      --seed SEED           Random number generator seedc

Reproduce the Results
---------------------

The code can be run serially, or using multiple parallel workers via
[ipyparallel](https://ipyparallel.readthedocs.io/en/latest/).
Moreover, it is possible to only run a few loops to test whether the
code is running or not.

1. Run **test** loops **serially**

        python ./paper_simulation.py ./experiment_ipa_config.json -t -s

2. Run **test** loops in **parallel**

        # start workers in the background
        # N is the number of parallel process, often "# threads - 1"
        ipcluster start --daemonize -n N

        # run the simulation
        python ./paper_simulation.py ./experiment_ipa_config.json -t

        # stop the workers
        ipcluster stop

3. Run the **whole** simulation

        # start workers in the background
        # N is the number of parallel process, often "# threads - 1"
        ipcluster start --daemonize -n N

        # run the simulation
        python ./paper_simulation.py ./experiment_ipa_config.json

        # stop the workers
        ipcluster stop

The results are saved in a new folder `data/<data>-<time>_speed_contest_journal_ipa_<flag_or_hash>`
containing the following files

    parameters.json  # the list of global parameters of the simulation
    arguments.json  # the list of all combinations of arguments simulated
    data.json  # the results of the simulation

Figure 1., 2., 3., and 4. from the paper are produced then by running

    ./prepare_figures.sh

License
-------

The code is provided under MIT license.
