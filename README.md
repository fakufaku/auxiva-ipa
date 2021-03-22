AuxIVA with Iterative Projection with Adjustment Updates
========================================================

This repository contains the code for the paper "Independent Vector Analysis
via Log-Quadratically Penalized Quadratic Minimization" by [Robin
Scheibler](http://robinscheibler.org).

Abstract
--------

We propose a new algorithm for blind source separation (BSS) using independent
vector analysis (IVA). This is an improvement over the popular auxiliary
function based IVA (AuxIVA) with iterative projection (IP) or iterative source
steering (ISS). We introduce iterative projection with adjustment (IPA), where
we update one demixing filter and jointly adjust all the other sources along
its current direction. Each update involves solving a non-convex minimization
problem that we term log-quadratically penalized quadratic minimization
(LQPQM), that we think is of interest beyond this work. In the general case, we
show that its global minimum corresponds to the largest root of a univariate
function, reminiscent of modified eigenvalue problems. We propose a simple
procedure based on Newton-Raphson to efficiently compute it. Numerical
experiments demonstrate the effectiveness of the proposed method. First, we
show that it efficiently decreases the value of the surrogate function. In
further experiments on synthetic mixtures, we study the probability of finding
the true demixing matrix and convergence speed. We show that the proposed
method combines high success rate and fast convergence. Finally, we validate
the performance on a reverberant blind speech separation task. We find that all
the AuxIVA-based methods perform similarly in terms of acoustic BSS metrics.
However, AuxIVA-IPA converges faster. We measure up to 8.5 times speed-up in
terms of runtime compared to the next best AuxIVA-based method, depending on
the number of channels and the signal-to-noise ratio (SNR).

Author
------

* [Robin Scheibler](http://robinscheibler.org)


Summary of Experiments
----------------------

### Solve Random SeDJoCo Problems

Evaluate the performance of the proposed method to minimize the surrogate function only.

### Separation of Synthetic Mixtures

Evaluate many IVA algorithms for the separation of data following the spherical Laplace distribution.

### Separation of Reverberant Speech Mixtures

Evaluate the performance of many IVA algorithms for the separation of reverberant speech.

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

The script `prepare_figures.sh` was used to prepare the figures of the paper.
If the simulations are re-run, it should be edited to adjust the result file names.

### Random SeDJoCo Problems

        # Run the simulation, creates <result_file> in `data/` folder
        python ./experiment_head.py
        
        # Create the figure
        python ./experiment_head_plot.py data/<result_file>

### Separation of Synthetic Mixtures

        # Run the simulation, creates <result_file> in `data/` folder
        python ./experiment_iva.py
        
        # Create the figure
        python ./experiment_iva_plot.py data/<result_file>

### Separation of Reverberant Speech Mixtures

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


The figures and tables can be produced as follows.

    # figure with box-plots of SDR/SIR
    python ./make_figure3_journal_ipa_separation_hist.py \
      ./data/<result_folder>/ --pca

    # figure with wall-clock vs SDR/SIR
    python ./make_figure4_journal_ip_speed_contest.py \
      ./data/<result_folder>/ --pca --pickle

    # runtime table
    python ./make_table_runtime.py data/<result_folder>/ --pickle

License
-------

The code is provided under MIT license.
