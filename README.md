# Knowledge Representation — Probabilistic Graphical Models

Python implementations of three algorithms for discrete Bayesian networks, written for the Knowledge Representation course (Computer Engineering, Universidad de Cantabria). Code and comments are in Spanish.

## `alg_eliminacion_variables/` — Variable elimination

- `elim_var.py`: `Variable` and `Factor` classes, factor product, marginalisation, evidence, and variable elimination for marginal and conditional queries. Running it checks results on the chain A → B → C against values worked out by hand. Standard library only.
- `elim_var_sencillo.py`: a simpler dictionary-based version that computes P(B) on a five-variable example.
- `eficacia.py`: correctness tests. It covers hand-computed cases, the alarm network (Russell & Norvig), a course example with every elimination order, and random binary and multi-valued networks with and without evidence, all compared against brute force.
- `coste_temporal.py`: empirical time cost as a function of the number of variables *n*, the cardinality *d* and the largest intermediate factor *w* (good vs bad elimination order in a star network). Writes `exp1_n.csv`, `exp2_d.csv`, `exp3_w.csv` and, if matplotlib is installed, `coste_temporal.png`.

## `alg_separacion/` — Graphical independence test (d-separation)

`sep_var.py` checks whether X ⊥ Y | Z in a DAG. It removes leaf nodes outside X ∪ Y ∪ Z, marries parents that share a child, drops edge directions, deletes Z and looks for a path between X and Y. Uses NetworkX.

## `alg_chowliu/` — Chow-Liu algorithm

`p3.py` learns a tree-shaped Bayesian network from data. It estimates pairwise mutual information, builds the maximum-weight spanning tree and orients it from a root. It then estimates the CPTs by maximum likelihood and computes the log-likelihood. It includes six tests (among them recovering a known tree from synthetic data and a brute-force check) and a timing experiment that saves `coste_temporal_chow_liu.png`. Uses NumPy and NetworkX.

## Running

Python 3.10+.

```bash
pip install numpy networkx matplotlib   # matplotlib is optional

cd alg_eliminacion_variables
python elim_var.py         # hand-checked examples
python eficacia.py         # correctness tests
python coste_temporal.py   # timing experiments (CSV + plot)

cd ../alg_separacion
python sep_var.py

cd ../alg_chowliu
python p3.py               # tests + timing experiment
python p3.py --sin-tiempos # tests only
```
