# Inferring Causal Graph Temporal Logic Formulas to Expedite Reinforcement Learning in Temporally Extended Tasks

Code for the paper by Hadi Partovi Aria, Zhe Xu. AAAI 2026 Bridge Program, 2026. https://arxiv.org/abs/2601.02666

GTL-CIRL couples Q-learning with Bayesian optimization to infer causal graph temporal logic (GTL) formulas. The inferred formulas shape the reward, and counterexample trajectories collected during learning refine the formula search.

## Files

`gtl_cirl_gene_regulation.py` has the gene regulation network environment, the GTL and causal GTL formulas, the GTL-CIRL learner and the baselines. Running it trains all methods and plots the success rates.

## Running

```bash
pip install -r requirements.txt
python gtl_cirl_gene_regulation.py
```

scikit-learn is optional; without it the formula search falls back to random search.

## Citation

```bibtex
  @misc{aria2026gtl,
    title = {Inferring Causal Graph Temporal Logic Formulas to Expedite Reinforcement Learning in Temporally Extended Tasks},
    author = {Hadi Partovi Aria and Zhe Xu},
    howpublished = {AAAI 2026 Bridge Program},
eprint = {2601.02666},
archivePrefix = {arXiv},
    year = {2026}
  }
```

## License

MIT, see `LICENSE`.
