# References

The papers whose techniques or terms are implemented in this repository. Keys are cited from
`CONTEXT.md` and `docs/LEARNING_LOG.md`. Keys A/B/E match the research program's reading list
(`~/develop/research/READING.md`) so the two lists can be cross-read; L keys are botMaker-only.
A paper enters this list when code or a glossary term rests on it, not when it is merely related.

| Key | Paper | What of it is here | Link |
|---|---|---|---|
| A1 | Arai, Maung, Schweitzer. *Optimal Column Subset Selection by A-Star Search.* AAAI 2015 | The search, the spectral lower bound, the rank-one downdate (§4) | https://personal.utdallas.edu/~haim/publications/astar.pdf |
| A2 | Arai, Maung, Xu, Schweitzer. *Unsupervised Feature Selection by Heuristic Search with Provable Bounds on Suboptimality.* AAAI 2016 | The bounded-certificate vocabulary (weighted variant not yet built) | https://doi.org/10.1609/aaai.v30i1.10082 |
| A3 | Wan, Schweitzer. *Heuristic Search for Approximating One Matrix in Terms of Another Matrix.* IJCAI 2021 | The corrected form of the downdate (residual, not projection) | https://www.ijcai.org/proceedings/2021/0221.pdf |
| A4 | Deshpande, Rademacher. *Efficient Volume Sampling for Row/Column Subset Selection.* FOCS 2010 | The (k+1) bracket between the rank-k floor and the best subset | https://arxiv.org/abs/1004.4057 |
| B1 | Chen, Epperly, Tropp, Webber. *Randomly Pivoted Cholesky.* CPAM 2025 | The RPCholesky selector; the selection gap it never measures | https://arxiv.org/abs/2207.06503 |
| B3 | Dereziński, Khanna, Mahoney. *Improved Guarantees and a Multiple-Descent Curve for CSS and the Nyström Method.* IJCAI 2021 | The subset-to-floor ratio the harness reports | https://arxiv.org/abs/2002.09073 |
| L1 | Farahat, Ghodsi, Kamel. *A novel greedy algorithm for Nyström approximation.* AISTATS 2011 | The greedy trace selector and the gain identity behind batched goal-depth bounds | https://proceedings.mlr.press/v15/farahat11a.html |
| L2 | Williams, Seeger. *Using the Nyström method to speed up kernel machines.* NeurIPS 2000 | The Nyström approximation and the word landmark | https://papers.nips.cc/paper/1866-using-the-nystrom-method-to-speed-up-kernel-machines |
| L3 | Hart, Nilsson, Raphael. *A formal basis for the heuristic determination of minimum cost paths.* IEEE TSSC 1968 | A* and admissibility | https://doi.org/10.1109/TSSC.1968.300136 |
| L4 | Bunch, Nielsen, Sorensen. *Rank-one modification of the symmetric eigenproblem.* Numer. Math. 1978 | The secular equation and interlacing used by the downdate | https://doi.org/10.1007/BF01396012 |
| L5 | He, Gu, Zhang. *Nodal domain partition and the number of communities in networks.* arXiv 2012 | The `roach_g20` instance: the eighty-node cockroach, its three-part ideal cut, and K = 3 | https://arxiv.org/abs/1201.5767 |
| L6 | Krebs, V. *Political books.* Unpublished; distributed in Newman's network data collection | The `polbooks` rung-3 instance (n = 105, m = 441, K = 3); the hand-assigned liberal / neutral / conservative alignments are the `value` node attribute. Retrieved 2026-09-10 | https://websites.umich.edu/~mejn/netdata/polbooks.zip |
| L7 | Girvan, Newman. *Community structure in social and biological networks.* PNAS 99(12):7821, 2002 | The `football` rung-3 instance (n = 115, m = 613, K = 12); the `value` node attribute is the athletic conference. Retrieved 2026-09-10 | https://doi.org/10.1073/pnas.122653799 |
| L8 | Leskovec, Kleinberg, Faloutsos. *Graph evolution: densification and shrinking diameters.* ACM TKDD 1(1), 2007; department labels from Yin, Benson, Leskovec, Gleich. *Local higher-order graph clustering.* KDD 2017 | The `email_eu_core` rung-3 instance via SNAP: 1005 nodes over 20 components, cut down to the 986-node largest component, labelled by the sender's department (K = 42). Retrieved 2026-09-10 | https://snap.stanford.edu/data/email-Eu-core.html |

Links for A/B keys were verified by the research program on 2026-09-04; L1–L4 were added
2026-09-04 from memory of the standard citations and should be checked on first use. L5 was
verified on 2026-09-10 against the arXiv e-print itself — the paper's LaTeX source and its Fig. 1,
2a and 2b — and not from memory.

## Log
- 2026-09-04 — created with the plan for BL-27; A/B keys mirror the research reading list.
- 2026-09-10 — L5 added with slice 2.6 of the two-hot span plan: `roach_g20_instance` is that
  paper's Fig. 1-2 cockroach, and its planted K = 3 labels are the "ideal cut" of its Fig. 2a. Its
  reported λ2 = 0.0057, λ3 = 0.0062, λ4 = 0.0246 are reproduced to four decimals by
  `test_the_roach_g20_laplacian_reproduces_the_papers_three_smallest_eigenvalues`.
- 2026-09-10 — L6, L7 and L8 added with the rung-3 loaders (`mllib.math.graph.community_graphs`).
  These are datasets, not techniques: each key names the paper the published communities come from,
  and the file itself is fetched with a pinned sha256 into a directory outside git, never committed.
  L6 has no paper — Krebs never published the network — so the collection that distributes it is
  cited instead. The football labels carry a known error the loader does not correct: five
  independents are mislabelled in the original file.
