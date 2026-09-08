# UCI datasets for the Nyström landmark-selection harness

Three small UCI files, committed because each is well under the 1 MB per-file ceiling (D-19). They
are the datasets the AAAI-15 column-subset-selection paper reports on, which is why these three and
not others: the same instances make the harness's numbers comparable to the reproduction in
`research/repro/astar-css`.

Source: UCI Machine Learning Repository, <https://archive.ics.uci.edu/>. The repository licenses its
datasets under Creative Commons Attribution 4.0 (CC BY 4.0); cite the original donors when results
are published. Files here are byte-identical to the UCI originals apart from line endings, which are
normalized to LF on commit.

| File | Rows | Columns | Label | Feature columns the harness reads |
|---|---|---|---|---|
| `SPECTF.test` | 187 | 45 | first column, 0/1 diagnosis | columns 1 onward (44 features) |
| `movement_libras.data` | 360 | 91 | last column, class 1–15 | columns 0 to 89 (90 features) |
| `wdbc.data` | 569 | 32 | second column, `M`/`B` | columns 2 onward (30 features) |

`wdbc.data` needs its own loader rather than a label position: its first column is a patient id and
its second is a non-numeric diagnosis, so both are dropped before the numeric parse. Getting this
wrong is not loud — an id column parses as a feature and quietly dominates the kernel, since it is
orders of magnitude larger than every real measurement.

The harness standardizes columns and subsamples rows before building a kernel, so nothing here is
used at full size; see `src/mllib/ml/projects/nystrom_uci_data.py`.

| Dataset | Original donors |
|---|---|
| SPECTF Heart | Krzysztof J. Cios, Lukasz A. Kurgan, Lucy S. Goodenday |
| Libras Movement | Daniel B. Dias, Sarajane M. Peres, Helton H. Bíscaro |
| Breast Cancer Wisconsin (Diagnostic) | William H. Wolberg, W. Nick Street, Olvi L. Mangasarian |
