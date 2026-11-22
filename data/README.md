# Elliptic dataset

This project targets the original Bitcoin transaction graph published by Elliptic in 2019. A node represents a transaction and a directed edge represents a payment flow. Use the publisher's listing to identify the release. [Official dataset on Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) · [Original research paper](https://arxiv.org/abs/1908.02591)

## Local layout

Development and CI work on a clean checkout. When working with an authorised local copy, keep the following files together under `data/raw/` or point `elliptic-check-data` at their existing directory. [File manifest in PyTorch Geometric's Elliptic reader](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/elliptic.html)

| Filename | Role |
| --- | --- |
| `elliptic_txs_classes.csv` | A transaction ID and its class. `1` means illicit, `2` means licit, and `unknown` means unlabelled. |
| `elliptic_txs_features.csv` | Transaction IDs, time steps and anonymised feature values. |
| `elliptic_txs_edgelist.csv` | Pairs of transaction IDs describing directed flows. |

Keep original files in `data/raw/` and future derived tables in `data/processed/`. Both locations are ignored by Git except for README files. Keep records, archives and extracted samples out of version control, published artifacts and CI logs.

## Licence and provenance

On 22 November 2026, Kaggle's metadata for `ellipticco/elliptic-data-set` listed **CC BY-NC-ND 4.0**. Verify the current terms before using a local copy. This licence applies to the dataset independently of any licence chosen for the project code. [Elliptic's Kaggle listing](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) · [Creative Commons licence deed](https://creativecommons.org/licenses/by-nc-nd/4.0/)

For a future experiment, record the release version, access date and file checksums so another researcher can identify the inputs.
