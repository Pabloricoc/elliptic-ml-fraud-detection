# Elliptic dataset

This project targets the original Bitcoin transaction graph published by Elliptic in 2019. A node represents a transaction and a directed edge represents a payment flow. Use the publisher's listing to identify the release. [Official dataset on Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) · [Original research paper](https://arxiv.org/abs/1908.02591)

## Local layout

Development and CI work on a clean checkout. When working with an authorised local copy, keep the following files together under `data/raw/` or point `elliptic-check-data` at their existing directory. [File manifest in PyTorch Geometric's Elliptic reader](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/elliptic.html)

| Filename | Role |
| --- | --- |
| `elliptic_txs_classes.csv` | Header `txId,class`. One explicit class per transaction. `1` means illicit, `2` means licit, and `unknown` means unlabelled. |
| `elliptic_txs_features.csv` | Headerless. Each row has 167 fields. The first is the transaction ID and the second is its time step. The remaining 165 fields are anonymous numeric features. |
| `elliptic_txs_edgelist.csv` | Header `txId1,txId2`. The first ID is the source and the second is the target of a directed flow. |

The original publication includes the time step among its 166 features. The loader calls the remaining fields `feature_001` through `feature_165` in file order. These names encode position only. The paper describes 49 snapshots with no edges between them. Its published counts of 203,769 nodes and 234,355 edges serve as size references when a local copy is checked. [Original research, section 2.1](https://arxiv.org/pdf/1908.02591) · [Reference CSV reader](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/elliptic.html)

The package stores IDs as decimal strings to preserve leading zeros and exact joins. It stores time steps as integers from 1 to 49, numeric features as float64 and class codes as a categorical column. Missing class rows, empty labels, non-finite features, duplicate transaction IDs and invalid edge references stop ingestion. A header-only edge list is valid for a subset with isolated nodes. A size discrepancy raises a warning, so smaller local subsets can be audited if they satisfy the same schema and relations.

The loader first streams each file to verify UTF-8, quoting, field counts and the absence of embedded NUL bytes. A second pass loads typed columns into pandas. This catches truncated rows and malformed identifiers while keeping the structural scan to one record at a time.

Keep original files in `data/raw/` and future derived tables in `data/processed/`. Both locations are ignored by Git except for README files. Keep records, archives and extracted samples out of version control, published artifacts and CI logs.

## Licence and provenance

On 22 November 2026, Kaggle's metadata for `ellipticco/elliptic-data-set` listed **CC BY-NC-ND 4.0**. Verify the current terms before using a local copy. This licence applies to the dataset independently of any licence chosen for the project code. [Elliptic's Kaggle listing](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) · [Creative Commons licence deed](https://creativecommons.org/licenses/by-nc-nd/4.0/)

Record the Kaggle version and access date alongside local file checksums. In PowerShell, `Get-FileHash .\data\raw\elliptic_txs_*.csv -Algorithm SHA256` computes the hashes for a local run.
