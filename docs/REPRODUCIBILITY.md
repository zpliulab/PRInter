# Reproducibility Details

## Randomness

Final runs used `--seed 42`. Randomness affects negative sampling, random-walk corpus generation, neural encoder initialization, and fold construction.

## Sequence Features

| Molecule | Feature | Dimension |
| --- | --- | ---: |
| lncRNA | normalized 4-mer frequency | 256 |
| protein | conjoint triad feature | 343 |

Protein conjoint triad grouping:

```text
{A,G,V}, {I,L,F,P}, {Y,M,T,S}, {H,N,Q,W}, {R,K}, {D,E}, {C}
```

## Network And Random Walk Hyperparameters

| Module | Hyperparameter | NPInter v2.0 | RAID v2.0 | NPInter v4.0 subset |
| --- | --- | ---: | ---: | ---: |
| Similarity network | Jaccard threshold | 0.5 | 0.5 | 0.5 |
| Walk mode | strategy | metapath | bipartite | metapath |
| Metapaths | patterns | `PPLPLL,LPLPLP` | N/A | `PPLPLL,LPLPLP` |
| Random walk | number of walks | 20 | 20 | 5 |
| Random walk | walk length | 20 | 10 | 20 |
| Random walk | context window | 5 | 5 | 5 |

## Contrastive Learning Hyperparameters

| Hyperparameter | Value |
| --- | ---: |
| negative samples per positive context pair | 5 |
| optimizer | Adam |
| learning rate | 0.001 |
| hidden dimensions | 300, 200 |
| embedding dimension | 200 |
| activation | sigmoid |
| dropout | 0.0 |

| Dataset | Epochs |
| --- | ---: |
| NPInter v2.0 | 5 |
| RAID v2.0 | 5 |
| NPInter v4.0 subset | 4 |

## Negative Interaction Sampling

Negative interaction pairs were generated with DDB-style degree-balanced negative sampling.

## SVM Hyperparameters

| Dataset | Kernel | C | Gamma | Class weight | Feature scaling |
| --- | --- | ---: | ---: | --- | --- |
| NPInter v2.0 | RBF | 0.1 | 0.02 | balanced | StandardScaler |
| RAID v2.0 | RBF | 0.05 | 0.1 | balanced | StandardScaler |
| NPInter v4.0 subset | RBF | 0.5 | 0.02 | balanced | StandardScaler |

## Threshold Justification

The default SVM decision threshold can produce imbalanced precision and recall. Therefore, threshold-adjusted metrics use the threshold that maximizes the Youden index on each evaluation fold:

```text
J = sensitivity + specificity - 1
```

This balances sensitivity and specificity. It should be reported as threshold-adjusted evaluation, not as a default-threshold result.
