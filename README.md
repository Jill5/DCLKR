# DCLKR

This is our Pytorch implementation for the paper:

> Shuhua Huang, Chenhao Hu, Weiyang Kong, and Yubao Liu. Disentangled Contrastive Learning for Knowledge-aware Recommender System. In ISWC'23.

## Introduction

DCLKR is a disentangled contrastive learning framework for knowledge-aware recommendation, which consists of three main components: (1) Knowledge Graph Disentangling Module, (2) Interaction Graph Disentangling Module, (3) Contrastive Learning Module.

## Environment Requirement

The code has been tested running under Python 3.7.0. The required packages are as follows:

- pytorch == 1.11.0
- numpy == 1.21.6
- scipy == 1.7.3
- sklearn == 0.20.0
- torch_scatter == 2.0.9
- torch_sparse == 0.6.13
- networkx == 2.5

## Usage

To demonstrate the reproducibility of the best performance reported in our paper and faciliate researchers to track whether the model status is consistent with ours, we provide the best parameter settings (might be different for the custormized datasets) in the scripts.

* Book-Crossing

```
python main.py --dataset book --n_factors 3 --context_hops 2 --lambda1 0.01 --lambda2 0.01
```

* MovieLens-1M

```
python main.py --dataset movie --n_factors 3 --context_hops 3 --lambda1 0.1 --lambda2 0.01
```

* Last.FM

```
python main.py --dataset music --n_factors 3 --context_hops 2 --lambda1 0.01 --lambda2 0.01
```

Important argument:

  * `n_factors`: It indicates the number of disentangled aspects.
  * `context_hops`: It indicates the aggregation depth.
  * `lambda1`: It indicates the weight to control the intra-view contrastive loss.
  * `lambda2`: It indicates the weight to control the inter-view contrastive loss.


## Dataset

We provide three processed datasets: Book-Crossing, MovieLens-1M, and Last.FM.

We follow the paper " [Ripplenet: Propagating user preferences on the knowledge
graph for recommender systems](https://github.com/hwwang55/RippleNet)." to process data.


|                       |               | Book-Crossing | MovieLens-1M | Last.FM |
| :-------------------: | :------------ | ----------:   | --------: | ---------: |
| User-Item Interaction | #Users        |      17,860   |    6,036  |      1,872 |
|                       | #Items        |      14,967   |    2,445  |      3,846 |
|                       | #Interactions |     139,746   |  753,772  |      42,346|
|    Knowledge Graph    | #Entities     |      77,903   |    182,011|      9,366 |
|                       | #Relations    |          25   |         12|         60 |
|                       | #Triplets     |   151,500     |  1,241,996|     15,518 |

* `ratings_final.txt`
  * Ratings file.
  * Each line is a rating record: (`user_id`, `item_id`, and `label`). Label `1` means positive, while `0` means negative.
* `kg_final.txt`
  * Triplets file.
  * Each line is a knowledge triplet: (`head_id`, `relation_id`, `tail_id`).

## Reference 

- We partially use the codes of [MCCLK](https://github.com/CCIIPLab/MCCLK).
- You could find all other baselines in Github.
