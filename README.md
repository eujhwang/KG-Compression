## Knowledge Graph Compression Enhances Diverse Commonsense Generation

## Introduction

-- This is the pytorch implementation of our [EMNLP 2023](https://aclanthology.org/2023.emnlp-main.37.pdf) paper "*Knowledge Graph Compression Enhances Diverse Commonsense Generation*".

Implementation is based on the pytorch implementation of [ACL 2022 paper](https://arxiv.org/abs/2203.07285) "Diversifying Content Generation for Commonsense Reasoning with Mixture of Knowledge Graph Experts"


## Create an environment

```
transformers==3.3.1
torch==1.7.0
nltk==3.4.5
networkx==2.1
spacy==2.2.1
torch-scatter==2.0.5+${CUDA}
psutil==5.9.0
```

-- For `torch-scatter`, `${CUDA}` should be replaced by either `cu101` `cu102` `cu110` or `cu111` depending on your PyTorch installation. For more information check [here](https://github.com/rusty1s/pytorch_scatter).


## Preprocess the data

-- Extract English ConceptNet and build graph.

```bash
cd data
wget https://s3.amazonaws.com/conceptnet/downloads/2018/edges/conceptnet-assertions-5.6.0.csv.gz
gzip -d conceptnet-assertions-5.6.0.csv.gz
cd ../preprocess
python extract_cpnet.py
python graph_construction.py
```

-- Preprocess multi-hop relational paths. Set `$DATA` to either `anlg` or `eg`.

```bash
export DATA=eg
python ground_concepts_simple.py $DATA
python find_neighbours.py $DATA
python filter_triple.py $DATA
```

## Run Model Example

```
python main.py --assign_ratio=0.8 --data_dir=data/anlg --eval_beams=3 --kg_loss_ratio=0.3 --learning_rate=3e-05 --max_source_length=40 --max_target_length=60 --metric_for_best_model=distinct_2 --mixtures=3 --model_name_or_path=facebook/bart-base --model_type=kgmoe --num_train_epochs=30 --opt_loss_ratio=0.3 --output_dir=output-anlg/KGMixtureOfExpertShen_Output --per_device_eval_batch_size=60 --per_device_train_batch_size=60 --pool_type=sag --seed=1053 --test_max_target_length=60 --val_max_target_length=60 --warmup_steps=0 --weight_decay=0.01 --fp16 --do_train --do_eval --do_predict --predict_with_generate --load_best_model_at_end --overwrite_output_dir --evaluate_during_training --use_wandb
```

assign_ratio: [0.6, 0.7, 0.8]
data_dir: [eg, anlg]
kg_loss_ratio: 0.3
opt_loss_ratio: 0.3
pool_type: sag
weight_decay: [0, 0.01]


## Citation

```
@inproceedings{hwang-etal-2023-knowledge,
    title = "Knowledge Graph Compression Enhances Diverse Commonsense Generation",
    author = "Hwang, EunJeong  and
      Thost, Veronika  and
      Shwartz, Vered  and
      Ma, Tengfei",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.37",
    doi = "10.18653/v1/2023.emnlp-main.37",
    pages = "558--572",
    abstract = "Generating commonsense explanations requires reasoning about commonsense knowledge beyond what is explicitly mentioned in the context. Existing models use commonsense knowledge graphs such as ConceptNet to extract a subgraph of relevant knowledge pertaining to concepts in the input. However, due to the large coverage and, consequently, vast scale of ConceptNet, the extracted subgraphs may contain loosely related, redundant and irrelevant information, which can introduce noise into the model. We propose to address this by applying a differentiable graph compression algorithm that focuses on the relevant knowledge for the task. The compressed subgraphs yield considerably more diverse outputs when incorporated into models for the tasks of generating commonsense and abductive explanations. Moreover, our model achieves better quality-diversity tradeoff than a large language model with 100 times the number of parameters. Our generic approach can be applied to additional NLP tasks that can benefit from incorporating external knowledge.",
}
```

Please kindly cite our paper if you find this paper and the codes helpful.

## Acknowledgements

Many thanks to the Github repository of ACL 2022 paper "[Diversifying Content Generation for Commonsense Reasoning with Mixture of Knowledge Graph Experts](https://arxiv.org/abs/2203.07285)" ([implementation](https://github.com/DM2-ND/MoKGE))

Our codes are modified based on their codes.
