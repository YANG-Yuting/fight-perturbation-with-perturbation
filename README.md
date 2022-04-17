# FPP

A defense method for word-level robustness.

This is the source code for the paper: "**Word Level Robustness Enhancement: Fight Perturbation with Perturbation**" in AAAI 2022.

# Datasets and models

Datasets and models are [here](https://drive.google.com/drive/folders/1TpSoYxYlj3fk8MUYhJrFE0C7AzSOyM0G?usp=sharing):

Two models (wordLSTM and BERT) for three tasks (MR, IMDB and SNLI).

# Prerequisites

Required packages are listed in the requirements.txt file:

`pip install -r requirements.txt`

We implement two attack algorithms (TextFooler and SemPSO) on four defense methods (Adversarial training, SAFER, FGWS and FPP(ours)). Here are the commands to test defense performances.

# For MR/IMDB

Parameters:

- --task: the task to be attacked (imdb/mr)
- --target_model: the target model to be attacked (wordLSTM/bert)
- --target_model_path: the path of target model to be attacked
- --data_path: the path of data
- --output_dir: the path for output adv examples
- **--kind: choose defense method**
    - org: attack original model
    - Enhance: attack model enhanced by FPP
    - SAFER: attack model enhanced by SAFER
    - FGWS: attack model enhanced by FGWS
    - adv: attack model enhanced by adversarial training

## For TextFooler attack

```
python attack_classification_hownet_top5.py --task mr --target_model wordLSTM --target_model_path /pub/data/huangpei/FPP-AAAI22-release-models/wordLSTM/mr --data_path /pub/data/huangpei/FPP-AAAI22-release-dataset/ --output_dir /pub/data/huangpei/FPP-AAAI22-release-output --gpu_id 0 --kind org

```

If you want to attack model after adversarial training, set “--kind adv” and “target_model_path” to the model after adversarial training. We also offer the adv model with “_adv” suffix.

```
python attack_classification_hownet_top5.py --task imdb --target_model bert --target_model_path /pub/data/huangpei/FPP-AAAI22-release-models/bert/imdb_adv --data_path /pub/data/huangpei/FPP-AAAI22-release-dataset/ --output_dir /pub/data/huangpei/FPP-AAAI22-release-output --gpu_id 0 --kind adv

```

## For SemPSO attack

```
python AD_dpso_sem.py --task mr --target_model bert --target_model_path /pub/data/huangpei/FPP-AAAI22-release-models/bert/mr --data_path /pub/data/huangpei/FPP-AAAI22-release-dataset/ --output_dir /pub/data/huangpei/FPP-AAAI22-release-output --gpu_id 1 --kind org

```

# For SNLI

Parameters:

- --target_model: the target model to be attacked (wordLSTM/bert)
- --target_model_path: the path of target model to be attacked
- --data_path: the path of data
- --output_dir: the path for output adv examples
- **--kind: choose defense method.**
    - org: attack original model
    - Enhance: attack model enhanced by FPP
    - SAFER: attack model enhanced by SAFER
    - FGWS: attack model enhanced by FGWS

## For TextFooler attack

```
cd SNLI
python attack_classification_hownet_top5_snli.py --target_model bert --target_model_path /pub/data/huangpei/FPP-AAAI22-release-models/bert/snli --data_path /pub/data/huangpei/FPP-AAAI22-release-dataset/ --output_dir /pub/data/huangpei/FPP-AAAI22-release-output --gpu_id 2 --kind org

```

If you want to attack model after adversarial training, set “--kind adv” and “target_model_path” to the model after adversarial training. We also offer the adv model with “_adv” suffix.

```
cd SNLI
python attack_classification_hownet_top5_snli.py --target_model bert --target_model_path /pub/data/huangpei/FPP-AAAI22-release-models/bert/snli_adv --data_path /pub/data/huangpei/FPP-AAAI22-release-dataset/ --output_dir /pub/data/huangpei/FPP-AAAI22-release-output --gpu_id 2 --kind adv

```

## For SemPSO attack

```
cd SNLI
python AD_dpso_sem.py --target_model bert --target_model_path /pub/data/huangpei/FPP-AAAI22-release-models/bert/snli --data_path /pub/data/huangpei/FPP-AAAI22-release-dataset/ --output_dir /pub/data/huangpei/FPP-AAAI22-release-output --gpu_id 2 --kind org
```
