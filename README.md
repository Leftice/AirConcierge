# AirConcierge: Generating Task-Oriented Dialogue via Efficient Large-Scale Knowledge Retrieval


## Overview

This repository contains the PyTorch implementation of text-to-SQL guided dialogue system introduced in the following paper:

> _AirConcierge: Generating Task-Oriented Dialogue via Efficient Large-Scale Knowledge Retrieval_. <br>
**Chieh-Yang Chen**, Pei-Hsin Wang, Shih-Chieh Chang, Da-Cheng Juan, Wei Wei, Jia-Yu Pan. <br>
Link : Will be added when "Findings of EMNLP (2020)" is released on ACL anthology

## Introduction
The task-oriented dialogue system is one of the rapidly growing fields with many practical applications, attracting more and more research attention recently. In order to assist users
in solving a specific task while holding conversations with human, the agent needs to understand the intentions of a user during the conversation and fulfills the request. Such a process often involves interacting with external KBs to access task-related information. We propose AirConcierge, an end-to-end trainable text-to-SQL guided framework to learn a neural agent that interacts with KBs using the generated SQL queries. Specifically, the neural agent first learns to ask and confirm the customer’s intent during the multi-turn interactions, then dynamically determining when to ground the user constraints into executable SQL queries so as to fetch relevant information from KBs. With the help of our method, the agent can use less but more accurate fetched results to generate useful responses efficiently, instead of incorporating the entire KBs. We evaluate the proposed method on the AirDialogue dataset.

## Dependencies & Prerequisites

* Cuda compilation tools, release 10.0
* Python 2.7.12
* Pytorch 1.3.1+cu100
* tensorflow-gpu (tested on 1.14.0, used in evaluator)
* tensorboard 1.14.0
* tensorboardX 1.9
* torchtext 0.4.0
* argparse, wget, tqdm
You can also set up the environment with the docker file we provided, it has created most of the required package libraries, but you may need to make minor modifications to add some missing packages. You can refer to the following [link](https://github.com/stevenokm/airdialogue-187) to get the detailed steps of the creation. Thanks to [steven](https://github.com/stevenokm/) for providing it.

## 1. Prepare Dataset
The download of the AirDialogue dataset, meta data and the pre-processing steps can refer to the [official link](https://github.com/google/airdialogue_model). Since the official [pre-processing code](https://github.com/google/airdialogue) and steps may have some little changes, you can also download from the URL provided [here](https://drive.google.com/file/d/1rtKhWK4Ca-VBi2gRqEpjuJma_DMjP6W_/view?usp=sharing). The pre-processing steps are the same as provided by the official, but the older version. You can also use the download script to download the data. (We recommend that you use this option to save time.)
```	
bash download.sh
```

## 2. Preprocessing
We preprocess the dataset in order to begin the training and measure the performance of our model. 
For AirDialogue dataset, please use script command :
```
bash processing_data.sh air
```
For Synthesized dataset, please use script command :
```
bash processing_data.sh syn
```

## Quick Start
In order to execute the code quickly and conveniently, the following script contains all the commands for training and evaluation. The commands in 3. (Training) and 4. (Evaluating on the AirDialogue dev set) are specifically used to execute on Airdialogue dataset, and are included in this script. We recommend that you execute this script, you can get all the results at once to save time.
For AirDialogue dataset, please use script command :
```
bash run_air.sh
```
For Synthesized dataset, please use script command :
```
bash run_syn.sh
```
If you want to know the purpose of each instruction in the script and execute it separately, you can refer to 3. (Training) and 4. (Evaluating on the AirDialogue dev set).

## 3. Training

#### Supervised Learning
The fist step is to train our model using supervised learning.
```	
CUDA_VISIBLE_DEVICES=0 python air.py --air --save_dir 'runs/AirConciergeSQL-epoch5/' --model_dir 'checkpoints/AirConciergeSQL-epoch5/' --checkpoint_every 2000 --adam --action_att --clip 1.0 --batch_size 32 --max_len 320 --mask --sigmkb --shuffle --sql
```	
#### Examine Training Meta Information
Training meta data will be written to the output directory, which can be examined using `tensorboard`. The following command will examine the training procedure of the supervised learning model.
```
tensorboard --logdir=./runs/AirConciergeSQL-epoch5/
```

## 4. Evaluating on the AirDialogue dev set
#### Quality of generated queries during inference time.
The fist step is to generate SQL query and check its quality. See Table 2, 3 in the paper for more details.
```
CUDA_VISIBLE_DEVICES=0 python air.py --air --save_dir 'runs/AirConciergeSQL-epoch5/' --model_dir 'checkpoints/AirConciergeSQL-epoch5/' --checkpoint_every 2000 --adam --action_att --clip 1.0 --batch_size 32 --max_len 500 --mask --sigmkb --sql --resume --dev --eval
```
```
python3 utils/lf_ex_acc.py --dev --air
```

#### BLEU score.
The second step would be to generate predictive inference files by running our model on dev.infer.src.data, dev.infer.tar.data and dev.infer.kb.
```
CUDA_VISIBLE_DEVICES=0 python air.py --air --save_dir 'runs/AirConciergeSQL-epoch5/' --model_dir 'checkpoints/AirConciergeSQL-epoch5/' --checkpoint_every 2000 --adam --action_att --clip 1.0 --batch_size 1 --max_len 320 --mask --sigmkb --sql --resume --dev --infer_bleu_prior --mode 'bleu_t1t2'
```

```
python3 utils/self_play_simulate_DB.py --air --infer_dev
```

```
CUDA_VISIBLE_DEVICES=0 python air.py --air --save_dir 'runs/AirConciergeSQL-epoch5/' --model_dir 'checkpoints/AirConciergeSQL-epoch5/' --checkpoint_every 2000 --adam --action_att --clip 1.0 --batch_size 1 --max_len 320 --mask --sigmkb --sql --resume --dev --infer_bleu --mode 'bleu_t1t2'
```

The third step would be to evaluate the model's performance once the predictative files are generated and get bleu score. The following three evaluation methods are provided. You can choose any one. The second is the official [AirDialogue tookit](https://github.com/google/airdialogue). If you cannot install it, you can choose the first method. It is generated by extracting the code in the official toolkit, the two are the same so they are fair, and the third is the old version tookit adopted by the paper because it has not been updated at that time.
```
python evaluator/evaluator_main.py --pred_data ./results/airdialogue/Inference_Bleu/t1t2/dev_inference_out.txt \
	                  --true_data ./data/airdialogue/tokenized/infer/dev.infer.tar.data \
	                  --task infer \
	                  --output ./results/airdialogue/Inference_Bleu/t1t2/dev_bleu.json	
```
	airdialogue score --pred_data results/airdialogue/Inference_Bleu/t1t2/dev_inference_out.txt \
                     --true_data data/airdialogue/tokenized/infer/dev.infer.tar.data \
                     --task infer \
                     --output results/airdialogue/Inference_Bleu/t1t2/dev_bleu.json
```
python evaluator/old_evaluator/evaluator_main.py --pred_data ./results/airdialogue/Inference_Bleu/t1t2/dev_inference_out.txt \
	                  --true_data ./data/airdialogue/tokenized/infer/dev.infer.tar.data \
	                  --task infer \
	                  --output ./results/airdialogue/Inference_Bleu/t1t2/old_dev_bleu.json
```

#### Self-play evaluation.
The fourth step would be to generate predictive self-play files by running our model on dev.selfplay.eval.data and dev.selfplay.eval.kb. See Table 1 in the paper for more details. Because our code only deals with one sample at a time during the evaluation, so the speed is relatively slow. If necessary, you can modifie the code to accelerate it by doing one batch at a time.

```
CUDA_VISIBLE_DEVICES=0 python air.py --air --save_dir 'runs/AirConciergeSQL-epoch5/' --model_dir 'checkpoints/AirConciergeSQL-epoch5/' --checkpoint_every 2000 --adam --action_att --clip 1.0 --batch_size 1 --max_len 320 --mask --sigmkb --sql --resume --dev --self_play_eval_prior
```

```	
python3 utils/self_play_simulate_DB.py --air --self_play_dev
```

```	
CUDA_VISIBLE_DEVICES=0 python air.py --air --save_dir 'runs/AirConciergeSQL-epoch5/' --model_dir 'checkpoints/AirConciergeSQL-epoch5/' --checkpoint_every 2000 --adam --action_att --clip 1.0 --batch_size 1 --max_len 320 --mask --sigmkb --sql --resume --dev --self_play_eval
```

## Citation
If you find this work useful in your research, please cite:
```bash
@inproceedings{chen2020airconcierge,
  title={AirConcierge: Generating Task-Oriented Dialogue via Efficient Large-Scale Knowledge Retrieval},
  author={Chieh-Yang Chen and Pei-Hsin Wang and Shih-Chieh Chang and Da-Cheng Juan and Wei Wei and Jia-Yu Pan},
  booktitle = {Findings of EMNLP 2020},
  year={2020}
}
```

## Acknowledgement
The implementation of seq2seq model is adapted from the [pytorch-seq2seq](https://github.com/IBM/pytorch-seq2seq) repository by [IBM](https://github.com/IBM). The code implementation of text-to-SQL refers to [SQLNet](https://github.com/xiaojunxu/SQLNet) repository by [xiaojunxu](https://github.com/xiaojunxu).
