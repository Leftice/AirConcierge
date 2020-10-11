# AirConcierge: Generating Task-Oriented Dialogue via Efficient Large-Scale Knowledge Retrieval


## Overview

This repository contains the PyTorch implementation of text-to-SQL guided dialogue system introduced in the following paper:

> _AirConcierge: Generating Task-Oriented Dialogue via Efficient Large-Scale Knowledge Retrieval_. <br>
**Chieh-Yang Chen**, Pei-Hsin Wang, Shih-Chieh Chang, Da-Cheng Juan, Wei Wei, Jia-Yu Pan. <br>
Link : Will be added when "Findings of EMNLP (2020)" is released

## Introduction
The task-oriented dialogue system is one of the rapidly growing fields with many practical applications, attracting more and more research attention recently. In order to assist users
in solving a specific task while holding conversations with human, the agent needs to understand the intentions of a user during the conversation and fulfills the request. Such a process often involves interacting with external KBs to access task-related information. We propose AirConcierge, an end-to-end trainable text-to-SQL guided framework to learn a neural agent that interacts
with KBs using the generated SQL queries. Specifically, the neural agent first learns to ask and confirm the customer’s intent during the multi-turn interactions, then dynamically determining when to ground the user constraints into executable SQL queries so as to fetch relevant information from KBs. With the help of our method, the agent can use less but more accurate fetched results to generate useful responses efficiently, instead of incorporating the entire KBs. We evaluate the proposed method on the AirDialogue dataset.

## Dependencies

* Cuda compilation tools, release 10.0
* Python 2.7.12
* Pytorch 1.3.1+cu100
* tensorboard 1.14.0
* tensorboardX 1.9
* torchtext 0.4.0

## 1. Prepare Dataset
The download of the AirDialogue dataset, meta data and the pre-processing steps can refer to the [official link](https://github.com/google/airdialogue_model). Since the official pre-processing code and steps may have some little changes, you can also download from the URL provided [here](https://drive.google.com/file/d/1jn6q5g7n4Dv_q2BhMs7j5pbQje91fSaZ/view?usp=sharing). The pre-processing steps are the same as provided by the official, but the older version.
	
	bash ./scripts/download.sh

## 2. Training
For getting baseline results
	
	python main.py --sess Baseline_session
	
For training via Complement objective

	python main.py --COT --sess COT_session

## 3. Evaluating on the AirDialogue dev set


## Benchmark on AirDialogue

The following table shows the best dev sccuracy in a 5-epoch training session. (Please refer to Figure 3a in the paper for details.)

| Model              | Baseline  | COT |
|:-------------------|:---------------------|:---------------------|
| PreAct ResNet-18                |               5.46%  |               4.86%  |


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
The implementation of seq2seq model is adapted from the [pytorch-seq2seq](https://github.com/IBM/pytorch-seq2seq) repository by [IBM](https://github.com/IBM). The code implementation of text-to-SQL refers to [SQLNet](https://github.com/xiaojunxu/SQLNet) repository by [xiaojunxu](https://github.com/xiaojunxu)
