<!---
Copyright 2022 The OFA-Sys Team. 
Copyright 2023 Kai Zhang @ Lehigh. 
All rights reserved.
This source code is licensed under the Apache 2.0 license found in the LICENSE file in the root directory.
-->

# [Nature Medicine'24] BiomedGPT
*A Generalist Vision-Language Foundation Model for Diverse Biomedical Tasks.* (https://arxiv.org/abs/2305.17100)
)

**BiomedGPT** is pre-trained and fine-tuned with multi-modal & multi-task biomedical datasets. This repository guides you to finetune BiomedGPT for VQA tasks on MIMIC-CXR and VinDR-CXR datasets. 

## Installation (Linux)

1. Clone this repository and navigate to the BiomedGPT folder
```bash
git clone https://github.com/hmkhoi2701/BioMedGPT_Finetune
cd BioMedGPT_Finetune/
```

2. Install required packages
```Shell
conda create --name biomedgpt python=3.7.4
python -m pip install pip==21.2.4
pip install -r requirements.txt
```

### Quick Start with Huggingface's transformers

Please check out this [Colab notebook](https://colab.research.google.com/drive/1AMG-OwmDpnu24a9ZvCNvZi3BZwb3nSfS?usp=sharing) for Fairseq-free inference. 

**Warning:** Extensive experiments using transformers have not been conducted, so we cannot confirm whether the results from transformers and fairseq are fully aligned.

## Checkpoints
The authors provided pretrained checkpoints of BiomedGPT (<a href="https://www.dropbox.com/sh/cu2r5zkj2r0e6zu/AADZ-KHn-emsICawm9CM4MqVa?dl=0">Dropbox</a>), which can be put in the `scripts/` folder for finetuning. Here, please download the ```biomedgpt_base.pt``` file.

## Implementation
The preprocessing, pretraining, finetuning and inference scripts are stored in the `scripts/` folder. You can follow the directory setting below:

```
BioMedGPT_Finetune/
├── checkpoints/
├── datasets/
│   ├── pretraining/
│   ├── finetuning/
│   └── ...
├── scripts/
│   ├── preprocess/
│   │   ├── pretraining/
│   │   └── finetuning/
│   ├── pretrain/
│   ├── vqa/
│   └── ...
└── ...
```

## Pretraining
Please follow [datasets.md](datasets.md) to prepare pretraining datasets, which includes 4 TSV files: <code>vision_language.tsv</code>, <code>text.tsv</code>, <code>image.tsv</code> and <code>detection.tsv</code> in the directory of `./datasets/pretraining/`.

<pre>
cd scripts/pretrain
bash pretrain_tiny.sh
</pre>
Feel free to modify the hyperparameters in the bash script for your requirements or ablation study.

### Zero-shot VQA inference using pre-trained checkpoints
Add ```--zero-shot``` argument in the script. Example script: ```/scripts/vqa/evaluate_vqa_rad_zero_shot.sh```.

**Warning:** The current implementation is not yet designed for chatbot or copilot applications, as its primary focus is on learning general representations in medicine that can be transferred to downstream tasks, as outlined in our paper. Large-scale training and instruction tuning for improving robust conversational abilities are still in progress.

## Downstreams
I customized the code to facilitate the VQA task on MIMIC-CXR and VinDR-CXR datasets. Here are the steps

1, Run ```scripts/vqa/generate_tsv.ipynb``` and modify the paths to the image folders. As a result there will be files named ```train.tsv``` and ```test.tsv``` at ```datasets/finetuning```.

2, Run 
```bash
cd scripts/vqa
# for fine-tuning
bash train_vqa_custom.sh
```

3, The finetuned weights will be stored at ```checkpoints```, while the logs will be at ```scripts/vqa/vqa_rad_logs```

# Related Codebase
* [OFA](https://github.com/OFA-Sys/OFA)
* [Fairseq](https://github.com/pytorch/fairseq)
* [taming-transformers](https://github.com/CompVis/taming-transformers)
* [self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch)
<br></br>


# Citation
If you use BiomedGPT model or our code for publications, please cite 🤗: 
```
@article{zhang2024generalist,
  title={A generalist vision--language foundation model for diverse biomedical tasks},
  author={Zhang, Kai and Zhou, Rong and Adhikarla, Eashan and Yan, Zhiling and Liu, Yixin and Yu, Jun and Liu, Zhengliang and Chen, Xun and Davison, Brian D and Ren, Hui and others},
  journal={Nature Medicine},
  pages={1--13},
  year={2024},
  publisher={Nature Publishing Group US New York}
}
```
<br></br>
