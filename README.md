# Large Language Model-based Human-Agent Collaboration for Complex Task Solving
This repository is based on our paper: Large Language Model-based Human-Agent Collaboration for Complex Task Solving. It contains the human-agent collaboration dataset we generated, as well as demo code for our fine-tuned human-agent collaboration policy model.   
<div  align="center">    
<img src="./pic/pic4.png" width = "600" height = "400" alt="pic" align=center />
</div>

## Overview
- The Code for different datasets is in `hotpotqa/`, `strategyqa/`, and `intercode/`.
  - start training by `scripts/run.sh`
  - local test environment is in `test_data/`
- Human-Agent Collaboration Dataset in `dataset/`

## Usage
### Getting Start
You can use following scripts to install related python package through pip:
```
git clone https://github.com/XueyangFeng/ReHAC.git
cd ReHAC
pip install -r requirements.txt
```

### Constructing Training Data
Here, we give an example where we set $\lambda=0.08$.
```
python data_preprocess.py ./dataset/gpt4/hotpotqa.jsonl 0.08 ./hotpotqa/data/advantage_sample_count_0.08.jsonl
```
The processed training data is then saved in `./hotpotqa/data/advantage_sample_count_0.08.jsonl` and you should set `TRAIN_DATA_DIR` in run.sh to this path.

You can also find the training data we have processed under `hotpotqa/data/advantage` and `strategyqa/data` and `intercode/data/sql` folders.

### Training and Test Process
```
cd hotpotqa/scripts
sh run.sh
```

## Results
We random sample 100 questions for test for each dataset.
The evaluation result of HotpotQA dataset is under the following figure:
<div  align="center">    
<img src="./pic/main_result.png" width = "100%" alt="pic" align=center />
</div>


(a) Human-agent collaboration evaluation. (b) GPT-4-agent collaboration evaluation. The bars below the 0-axis represent the human intervention cost $\lambda C$, the entire columns, composed of the bars above and below the 0-axis, represent the task reward $T$, and the bars above the 0-axis represent the reward $R$ ($R=T - \lambda C$). Numbers within the bars means the human intervention rate. $ReHAC\_{GPT-4}$ and $ReHAC\_{Human}$ represent the policy model trained on GPT-4-agent and human-agent collaboration datasets, respectively. ReHAC outperforms other baselines in human-agent collaboration scenarios.
 

<div  align="center">    
<img src="./pic/curve.png" width = "100%" alt="pic" align=center />
</div>

We provide original evaluation outputs of ReHAC
under `hotpotqa/results`, `strategyqa/results`, and `intercode/results`.



