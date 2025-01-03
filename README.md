# CoEvo: Continual Evolution of Symbolic Solutions Using Large Language Models
## Install Dependency
Requires: python >= 3.10
```shell
conda create -n coevo python=3.10
conda activate coevo
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # install your torch manually
cd CoEvo
pip install -e . 
```

## Run
### model_config
```json
{
  "host": "<--API_ENDPOINT-->",
  "key":"<--API_Key-->",
  "model":"gpt-4o-mini",
  "url": "/v1/chat/completions",
  "timeout": 120
}
```
Requires: python >= 3.10
```shell
python main_run.py --problem oscillation_1
```
 - **Running log supported:**
```shell
conda activate coevo  # tensorboard dependecy is included in the package
cd res
tensorboard.exe --logdir .  # for windows users
```


## Validate the results for the paper
Requires: python >= 3.10
```shell
python test_res.py --problem oscillation_1 --paper coevo --model gpt35
```

## Contact

If you are interested in CoEvo or if you encounter any difficulty using the platform, you can:

1. Join our QQ Group

   <img src="./assets/figs/qq.jpg" style="width: 30%; height: auto;">

2. Contact us through email pingguo5-c@my.cityu.edu.hk


## Citation
```
@misc{guo2024coevocontinualevolutionsymbolic,
      title={CoEvo: Continual Evolution of Symbolic Solutions Using Large Language Models}, 
      author={Ping Guo and Qingfu Zhang and Xi Lin},
      year={2024},
      eprint={2412.18890},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2412.18890}, 
}
```
