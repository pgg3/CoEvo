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