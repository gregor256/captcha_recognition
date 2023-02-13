Captcha recognition
___

### Task

Recognise captcha via neural network. Task is described in `task_description/laba.pdf`.

### Full solution

Full solution is written in `jupyter_notebook/captcha_solver.ipynb`

### Setup

To download files and setup environment run:

```
python -m venv .venv
pip install -r requirements.txt
curl -L $(yadisk-direct https://disk.yandex.ru/d/JQn56xLQ_3QPHw.) -o sample.zip
unzip -o -q sample.zip "samples/samples/*"
``` 

### Execution

1. You may run `jupyter_notebook/captcha_solver.ipynb` in jupyter
2. To execute python code and fit the model:
   Change config in `config/config.yaml` if you want. <br>
   Run:

```commandline
python python/captcha_solver.py
```

Model parameters will be saved at  `result_model/result`.

to load them to your python-code use

```
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()
```