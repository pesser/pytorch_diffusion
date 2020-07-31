# PyTorch pretrained Diffusion Models
A PyTorch reimplementation of [Denoising Diffusion Probabilistic
Models](https://hojonathanho.github.io/diffusion/) with checkpoints converted
from [the author's TensorFlow
implementation](https://github.com/hojonathanho/diffusion).


## Quickstart
Running

```
pip install -e git+https://github.com/pesser/pytorch_diffusion.git#egg=pytorch_diffusion
pytorch_diffusion_demo
```

will start a [Streamlit](https://www.streamlit.io/) demo. It is recommended to
run the demo with a GPU available.

![demo](assets/demo.gif)


## Usage
Diffusion models with pretrained weights for `cifar10`, `lsun-bedroom`,
`lsun_cat` or `lsun_church` can be loaded as follows:

```
from pytorch_diffusion import Diffusion

diffusion = Diffusion.from_pretrained("lsun_church")
samples = diffusion.denoise(4)
diffusion.save(samples, "lsun_church_sample_{:02}.png")
```

Prefix the name with `ema_` to load the averaged weights that produce better
results. The U-Net model used for denoising is available via `diffusion.model`
and can also be instantiated on its own:

```
from pytorch_diffusion import Model

model = Model(resolution=32,
              in_channels=3,
              out_ch=3,
              ch=128,
              ch_mult=(1,2,2,2),
              num_res_blocks=2,
              attn_resolutions=(16,),
              dropout=0.1)
```

This configuration example corresponds to the model used on CIFAR-10.


## Producing samples
If you installed directly from github, you can [find the cloned
repository](https://pip.pypa.io/en/stable/reference/pip_install/#vcs-support)
in `<venv path>/src/pytorch_diffusion` for virtual environments, and
`<cwd>/src/pytorch_diffusion` for global installs. There, you can run

```
python pytorch_diffusion/diffusion.py <name> <bs> <nb>
```

where `<name>` is one of `cifar10`, `lsun-bedroom`, `lsun_cat`, `lsun_church`,
or one of these names prefixed with `ema_`, `<bs>` is the batch size and `<nb>`
the number of batches. This will produce samples from the PyTorch models and
save them to `results/<name>/`.


## Results

Evaluating 50k samples with
[torch-fidelity](https://github.com/toshas/torch-fidelity) gives


| Dataset            | EMA | Framework  | Model            | FID      |
|--------------------|-----|------------|------------------|----------|
| CIFAR10 Train      | no  | PyTorch    | `cifar10`        | 12.13775 |
|                    |     | TensorFlow | `tf_cifar10`     | 12.30003 |
|                    | yes | PyTorch    | `ema_cifar10`    | 3.21213  |
|                    |     | TensorFlow | `tf_ema_cifar10` | 3.245872 |
| CIFAR10 Validation | no  | PyTorch    | `cifar10`        | 14.30163 |
|                    |     | TensorFlow | `tf_cifar10`     | 14.44705 |
|                    | yes | PyTorch    | `ema_cifar10`    | 5.274105 |
|                    |     | TensorFlow | `tf_ema_cifar10` | 5.325035 |


To reproduce, generate 50k samples from the converted PyTorch models provided
in this repo with

```
`python pytorch_diffusion/diffusion.py <Model> 500 100`
```

and with

```
python -c "import convert as m; m.sample_tf(500, 100, which=['cifar10', 'ema_cifar10'])"
```

for the original TensorFlow models.



## Running conversions
The [converted pytorch checkpoints are provided for
download](https://heibox.uni-heidelberg.de/d/01207c3f6b8441779abf/). If you
want to convert them on your own, you can follow the steps described here.

### Setup
This section assumes your working directory is the root of this repository.
Download the pretrained TensorFlow checkpoints. It should follow the original
structure,

```
diffusion_models_release/
  diffusion_cifar10_model/
    model.ckpt-790000.data-00000-of-00001
    model.ckpt-790000.index
    model.ckpt-790000.meta
  diffusion_lsun_bedroom_model/
    ...
  ...
```

Set the environment variable `TFROOT` to the directory where you want to store
the author's repository, e.g.

```
export TFROOT=".."
```

Clone the [diffusion repository](https://github.com/hojonathanho/diffusion),

```
git clone https://github.com/hojonathanho/diffusion.git ${TFROOT}/diffusion
```

and install their required dependencies
(`pip install ${TFROOT}/requirements.txt`). Then add the following to your
`PYTHONPATH`:

```
export PYTHONPATH=".:./scripts:${TFROOT}/diffusion:${TFROOT}/diffusion/scripts:${PYTHONPATH}"
```

### Testing operations
To test the pytorch implementations of the required operations against their
TensorFlow counterparts under random initialization and random inputs, run

```
python -c "import convert as m; m.test_ops()"
```

### Converting checkpoints
To load the pretrained TensorFlow models, copy the weights into the pytorch
models, check for equality on random inputs and finally save the corresponding
pytorch checkpoints, run

```
python -c "import convert as m; m.transplant_cifar10()"
python -c "import convert as m; m.transplant_cifar10(ema=True)"
python -c "import convert as m; m.transplant_lsun_bedroom()"
python -c "import convert as m; m.transplant_lsun_bedroom(ema=True)"
python -c "import convert as m; m.transplant_lsun_cat()"
python -c "import convert as m; m.transplant_lsun_cat(ema=True)"
python -c "import convert as m; m.transplant_lsun_church()"
python -c "import convert as m; m.transplant_lsun_church(ema=True)"
```

Pytorch checkpoints will be saved in

```
diffusion_models_converted/
  diffusion_cifar10_model/
    model-790000.ckpt
  ema_diffusion_cifar10_model/
    model-790000.ckpt
  diffusion_lsun_bedroom_model/
    model-2388000.ckpt
  ema_diffusion_lsun_bedroom_model/
    model-2388000.ckpt
  diffusion_lsun_cat_model/
    model-1761000.ckpt
  ema_diffusion_lsun_cat_model/
    model-1761000.ckpt
  diffusion_lsun_church_model/
    model-4432000.ckpt
  ema_diffusion_lsun_church_model/
    model-4432000.ckpt
```

### Sample TensorFlow models
To produce `N` samples from each of the pretrained TensorFlow models, run

```
python -c "import convert as m; m.sample_tf(N)"
```

Pass a list of model names as keyword argument `which` to specify which models
to sample from. Samples will be saved in `results/`.
