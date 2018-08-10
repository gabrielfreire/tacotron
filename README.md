# A TensorFlow Implementation of Tacotron: A Fully End-to-End Text-To-Speech Synthesis Model

## Requirements

  * NumPy >= 1.11.1
  * TensorFlow >= 1.3
  * librosa
  * tqdm
  * matplotlib
  * scipy

## Data

We train the model on three different speech datasets.
  1. [LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset/)

LJ Speech Dataset is recently widely used as a benchmark dataset in the TTS task because it is publicly available. It has 24 hours of reasonable quality samples.

## Training
  * STEP 0. Download [LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset/) or prepare your own data.
  * STEP 1. Adjust hyper parameters in `hyperparams.py`. (If you want to do preprocessing, set `prepro` True`.
  * STEP 2. Run `python train.py`. (If you set `prepro` True, run `python prepro.py` first)
  * STEP 3. Run `python eval.py` regularly during training.

## Sample Synthesis

We generate speech samples based on [Harvard Sentences](http://www.cs.columbia.edu/~hgs/audio/harvard.html) as the original paper does. It is already included in the repo.

  * Run `python synthesize.py` and check the files in `samples`.

## Training Curve

<img src="fig/training_curve.png">


## Attention Plot

<img src="fig/attention.gif">

## Pretrained Files
  * Keep in mind 200k steps may not be enough for the best performance.
  * [LJ 200k](https://www.dropbox.com/s/8kxa3xh2vfna3s9/LJ_logdir.zip?dl=0)
  * [WEB 200k](https://www.dropbox.com/s/g7m6xhd350ozkz7/WEB_logdir.zip?dl=0)


## Papers that referenced this repo

  * [Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention](https://arxiv.org/abs/1710.08969)
  * [Storytime - End to end neural networks for audiobooks](http://web.stanford.edu/class/cs224s/reports/Pierce_Freeman.pdf)
