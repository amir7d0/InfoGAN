# InfoGAN

This repository contains an implementation of InfoGAN on the MNIST dataset using TensorFlow 2.0.


## Requirements
```sh
tensorflow==2.11.0
tensorflow-probability==0.19.0
numpy==1.23.4
matplotlib==3.6.2
```

## Usage

1. Clone the repository: 
```sh
git clone https://github.com/amir7d0/InfoGAN.git
```
2. Edit the **`config.py`** file to set the training parameters and the dataset to use. Choose *`dataset`* from **['MNIST', 'FashionMNIST', 'SVHN', 'CelebA']**
3. Run the training script:
```sh
python main.py
```


## Files

* `config.py`: Contains all the configuration parameters for training the model.
* `datasets.py`: Contains code for loading and preprocessing the dataset.
* `distributions.py.py`: Contains the code for the distributions.
* `funcs.py`: Contains the code for Callbacks, sample, and plot functions.
* `models.py`: Contains the code for the generator, discriminator, recognition networks.
* `infogan_model.py`: Contains the code for the InfoGAN class and train_step function.
* `train.py`: Contains the code for training the model.

## Results






## References

1. **X. Chen, Y. Duan, R. Houthooft, J. Schulman, I. Sutskever, P. Abbeel.** *"Infogan: Interpretable representation learning by information maximizing generative adversarial nets."* [[arxiv](https://arxiv.org/abs/1606.03657)]
2. **openai/InfoGAN** [[repo](https://github.com/openai/InfoGAN)]
3. **lisc55/InfoGAN** [[repo](https://github.com/lisc55/InfoGAN)]


