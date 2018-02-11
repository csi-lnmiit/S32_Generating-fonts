## Generating Fonts
We want to create a neural network that can generate characters or in other words we will create a font vector (latent vector) which defines a character and then we embed all fonts in a space where similar fonts have similar vectors.

### Requirements
- Python2.7
- Lasagne(to build and train neural networks in Theano)
- Theano
- numpy
- model
- random
- h5py(interface to the HDF5 binary data format)
- math

### Training
We are taking training data set from [Erikbern’s page](https://erikbern.com/2016/01/21/analyzing-50k-fonts-using-deep-neural-networks.html) where he has provided 50k fonts for analysis and reference purposes.
Here's a link to the data which we are using  : [Fonts data](https://drive.google.com/file/d/0B0GtwTQ6IF9AU3NOdzFzUWZ0aDQ/view)

All fonts need to be vertically aligned and scaled to fit bitmap and each character to be scaled to 64*64. 
Some notes on model:
- 4 hidden layers of fully connected layers of width 1024.
- The final layer is a 4096 layer (64 * 64) with sigmoid nonlinearity so that the output is between 0 (white) and 1 (black).
- L1 loss between predictions and target. This works much better than L2 which generates very “gray” images – you can see qualitatively in the pictures above.
- Pretty strong L2 regularization of all parameters.
- Leaky rectified units (alpha=0.01) of nonlinearity on each layer.
- The first layer is 102D – each font is a 40D vector joined with a 62D binary one-hot vector of what is the character.
- Learning rate is 1.0 which is shockingly high – seemed to work well. Decrease by 3x when no improvements on the 10% test set is achieved in any epoch.
- Minibatch size is 512 – seemed like larger minibatches gave faster convergence for some weird reason.
- No dropout, didn’t seem to help. I did add some moderate Gaussian noise (of sigma 0.03) to the font vector and qualitatively it seemed to help a bit.
- Very simple data augmentation by blurring the input randomly with sigma sampled from [0, 1]. My theory was that this would help fitting characters that have thin lines.

