# Image Generation GANs
A straightforward TensorFlow implementation of Generative Adversarial Networks (GANs) for creating illustrations.

## Generated Images
After training on a custom dataset of 20,000 anime faces, the model generates these images:

## Overfitting Check
To ensure the model doesn't just memorize the training set, generated images are compared to the closest training images by mean squared error. The top row shows generated images, while the columns show the five closest training images.

The results show the generator creates unique images rather than copying the training set.

## How It Works
GANs involve two neural networks: a discriminator and a generator. The discriminator learns to distinguish real images from generated ones, while the generator aims to produce images that the discriminator can't tell are fake. Both networks train together, enhancing the generator's ability to create realistic images.

## Model Architecture
Inspired by [DCGANs](http://arxiv.org/abs/1511.06434) with key modifications:

1. **No Strided Convolutions**: Uses bilinear upsampling and stride-1 convolutions for the generator; discriminator uses stride-1 convolutions and 2x2 max pooling.
2. **Minibatch Discrimination**: Enhances training (see [Improved Techniques for Training GANs](http://arxiv.org/abs/1606.03498)).
3. **Additional Fully Connected Layers**: Both networks include more fully connected layers.
4. **Regularization Term**: Includes an auxiliary z-predictor network to prevent generator collapse, maintaining diverse outputs.

## Training the Model
**Dependencies**: TensorFlow, PrettyTensor, numpy, matplotlib

