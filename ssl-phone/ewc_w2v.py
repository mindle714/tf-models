"""
An implementation of Elastic Weight Consolidation from the paper, "Overcoming
catastrophic forgetting in neural networks".

https://arxiv.org/abs/1612.00796
"""
from copy import deepcopy
import numpy as np
import tensorflow as tf


def fisher_matrix(model, data):
    """
    Compute the Fisher matrix, representing the importance of each weight in the
    model. This is approximated using the variance of the gradient of each
    weight, for some number of samples from the dataset.

    :return: The main diagonal of the Fisher matrix, shaped to match the weights
             returned by `model.trainable_weights`.
    """
    weights = model.trainable_weights
    is_unet = weights[0].name.startswith('wav2vec2_unet')

    if is_unet:
        weights = [e for e in weights if 'wav2vec2_unet/conv' not in e.name]
        assert len(model.trainable_weights) - len(weights) == 50

    else:
        weights = [e for e in weights if 'wav2vec2_phone/dense' not in e.name]
        assert len(model.trainable_weights) - len(weights) == 4

    variance = [tf.zeros_like(tensor) for tensor in weights]

    _pcm, _pcm_len = data
    batch_size = _pcm.shape[0]

    for idx in range(batch_size):
        pcm = _pcm[idx]; pcm_len = _pcm_len[idx]
        pcm = tf.expand_dims(pcm, 0)
        pcm_len = tf.expand_dims(pcm_len, 0)

        with tf.GradientTape() as tape:
            loss = model((pcm, pcm_len), ssl_only_ewc = True)
        
        gradients = tape.gradient(loss, weights)

        # If the model has converged, we can assume that the current weights
        # are the mean, and each gradient we see is a deviation. The variance is
        # the average of the square of this deviation.
        _variance = []
        for var, grad in zip(variance, gradients):
            if grad is not None:
                _variance.append(var + (grad ** 2))
            else:
                _variance.append(var)
        variance = _variance

    fisher_diagonal = [tensor / batch_size for tensor in variance]
    return fisher_diagonal


def ewc_loss(lam, model, optimal_weights, samples):
    """
    Generate a loss function which will penalise divergence from the current
    state. It is assumed that the model achieves good accuracy on `dataset`,
    and we want to preserve this behaviour.

    The penalty is scaled according to how important each weight is for the
    given dataset, and `lam` (lambda) applies equally to all weights.
    """
    if optimal_weights[0].name.startswith('wav2vec2_unet'):
        optimal_weights = [e for e in optimal_weights if 'wav2vec2_unet/conv' not in e.name]
    else:
        optimal_weights = [e for e in optimal_weights if 'wav2vec2_phone/dense' not in e.name]
    fisher_diagonal = fisher_matrix(model, samples)

    def loss_fn(new_model):
        # We're computing:
        # sum [(lambda / 2) * F * (current weights - optimal weights)^2]
        loss = 0
        current = new_model.trainable_weights
        if current[0].name.startswith('wav2vec2_unet'):
            current = [e for e in current if 'wav2vec2_unet/conv' not in e.name]
        else:
            current = [e for e in current if 'wav2vec2_phone/dense' not in e.name]

        for f, c, o in zip(fisher_diagonal, current, optimal_weights):
            loss += tf.reduce_sum(f * ((c - o) ** 2))

        return loss * (lam / 2)

    return loss_fn


def fim_mask(model, dataset, samples, threshold):
    """
    Generate a mask for a model. Where the mask is 1, the model's weight may
    be modified.

    :param model: Model optimised for the given dataset.
    :param dataset: NumPy arrays (inputs, labels).
    :param samples: Number of samples of dataset to take when estimating
                    importance of weights. More samples improves estimates.
    :param threshold: Weights with importance > threshold may not be trained
                      further.
    :return: Boolean mask indicating which weights can continue training.
    """
    fisher_diagonal = fisher_matrix(model, dataset, samples)
    mask = [tensor < threshold for tensor in fisher_diagonal]
    return mask


def combine_masks(mask1, mask2):
    """
    :param mask1: A mask generated using `fim_mask`.
    :param mask2: A mask generated using `fim_mask`.
    :return: A new mask, the composition of the two inputs.
    """
    if mask1 is None:
        return mask2
    elif mask2 is None:
        return mask1
    else:
        return [tf.logical_and(tensor1, tensor2)
                for tensor1, tensor2 in zip(mask1, mask2)]


def apply_mask(gradients, mask):
    """
    Apply a gradient mask to a model's gradients.

    :param gradients: Gradients for weights.
    :param mask: Mask indicating which weights are allowed to be updated.
    :return: Gradients with same shape, but some values set to 0.
    """
    if mask is not None:
        gradients = [grad * tf.cast(mask, tf.float32)
                     for grad, mask in zip(gradients, mask)]
    return gradients
