import logging

import numpy as np


def calc_channels_means_stds(images):
    """
    Two-pass algorithm from Wikipedia
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Two-pass_algorithm
    """
    nb_channels = images[0].shape[2]

    n = 0
    sum1 = np.zeros((nb_channels,), dtype=np.float32)
    sum2 = np.zeros((nb_channels,), dtype=np.float32)

    for image in images:
        sum1 += image.sum(axis=(0, 1))
        n += image.shape[0] * image.shape[1]

    channel_mean = sum1 / n

    for image in images:
        sum2 += np.sum((image - channel_mean) * (image - channel_mean), axis=(0, 1))

    channel_variance = sum2 / (n - 1)
    channel_std = np.sqrt(channel_variance)

    channel_mean = channel_mean.astype(np.float16)
    channel_std = channel_std.astype(np.float16)

    return channel_mean, channel_std


def normalize_images(images, channel_mean, channel_std):
    images_normalized = np.copy(images).astype(np.float32)

    images_normalized -= channel_mean
    images_normalized /= channel_std

    logging.info('Images normalized: %s', images_normalized.shape)
    return images_normalized
