"""Quick reading and minimal preprocessing of stanford dogs"""
import os
import re

import tensorflow as tf


def _single_class_reader(directory):
    """Get a reader which reads images one at a time from a particular
    directory."""
    fnames = [os.path.join(directory, fname) for fname in os.listdir(directory)
              if re.search(r'jpe?g', fname)]
    tf.logging.info('class %s, %d files',
                    directory, len(fnames))
    fnames = tf.constant(fnames)
    fname_queue = tf.train.string_input_producer(fnames)
    reader = tf.WholeFileReader()

    key, value = reader.read(fname_queue)
    img_value = tf.image.decode_jpeg(value, channels=3)

    return img_value


def _random_crop_and_scale(img, final_resolution, crop_size):
    """takes random crop of the image (proportionally somewhere around
    crop_size) and rescales it bilinearly to final_resolution."""
    img = tf.expand_dims(img, 0)  # need a fake batch dimension
    crop_size = tf.random_uniform([], minval=crop_size - crop_size/2,
                                  maxval=crop_size + crop_size/2)
    crop_offsets = tf.random_uniform([2], minval=0.0,
                                     maxval=1.0 - (crop_size - crop_size/2))
    boxes = tf.stack([
        crop_offsets[0],  # y1
        crop_offsets[1],  # x1
        crop_offsets[0] + crop_size,  # y2
        crop_offsets[1] + crop_size   # x2
    ])
    boxes = tf.expand_dims(boxes, 0)
    box_ind = [1]

    return tf.squeeze(tf.image.crop_and_resize(img,
                                               boxes,
                                               box_ind,
                                               final_resolution),
                      0)


def dog_tensor(dogdir, batch_size, class_regex='.*', resolution=(224, 224),
               crop_type='random', num_crops=1, crop_size=0.5):
    """Loads some dog images, takes a crop or crops, resizes to given
    resolution and batches up. Data is expected to be in folders per class,
    and can be filtered by a regex. Labels are assigned alphabetically.

    Args:
        dogdir (str): the top level directory of the dataset.
        batch_size (int): the batch size of returned images.
        class_regex (Optional[str or compiled pattern]): a regular expression
            to filter classes.
        resolution (Optional[tuple]): final size of the images -- default is
            the classic (224, 224).
        crop_type (Optional[str]): whether to take random or regularly spaced
            crops. Options are currently just `random` for random crops.
        num_crops (Optional[int]): how many crops to take of each image.
            Currently only supports one, but for validation should support
            several.
        crop_size (Optional[float]): the size of the crops as a fraction of the
            image. If we are taking random crops then we will randomise this
            slightly as well.

    Returns:
        images: a batch of images.
        labels: a batch of labels.
    """
    class_dirs = [os.path.join(dogdir, dirname)
                  for dirname in os.listdir(dogdir)]
    class_dirs = [dirname for dirname in class_dirs
                  if os.path.isdir(dirname)]
    tf.logging.info('%d classes before filtering', len(class_dirs))
    class_dirs = [dirname for dirname in class_dirs
                  if re.search(class_regex, dirname)]
    tf.logging.info('%d classes after filtering', len(class_dirs))

    readers = []
    for i, cdir in enumerate(sorted(class_dirs)):
        img = _single_class_reader(cdir)
        # img = _random_crop_and_scale(img, resolution, crop_size)
        img = tf.image.resize_bilinear(tf.expand_dims(img, 0), resolution)
        img = tf.squeeze(img, 0)
        tf.summary.image('processed/{}'.format(os.path.basename(cdir)),
                         tf.expand_dims(img, 0))
        label = tf.constant(i)
        readers.append((img, label))

    img_batch, label_batch = tf.train.shuffle_batch_join(
        readers, batch_size, batch_size * 3, 50)

    return img_batch, label_batch, len(class_dirs)
