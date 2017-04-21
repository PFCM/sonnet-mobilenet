# MobileNet Sonnet

Implementation of Google's [MobileNet](https://arxiv.org/pdf/1704.04861.pdf)
using Deepmind's
[Sonnet](https://github.com/deepmind/sonnet).
Includes a rough script to test it out on Stanford Dogs
(or some other dataset of images divided up into
folders)

### requirements
[Sonnet](https://github.com/deepmind/sonnet)

[Tensorflow](https://www.tensorflow.org)


### basic usage

```
> python train.py --dogdir=/path/to/dogs
```

Logs will be saved by default in `/tmp/mobilenet`

See `train.py` for more flags.
