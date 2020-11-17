# A Demo Face Recognition Application

By - [Sourav Agrawal](https://github.com/agrawalsourav98)

## Part A

Basic Procedure for training:

1.  Create the dataset (dataset/create_dataset.py)
2.  Load the dataset (dataset/load_dataset.py)
3.  Perform training on ResNet-50(train.py)

For Testing:

1. Perform test directly using test.py

For face extraction while building the dataset, MTCNN module from [facenet-pytorch](https://github.com/timesler/facenet-pytorch) is used. The data and mtcnn inside dataset folder are directly taken without any modification.

The initial weights are taken from [VGGFace2-pytoch](https://github.com/cydonia999/VGGFace2-pytorch) The weights were trained on the VGGFace2 dataset on a ResNet-50 model from scratch. Initialization Weight file : [resnet50_scratch](https://drive.google.com/open?id=1gy9OJlVfBulWkIEnZhGpOLu084RgHw39)

A sample use of the package is shown in google colab.
[Link](https://colab.research.google.com/drive/11Ylg1yQDctZ08plWxFMa1oIeO8xxE16k?usp=sharing)

You can find all the necessary resources [here](https://drive.google.com/drive/folders/1St2JNxyQ4zO3gZXQqOklJq9tPsbWMLq6?usp=sharing)
