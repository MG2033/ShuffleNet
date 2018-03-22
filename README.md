# ShuffleNet
An implementation of `ShuffleNet` introduced in TensorFlow. According to the authors, `ShuffleNet` is a computationally efficient CNN architecture designed specifically for mobile devices with very limited computing power. It outperforms `Google MobileNet` by
small error percentage at much lower FLOPs.

Link to the original paper: [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/abs/1707.01083)


## ShuffleNet Unit
<div align="center">
<img src="https://github.com/MG2033/ShuffleNet/blob/master/figures/unit.PNG"><br><br>
</div>

### Group Convolutions
The paper uses the group convolution operator. However, that operator is not implemented in TensorFlow backend. So, I implemented the operator using graph operations.

This issue was discussed here: [Support Channel groups in convolutional layers #10482](https://github.com/tensorflow/tensorflow/pull/10482)
## Channel Shuffling
<div align="center">
<img src="https://github.com/MG2033/ShuffleNet/blob/master/figures/shuffle.PNG"><br><br>
</div>

### Channel Shuffling can be achieved by applying three operations:
1. Reshaping the input tensor from (N, H, W, C) into (N, H, W, G, C').
2. Performing matrix transpose operation on the two dimensions (G, C').
3. Reshaping the tensor back into (N, H, W, C). 

    N: Batch size,
    H: Feature map height,
    W: Feature map width,
    C: Number of channels,
    G: Number of groups,
    C': Number of channels / Number of groups

    Note that: The number of channels should be divisible by the number of groups.

## Usage
### Main Dependencies
 ```
 Python 3 or above
 tensorflow 1.3.0
 numpy 1.13.1
 tqdm 4.15.0
 easydict 1.7
 matplotlib 2.0.2
 ```
### Train and Test
1. Prepare your data, and modify the data_loader.py/DataLoader/load_data() method.
2. Modify the config/test.json to meet your needs.

### Run
```
python main.py --config config/test.json
```

## Results
The model have successfully overfitted TinyImageNet-200 that was presented in [CS231n - Convolutional Neural Networks for Visual Recognition](https://tiny-imagenet.herokuapp.com/). I'm working on ImageNet training..

## Benchmarking
The paper has achieved 140 MFLOPs using the vanilla version. Using the group convolution operator implemented in TensorFlow, I have achieved approximately 270 MFLOPs. The paper counts multiplication+addition as one unit, so roughly dividing 270 by two, I have achieved what the paper proposes.

To calculate the FLOPs in TensorFlow, make sure to set the batch size equal to 1, and execute the following line when the model is loaded into memory.
```
tf.profiler.profile(
        tf.get_default_graph(),
        options=tf.profiler.ProfileOptionBuilder.float_operation(), cmd='scope')
```

## TODO
* Training on ImageNet dataset. In progress...

## Updates
* Inference and training are working properly.

## License
This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Acknowledgments
Thanks for all who helped me in my work and special thanks for my colleagues: [Mo'men Abdelrazek](https://github.com/moemen95), and [Mohamed Zahran](https://github.com/moh3th1).

