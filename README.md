# ShuffleNet
An implementation of `ShuffleNet` introduced in  in TensorFlow. According to the authors, `ShuffleNet` is a computationally efficient CNN architecture designed specifically for mobile devices with very limited computing power. It outperforms `Google MobileNet` by
some error percentage at much lower FLOPs.

Link to the original paper: [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/abs/1707.01083)

## ShuffleNet Unit
<div align="center">
<img src="https://github.com/MG2033/ShuffleNet/blob/master/figures/unit.PNG"><br><br>
</div>

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
