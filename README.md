# ShuffleNet
An implementation of `ShuffleNet` introduced in  in TensorFlow. According to the authors, `ShuffleNet` is a computationally efficient CNN architecture designed specifically for mobile devices with very limited computing power. It outperforms `Google MobileNet` by
some error percentage at much lower FLOPs.

Link to the original paper: [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/abs/1707.01083)

## ShuffleNet Unit
<div align="center">
<img src="https://github.com/MG2033/ShuffleNet/blob/master/figures/unit.PNG"><br><br>
</div>

### Group Convolutions
The paper uses the group convolution operator. However, that operator is not implemented in TensorFlow backend. So, I implemented the operator using graph operations. Despite the fact that this is the same operator as the one stated in the paper, it lead to slower performance than the regular convolution. So, to get the same performance stated in the paper, CuDNN efficient implementation for the operator should be done. """CALL FOR CONTRIBUTION"""

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
 tensorflow 1.3.0
 numpy 1.13.1
 tqdm 4.15.0
 bunch 1.0.1
 matplotlib 2.0.2
 ```
### Train and Test
1. Prepare your data, and modify the data_loader.py/DataLoader/load_data() method.
2. Modify the config/test.json to meet your needs.

### Run
```
python main.py config/test.json
```

## License
This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Acknowledgments
Thanks for all who helped me in my work and special thanks for my colleagues: [Mo'men Mohamed](https://github.com/moemen95), and Mohamed Zahran.

