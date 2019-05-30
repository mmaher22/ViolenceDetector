# Violence Detection in Videos


## METHODOLOGY
#### Using CNN-ConvLSTM 
Our main approach was based on where we use a convolutional neural network for feature extraction of difference of frames in a video scene. Then, the output features are introduced to a Conv-LSTM network that can capture the temporal changes. In the Conv-LSTM gates, both the spatial and temporal features can be detected which helps in recognition of motion patterns in the video. 1 shows the used architecture where we tried two different pretrained CNNs on ImageNet for the CNN that extract high level features from difference of consecutive frames. In addition, we tried to concatenate the HOG features of each frame with the output from CNN in order to help the ConvLSTM network in capturing better tempo-spatial features. It is worth mentioning that, the frames of video scene are pre-processed by scaling to a fixed image resolution, and normalization. In addition, during the training process and based on the literature, we augment some other modified frames like horizontal flipped ones, and cropped parts from the frames either from the center or from the corners. These augmented images help in regularization of the network, and avoid over-fitting over scenes used in the training process.

<img src="https://github.com/mmaher22/ViolenceDetector/blob/master/arch1.png">

To run the CNN-ConvLSTM architecture:
We need to define 4 different directories:
1. Training Violence Scenes: Containing folder for training violence scenes and inside each folder the frames of the video has the numbering system as follows: Frame1.JPG, Frame2.JPG, ...., etc
2. Testing Violence Scenes: Containing folder for testing violence scenes in a similar way.
3. Training Non Violence Scenes
4. Testing Non Violence Scenes

```
python main.py --violenceDirTrain [Directory to training Violence scenes and each folder their contains frames for each video] --noviolenceDirTrain [Directory to training NonViolence scenes and each folder their contains frames for each video] --violenceDirTest [Directory to testing Violence scenes and each folder their contains frames for each video] --noviolenceDirTest [Directory to testing NonViolence scenes and each folder their contains frames for each video]
```


#### Detection from Caption Generators
Caption generation from videos is an emerging field and some good results have been reported in the field. In further improvements were reported using bidirectional proposal generation network, and then using the best proposal frame to start with the caption generation. The LSTM context from the proposal generation network is fed into the caption generation decoder to get the captions. We tried to use their implementation form Github 2 to generate captions but have not been able to replicate it to a performance that we can build upon. The plan was to generate the captions on the videos with violence and see if the captions can be used to classify a particular segment of the video. As the region proposal network itself can detect events within a particular video sequence, we adapted this network to provide proposal for violence related events in the sequence. The network used visual features like (HOG) from each frame and a sequence of 150 frames was used
as input to the Bidirectional LSTM. The LSTM output from the forward part of the LSTM encodes the previous context and the reverse LSTM encodes the future context. A sigmoid activation on the output of each of the time frame was calcualted as the proposal score for the network. Finally we combine the forward and reverse proposal score by multiplying to get the final score. We trained the network by labeling a score of 1 for each frame containing violence and 0 for other frames.

<img src="https://github.com/mmaher22/ViolenceDetector/blob/master/arch2.png">

<hr>

## Project Report and Results
FROM <a href = "https://github.com/mmaher22/ViolenceDetector/blob/master/Violence_Detection_Report.pdf"> HERE </a>
