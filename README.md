# Self-Driving-Car-Tensorflow
Autopilot written in Tensorflow for Self Driving Cars. 
# Download the data:
```sh 
./get_data.sh # Downloads the driving data. Around 2.2GB
```
if you are on windows, you can download it from [here.](https://drive.google.com/file/d/0B-KJCaaF7elleG1RbzVPZWV4Tlk/view)

# How to run:
```sh 
python run_webcam.py # runs a live feed from a webcam.
```
```sh 
python run_dataset.py # runs a live feed from a webcam.
```

```sh 
python train.py # The model comes pretrained already. But you can train if you want to. GTX 1070 needed around an hour.
```

# Other datasets
- Udacity: https://medium.com/udacity/open-sourcing-223gb-of-mountain-view-driving-data-f6b5593fbfa5
70 minutes of data ~ 223GB
Format: Image, latitude, longitude, gear, brake, throttle, steering angles and speed
- Udacity Dataset: https://github.com/udacity/self-driving-car/tree/master/datasets [Datsets ranging from 40 to 183 GB in different conditions]
The last two are not optimal for this, since you need to make some code changes. 
- Comma.ai Dataset [80 GB Uncompressed] https://github.com/commaai/research
- Apollo Dataset with different environment data of road: http://data.apollo.auto/?locale=en-us&lang=en


# Credits:
- https://github.com/SullyChen/Autopilot-TensorFlow
- Research paper: End to End Learning for Self-Driving Cars by Nvidia. [https://arxiv.org/pdf/1604.07316.pdf]
- Nvidia blog: https://devblogs.nvidia.com/deep-learning-self-driving-cars/ 
- https://devblogs.nvidia.com/explaining-deep-learning-self-driving-car/

If anything is unclear, just hit me up. 
