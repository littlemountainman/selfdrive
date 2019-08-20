# Self-Driving-Car-Keras
Autopilot written in Keras for Self Driving Cars.

# Demo 
Youtube video goes live soon.

# Download the data:
```sh 
./get_data.sh # Downloads the driving data. Around 2.2GB
```
if you are on windows, you can download it from [here.](https://drive.google.com/file/d/0B-KJCaaF7elleG1RbzVPZWV4Tlk/view?usp=sharing)

# How to run:
```sh 
python3 load_data.py # converts data into proper format for training.
```
```sh 
python3 train_model.py # You can train the model on your own, or use my pretrained one. And skip this step
```

```sh 
python app.py # Evaluates the model. You can now use a driving video from youtube or record it yourself. Put the path to the video in app.py line 28.
```

# Credits:
- Research paper: End to End Learning for Self-Driving Cars by Nvidia. [https://arxiv.org/pdf/1604.07316.pdf]
- Nvidia blog: https://devblogs.nvidia.com/deep-learning-self-driving-cars/ 
- https://devblogs.nvidia.com/explaining-deep-learning-self-driving-car/

If anything is unclear, just hit me up. 


