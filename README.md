# rPPG_CNN

This repository hosts a CNN model for remote heart rate sensing using frames from a user's face as an input.

# Pre-requisites
1. Pytorch
2. Numpy
3. Pandas
4. TensorboardX

These can be installed using ```pip install <library-name>```

# Training
To train the CNN, there are 3 steps:
1. Pre-processing the video to convert it into an image dataset.
2. Placing ground truth data.
3. Train the CNN on the image dataset obtaines from step 2.

# Pre-process the video files:
Your video files should all be inside one folder.
Then run the following command in your terminal: 
```bash
python data_preprocess.py --video_dir=<path to video folder> --images_dir=<path where to save face images> --rotation=-90 --max_frames=0
```
The ```rotation``` command is used to specify the rotation to apply on a single frame of video, sometimes it can happen that a video is played horizontally, making face detection fail, applying a rotation or 90 or -90 will fix this issue.

The ```max_frames=0``` command can be used to limit the number of frames to extract from the video to cut running time on the script (if the video is too long). Setting it to 0 means all frames are extracted from the video automatically.

# Place the ground truth data files:
Once the video processing is done, make sure that the ground truth files for each of the video have the SAME NAME as the video. Put all these files in the folder you specified as ```---images_dir```.

# Training:
Now training can be run. To start training the CNN, simply do:
```bash
python main.py --images_dir=<path to images dir> --image_size=256 --T=64, --N=32 --batch_size=32 --n_threads=4 --epochs=5 --learning_rate=1e-3 --save_iter=200
```
If you followed all the previous steps correctly, training should now run without problems. To read description of what each of the command line options do, read the "help" parameter in the main.py lines 11-21.

# Tensorboard:
Tensorboardx allows us to save training run data in PyTorch. To monitor the loss function as training progresses, please start tensorboard by running the following script in the terminal:
```bash
tensorboard --logdir=<path to tensorboardx log dir>
```
Then open up your browser and navigate to "localhost:6006" to display the visualizations.

# TODOs:
1. Bug fixes.
2. Make data pipeline better.
3. Make face detector robust to rotation.
4. Make the predict function run correctly.
5. Maybe changes in the algorithmic logic.
6. Verify the rPPG input signal as being correct.
