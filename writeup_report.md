# **Behavioral Cloning Project** 

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md (this file) summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

After executing this command, the simulator must be opened and autonomous driving will start when Autonomous Mode is entered.

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network based on [nVidia's proposal](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). Data normalization occurs using a Lambda layer that receives images in the original size, and then immediately crops to eliminate sky and other areas of non interest. The point of cropping here instead of earlier in the code is that GPU cropping is much faster than CPU, thus reducing execution time.

My initial approach was based on the LeNet model, since I was familiar with it from previous exercises. Unfortunately, my vehicle kept driving off-course and behaved erratically.

#### 2. Attempts to reduce overfitting in the model

My initial LeNet model overfitted with my initial epoch parameter (10), which I greatly reduced to around 4. Since I had to discard this model in favor of nVidia's proposal, I was able to lower to 3 epochs initially in order to avoid overfitting. The model was trained and validated using different data sets that result from the validation_split parameter in Keras' fit method.

Afterwards, testing was carried out in the simulator until 100% safe and comfortable driving was achieved on track 1.

For reference, my time spend, validation and loss results for the generated model included in this project are:

* Epoch 1 - 109s - loss: 0.0159 - val_loss: 0.0133
* Epoch 2 - 73s - loss: 0.0135 - val_loss: 0.0123
* Epoch 3 - 74s - loss: 0.0128 - val_loss: 0.0118

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. This is set as a parameter in the model.compile line at the end of the model implementation.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I recorded my own data, but for best experience I used Udacity's supplied example data. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to drive around the track while recording the journey. I imitially thought of using a LeNet model and building upon that, but I found that my vehicle would veer to the left and drive in circles.

I decided to focus on improving the quality of the model input by doing the following:

* Counter-clockwise driving direction throws off steering balance towards the left. By duplicated and flipping images and steering data, balance is restored. 
* Add left and right camera images to feed the model more data. Adding these images also required adding additional steering values with a compensation factor determined by trial & error.

At this stage, my car was able to reach turn just before the bridge without intervention. At that point though, the car veered off course. I decided to crop images in order to trim out unnecessary noise that could have a negative effect on model training. I initially thought of cropping images using CV2 in my initialization loop, but research indicated that leaving this to the model would actually provide better results by delegating this image manipulation to the GPU.

During this entire process, my model was not overfitting at all, but it simply wasn't generating a reliable result for safe driving.

At this point, I switch to nVidia's proposed model architecture which worked out of the box for 99% of the track. A close call at the end prompted me to add the fully connected, activation and dropout layers at the end.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes:

* Initial Lambda normalization step
* Image cropping for unnecessary area removal
* 24 RELU layer, 5x5 sample size
* 36 RELU layer, 5x5 sample size
* 48 RELU layer, 5x5 sample size
* 64 RELU layer, 3x3 sample size
* 64 RELU layer, 3x3 sample size
* Flatten layer
* Fully Connected 100
* ELU Activation
* Dropout (0.5)
* Fully Connected 50
* ELU Activation
* Dropout (0.5)
* Fully Connected 10
* ELU Activation
* Dropout (0.5)
* Fully Connected

Here is a visualization of the architecture citing nVidia's original article:

![nVidia][https://gtoran.github.io/repository-assets/CarND-Behavioral-Cloning-P3/nvidia-cnn-architecture.png]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving, along with left and right camera frames:

![Center Lane Driving - Center Cam](https://gtoran.github.io/repository-assets/CarND-Behavioral-Cloning-P3/center_2016_12_01_13_32_46_587.jpg)

![Center Lane Driving - Left Cam](https://gtoran.github.io/repository-assets/CarND-Behavioral-Cloning-P3/left_2016_12_01_13_32_46_587.jpg)

![Center Lane Driving - Right Cam](https://gtoran.github.io/repository-assets/CarND-Behavioral-Cloning-P3/right_2016_12_01_13_32_46_587.jpg)

To augment the data sat, I also flipped images and angles thinking that this would help compensate a tendency to steer left. For example, here is the above center image along with a flipped version:

![Image Flip - Center Cam](https://gtoran.github.io/repository-assets/CarND-Behavioral-Cloning-P3/center_2016_12_01_13_32_46_587.jpg)

![Image Flip - Center Cam (Flipped)](https://gtoran.github.io/repository-assets/CarND-Behavioral-Cloning-P3/center_2016_12_01_13_32_46_587-flipped.jpg)

In any case, since my manual driving was a little bit erratic, I decided to use Udacity's sample provided data just to make sure that my data quality was optimum. Augmenting data by joining it with my own recording would be a sizeable improvement, although this would require one of the following:

* Model generator that allows selection of data without loading entire dataset in memory
* Higher memory environment

Without this additional augmentation, the amount of data samples for this training set is just under 39000.

### Areas of improvement

* Optimize memory usage by implementing a generator. Initial tests returned erratic behaviors to the car upon retraining.

### Observations

* Removing centered steering angle measurements and camera frames reduced my training set by aprox. 3K images, but resulted in overfitting and erroneous driving behavior afterwards.
