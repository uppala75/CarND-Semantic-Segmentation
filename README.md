# Semantic Segmentation


### Project Goals
The goal of this project is to label the pixels of a in images using a Fully Convolutional Network (FCN) as "road" or "not road". This helps the autonomous vehicle in distinguishing the "road" portion of the image as drivable. 


### Model Architecture
The FCN used for this project is based on [this]( https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf) VGG-16 architecture. The FCN was built by converting a pre-trained VGG-16 network.

An Adam Optimizer was used to help the model converge faster, and this model was trained/tested using the KITTI [dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php). The training images in this dataset have labeled data at the pixel level as road/not road.

I trained the model with the following hyper-parameters after some experimentation: Number of epochs: 30, Batch size: 8, Drop out: 0.5, & learning rate: 1e-4

### Results
As shown in the image below, the training loss decreases with the number of epochs

![png](epochsvslossplot.png)

Here are some examples of the output of the network which highlights the portion of the image which is road, in green. 

![png](um_000053.png)
![png](uu_000098.png)
![png](umm_000049.png)
![png](umm_000008.png)
![png](umm_000086.png)

### Comments
Although the model performs relatively well, there are a number of things to improve the performance using image augmentation and iterating with hyper-parameter selection for training. It can further be improved by creating more classes for segmentation. For example, the model can be trained to classify roads, traffic signs, people, cars etc

### The following is Udacity's default README for this project 

----
### Introduction

In this project, you'll label the pixels of a road in images using a Fully Convolutional Network (FCN).

### Setup
##### Frameworks and Packages
Make sure you have the following installed:

- [Python 3](https://www.python.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [NumPy](http://www.numpy.org/)
- [SciPy](https://www.scipy.org/)

##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder  (**all images from the most recent run**)
 
### Tips
- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip)
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [forum post](https://discussions.udacity.com/t/here-is-some-advice-and-clarifications-about-the-semantic-segmentation-project/403100/8?u=subodh.malgonde) for more information.  A summary of additional points, follow. 
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy. 
- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.
 
### Using GitHub and Creating Effective READMEs
If you are unfamiliar with GitHub , Udacity has a brief [GitHub tutorial](http://blog.udacity.com/2015/06/a-beginners-git-github-tutorial.html) to get you started. Udacity also provides a more detailed free [course on git and GitHub](https://www.udacity.com/course/how-to-use-git-and-github--ud775).

To learn about REAMDE files and Markdown, Udacity provides a free [course on READMEs](https://www.udacity.com/courses/ud777), as well. 

GitHub also provides a [tutorial](https://guides.github.com/features/mastering-markdown/) about creating Markdown files.
