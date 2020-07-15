# Restore-trained-weights-after-network-modification
Any experimental setup code to demonstrate how you can restore saved checkpoint after altering network graph and resume training.
<br>[TensorFlow v1.x]

__Usage:__ 
<br> 1. Run `python original_model.py` to train the base model for few epochs and create a checkpoint.
<br> 2. Run modified network in the given python files to load the checkpoint created above and resume training.


<br> For more details: https://www.divakar-verma.com/post/restoring-trained-weights-after-network-modification
#### Base Network model
![Base Model](https://github.com/vdivakar/Restore-trained-weights-after-network-modification/blob/master/blog-post-data/original_model.png)

__Case 1 & 2__
<br>Removing nodes from network. See `Case_1-2.py`
<br> ![Removing Nodes](https://github.com/vdivakar/Restore-trained-weights-after-network-modification/blob/master/blog-post-data/case_1-2_.png)

__Case 3__
<br>Adding non-trainable variables to the network. See `Case_3.py`
<br> ![Add non-trainable vars](https://github.com/vdivakar/Restore-trained-weights-after-network-modification/blob/master/blog-post-data/case_3.png)

__Case 4__
<br>Adding trainable variables to the network. See `Case_4.py`
<br> ![Add trainable vars](https://github.com/vdivakar/Restore-trained-weights-after-network-modification/blob/master/blog-post-data/case_4.png)
<br> ![vars](https://github.com/vdivakar/Restore-trained-weights-after-network-modification/blob/master/blog-post-data/icon.png)

<br> Feel free to play around with the code and to raise a pull request.
