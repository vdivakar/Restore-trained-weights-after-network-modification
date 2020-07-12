from utils import *

'''
This file will modify the original network by removing nodes (Layer 3).
We will be restoring weights from the checkpoint of original model.
'''

## Define Placeholders
input_placeholder = tf.placeholder(tf.float32, shape=[None,512,512,3], name="inputs")
label_placeholder = tf.placeholder(tf.float32, shape=[None,512,512,3], name="labels")

## Define variables
with tf.variable_scope("model_vars"):
    w1 = tf.get_variable("w1", [3,3,3,16], initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=False), trainable=True)
    w2 = tf.get_variable("w2", [3,3,16,3],initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=False), trainable=True)
    w3 = tf.get_variable("w3", [3,3,3,3],initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=False), trainable=True)

    b1 = tf.get_variable("b1", [16], initializer=tf.constant_initializer(0), trainable=True)
    b2 = tf.get_variable("b2", [3],  initializer=tf.constant_initializer(0), trainable=True)
    b3 = tf.get_variable("b3", [3],  initializer=tf.constant_initializer(0), trainable=True)

## Define model (& layers)
def original_model():
    conv1   = tf.nn.conv2d(input_placeholder, w1, strides=[1,1,1,1], padding="SAME", name="Layer1_conv")
    conv1_b = tf.nn.bias_add(conv1, b1, name="Layer1_bias")
    act_1   = tf.nn.leaky_relu(conv1_b, name="Layer1_act")
    act_1   = tf.nn.leaky_relu(act_1, name="Layer1_act") #Adding extra node

    conv2   = tf.nn.conv2d(act_1, w2, strides=[1,1,1,1], padding="SAME", name="Layer2_conv")
    conv2_b = tf.nn.bias_add(conv2, b2, name="Layer2_bias")
    act_2   = tf.nn.leaky_relu(conv2_b, name="Layer2_act") #Change relu to leaky_relu
    act_2   = tf.nn.leaky_relu(act_2, name="Layer2_act") #Adding extra node

    conv3   = tf.nn.conv2d(act_2, w3, strides=[1,1,1,1], padding="SAME", name="Layer3_conv")
    output  = tf.nn.bias_add(conv3, b3, name="Layer3_bias")

    skip_connection = tf.add(input_placeholder, output, name="output") #Adding Skip connection
    return skip_connection

## Dataset
tr_input, tr_label = get_tr_dataset()
te_input = get_te_dataset()

## Loss & Optimizer
output_tr = original_model()
output_te = original_model()
loss = tf.losses.mean_squared_error(output_tr, label_placeholder)
train_op = tf.train.AdamOptimizer().minimize(loss)

init = tf.global_variables_initializer()
saver_full = tf.train.Saver()

trainable_vars = get_trainable_vars()
global_vars = get_global_vars()
print_list(global_vars, "Global Variables")
print_list(trainable_vars, "All Trainable Variables")

with tf.Session() as sess:
    # sess.run(init) #Do Not put init after restoring checkpoints
    saver_full.restore(sess, "checkpoint/model-original")

    start = timer()
    for epoch in range(3):
        _, loss_val = sess.run([train_op, loss], feed_dict={
            input_placeholder:tr_input,
            label_placeholder:tr_label
        })
        print("Epcoh: %2d| Loss: %f" %(epoch, loss_val))

    end = timer()
    print("==> Time to train: {t} sec".format(t=end-start))

    save_path = saver_full.save(sess, "checkpoint/model-Case_3")
    print("Model saved in path: %s" % save_path)

    # out_img = sess.run(output_te, feed_dict={input_placeholder:te_input})
    # save_model_out(out_img[0], "test")
