#!/usr/bin/env python
# coding: utf-8

# # When looking through this notebook, please also read the python comments carefully. 
# # First, we import tensorflow and other commomly used libraries

# In[394]:


import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow import keras
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[395]:


# check the version of numpy and tensorflow
print( np.__version__)
print( tf.__version__)


# In[396]:


#check device
device_list = device_lib.list_local_devices()
for d in device_list:
    print( d.name)


# In[397]:


#using banknote data set as we use in homeowork 4


# In[570]:


# load and preprocess data set
train_dat = np.genfromtxt( "train.csv", delimiter=",")
test_dat = np.genfromtxt( "test.csv", delimiter=',')

x_train = train_dat[:, :-1]
y_train = train_dat[:, -1]
y_train[ y_train == 0 ] = 0

x_test = test_dat[:, :-1]
y_test = test_dat[:, -1]
y_test[ y_test == 0] = 0


# In[571]:


x_train.shape, y_train.shape, x_test.shape, y_test.shape


# In[604]:


def build_three_layer_graph_nn(width):
    tf.reset_default_graph()
    
    n_inputs = 4
    n_outputs = 2
    nodes_width = width
    
    #place holders
    tf_x = tf.placeholder( dtype=np.float32, shape=[None, n_inputs])
    tf_y = tf.placeholder( dtype=np.int32, shape = [None,])
    
#     Hidden Layer 1
    W1 = tf.Variable(tf.random_normal([n_inputs, width]), name='W1')
    b1 = tf.Variable(tf.random_normal([width]), name='b1')
#     Hidden Layer 2
    W2 = tf.Variable(tf.random_normal([width, width]), name='W2')
    b2 = tf.Variable(tf.random_normal([width]), name='b2')
#     Hidden Layer 3
    W3 = tf.Variable(tf.random_normal([width, n_outputs]), name='W3')
    b3 = tf.Variable(tf.random_normal([n_outputs]), name='b3')
    
    #graph
    l1 = tf.add(tf.matmul(tf_x,W1),b1)
    l1_val = tf.nn.relu(l1)
    
    l2 = tf.add(tf.matmul(l1_val,W2),b2)
    l2_val = tf.nn.relu(l2)
    
    out = tf.add(tf.matmul(l2_val,W3),b3)
    out_val = tf.nn.relu(out)
    
    loss = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits( labels=tf_y, logits=out_val))
    
    tf_lr = tf.placeholder( dtype=np.float32, shape=[] )
    opt = tf.train.AdamOptimizer()
    tf_train_op = opt.minimize(loss)
    
    return {'tf_x':tf_x, 'tf_y':tf_y,  'tf_loss':loss, 'logits':out_val, 'tf_train_op':tf_train_op}


# In[605]:


def build_five_layer_graph_nn(width):
    tf.reset_default_graph()
    
    n_inputs = 4
    n_outputs = 2
    nodes_width = width
    
    #place holders
    tf_x = tf.placeholder( dtype=np.float32, shape=[None, n_inputs])
    tf_y = tf.placeholder( dtype=np.int32, shape = [None,])
    
#     Hidden Layer 1
    W1 = tf.Variable(tf.random_normal([n_inputs, width]), name='W1')
    b1 = tf.Variable(tf.random_normal([width]), name='b1')
#     Hidden Layer 2
    W2 = tf.Variable(tf.random_normal([width, width]), name='W2')
    b2 = tf.Variable(tf.random_normal([width]), name='b2')
#     Hidden Layer 3
    W3 = tf.Variable(tf.random_normal([width, width]), name='W3')
    b3 = tf.Variable(tf.random_normal([width]), name='b3')
#     Hidden Layer 4
    W4 = tf.Variable(tf.random_normal([width, width]), name='W4')
    b4 = tf.Variable(tf.random_normal([width]), name='b4')
#     Hidden Layer 5
    W5 = tf.Variable(tf.random_normal([width, n_outputs]), name='W5')
    b5 = tf.Variable(tf.random_normal([n_outputs]), name='b5')
    
    #graph
    l1 = tf.add(tf.matmul(tf_x,W1),b1)
    l1_val = tf.nn.relu(l1)
    
    l2 = tf.add(tf.matmul(l1_val,W2),b2)
    l2_val = tf.nn.relu(l2)
    
    l3 = tf.add(tf.matmul(l2_val,W3),b3)
    l3_val = tf.nn.relu(l3)
    
    l4 = tf.add(tf.matmul(l3_val,W4),b4)
    l4_val = tf.nn.relu(l4)
    
    out = tf.add(tf.matmul(l4_val,W5),b5)
    out_val = tf.nn.relu(out)
    
    loss = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits( labels=tf_y, logits=out_val))
    
    tf_lr = tf.placeholder( dtype=np.float32, shape=[] )
    opt = tf.train.AdamOptimizer()
    tf_train_op = opt.minimize(loss)
    
    return {'tf_x':tf_x, 'tf_y':tf_y,  'tf_loss':loss, 'logits':out_val, 'tf_train_op':tf_train_op}


# In[606]:


def build_nine_layer_graph_nn(width):
    tf.reset_default_graph()
    
    n_inputs = 4
    n_outputs = 2
    nodes_width = width
    
    #place holders
    tf_x = tf.placeholder( dtype=np.float32, shape=[None, n_inputs])
    tf_y = tf.placeholder( dtype=np.int32, shape = [None,])
    
#     Hidden Layer 1
    W1 = tf.Variable(tf.random_normal([n_inputs, width]), name='W1')
    b1 = tf.Variable(tf.random_normal([width]), name='b1')
#     Hidden Layer 2
    W2 = tf.Variable(tf.random_normal([width, width]), name='W2')
    b2 = tf.Variable(tf.random_normal([width]), name='b2')
#     Hidden Layer 3
    W3 = tf.Variable(tf.random_normal([width, width]), name='W3')
    b3 = tf.Variable(tf.random_normal([width]), name='b3')
#     Hidden Layer 4
    W4 = tf.Variable(tf.random_normal([width, width]), name='W4')
    b4 = tf.Variable(tf.random_normal([width]), name='b4')
#     Hidden Layer 5
    W5 = tf.Variable(tf.random_normal([width, width]), name='W5')
    b5 = tf.Variable(tf.random_normal([width]), name='b5')
#     Hidden Layer 6
    W6 = tf.Variable(tf.random_normal([width, width]), name='W6')
    b6 = tf.Variable(tf.random_normal([width]), name='b6')
#     Hidden Layer 7
    W7 = tf.Variable(tf.random_normal([width, width]), name='W7')
    b7 = tf.Variable(tf.random_normal([width]), name='b7')
#     Hiden Layer 8
    W8 = tf.Variable(tf.random_normal([width, width]), name='W8')
    b8 = tf.Variable(tf.random_normal([width]), name='b8')
#     Hidden Layer 9
    W9 = tf.Variable(tf.random_normal([width, n_outputs]), name='W9')
    b9 = tf.Variable(tf.random_normal([n_outputs]), name='b9')
    
    #graph
    l1 = tf.add(tf.matmul(tf_x,W1),b1)
    l1_val = tf.nn.relu(l1)
    
    l2 = tf.add(tf.matmul(l1_val,W2),b2)
    l2_val = tf.nn.relu(l2)
    
    l3 = tf.add(tf.matmul(l2_val,W3),b3)
    l3_val = tf.nn.relu(l3)
    
    l4 = tf.add(tf.matmul(l3_val,W4),b4)
    l4_val = tf.nn.relu(l4)
    
    l5 = tf.add(tf.matmul(l4_val,W5),b5)
    l5_val = tf.nn.relu(l5)
    
    l6 = tf.add(tf.matmul(l5_val,W6),b6)
    l6_val = tf.nn.relu(l6)
    
    l7 = tf.add(tf.matmul(l6_val,W7),b7)
    l7_val = tf.nn.relu(l7)
    
    l8 = tf.add(tf.matmul(l7_val,W8),b8)
    l8_val = tf.nn.relu(l8)
    
    out = tf.add(tf.matmul(l8_val,W9),b9)
    out_val = tf.nn.relu(out)
    
    loss = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits( labels=tf_y, logits=out_val))
    
    opt = tf.train.AdamOptimizer()
    tf_train_op = opt.minimize(loss)
    
    return {'tf_x':tf_x, 'tf_y':tf_y, 'tf_loss':loss, 'logits':out_val, 'tf_train_op':tf_train_op}
    


# In[607]:


def predict_nn( model, sess, x):
#     x = x.reshape( len(x), -1)
    logits_val = sess.run( model['logits'], feed_dict = { model['tf_x']: x})
    
    y_pred = np.argmax( logits_val, axis=1)
    return y_pred


def train_model_nn(sess,  model, x, y,val_x, val_y, epochs = 10,batch_size = 100):
    losses_val = []
    num = len(x)
#     x = x.reshape( num, -1)
    
    for i in range( epochs):
        print( "Epoch %d" % ( i + 1))
        
        #shuffle the data 
        random_idx = np.random.permutation( num)
        x = x[random_idx]
        y = y[random_idx]
        
        #go through the whole data set 
        start = 0
        end = batch_size
        while( end < num):
            x_batch = x[start:end]
            y_batch = y[start:end]
            
            feed_dict = {model['tf_x']:x_batch, model['tf_y']: y_batch}
            
            #train one step
            loss, _ = sess.run([ model['tf_loss'], model['tf_train_op']], feed_dict= feed_dict)
            
            #append to loss hist
            losses_val.append( loss)
            start += batch_size
            end += batch_size
        
        #check accuracy
        y_pred = predict_nn( model, sess, x)
        acc =np.sum( y_pred == y) / len( y)
        print( 'train acc = %g' % ( acc))
        
        y_pred = predict_nn( model, sess, val_x)
        acc =np.sum( y_pred == val_y) / len( y_pred)
        print( 'test acc = %g' % acc)
        
    #end training
    return losses_val



# In[608]:


width = [5,10,25,50,100]
for index in width:
    print("Current Width: " + str(index))
    print(" ")
    nn3 = build_three_layer_graph_nn(index) 
    sess =tf.Session()
    sess.run( tf.global_variables_initializer())
    print("Three Layer Neural Network of 10 Epochs Training/Testing Error using RELU")
    hist = train_model_nn( sess, nn3,x_train, y_train, x_test, y_test,epochs=10, batch_size=32)
    
    print(" ")
    nn5 = build_five_layer_graph_nn(index)
    sess =tf.Session()
    sess.run( tf.global_variables_initializer())
    print("Five Layer Neural Network of 10 Epochs Training/Testing Error using RELU")
    hist5 = train_model_nn( sess, nn5,x_train, y_train, x_test, y_test,epochs=10, batch_size=32)
    
    print(" ")
    nn9 = build_nine_layer_graph_nn(index)
    sess =tf.Session()
    sess.run( tf.global_variables_initializer())
    print("Nine Layer Neural Network of 10 Epochs Training/Testing Error using RELU")
    hist9 = train_model_nn( sess, nn9,x_train, y_train, x_test, y_test,epochs=10, batch_size=32)
    
    print(" ")


# In[609]:


def build_three_layer_graph_nn_xavier(width):
    tf.reset_default_graph()
    
    n_inputs = 4
    n_outputs = 2
    nodes_width = width
    
    #place holders
    tf_x = tf.placeholder( dtype=np.float32, shape=[None, n_inputs])
    tf_y = tf.placeholder( dtype=np.int32, shape = [None,])
    
#     Hidden Layer 1
    W1 = tf.Variable(tf.random_normal([n_inputs, width]), name='W1')
    b1 = tf.Variable(tf.random_normal([width]), name='b1')
#     Hidden Layer 2
    W2 = tf.Variable(tf.random_normal([width, width]), name='W2')
    b2 = tf.Variable(tf.random_normal([width]), name='b2')
#     Hidden Layer 3
    W3 = tf.Variable(tf.random_normal([width, n_outputs]), name='W3')
    b3 = tf.Variable(tf.random_normal([n_outputs]), name='b3')
    
    #graph
    l1 = tf.add(tf.matmul(tf_x,W1),b1)
    l1_val = tf.nn.tanh(l1)
    
    l2 = tf.add(tf.matmul(l1_val,W2),b2)
    l2_val = tf.nn.tanh(l2)
    
    out = tf.add(tf.matmul(l2_val,W3),b3)
    out_val = tf.nn.tanh(out)
    
    loss = tf.reduce_sum( tf.nn.sparse_softmax_cross_entropy_with_logits( labels=tf_y, logits=out_val))
    
    opt = tf.train.AdamOptimizer()
    tf_train_op = opt.minimize(loss)
    
    return {'tf_x': tf_x, 'tf_y': tf_y, 'tf_loss': loss, 'logits':out_val, 'tf_train_op': tf_train_op}


# In[610]:


def build_five_layer_graph_nn_xavier(width):
    tf.reset_default_graph()
    
    n_inputs = 4
    n_outputs = 2
    nodes_width = width
    
    #place holders
    tf_x = tf.placeholder( dtype=np.float32, shape=[None, n_inputs])
    tf_y = tf.placeholder( dtype=np.int32, shape = [None,])
    
#     Hidden Layer 1
    W1 = tf.Variable(tf.random_normal([n_inputs, width]), name='W1')
    b1 = tf.Variable(tf.random_normal([width]), name='b1')
#     Hidden Layer 2
    W2 = tf.Variable(tf.random_normal([width, width]), name='W2')
    b2 = tf.Variable(tf.random_normal([width]), name='b2')
#     Hidden Layer 3
    W3 = tf.Variable(tf.random_normal([width, width]), name='W3')
    b3 = tf.Variable(tf.random_normal([width]), name='b3')
#     Hidden Layer 4
    W4 = tf.Variable(tf.random_normal([width, width]), name='W4')
    b4 = tf.Variable(tf.random_normal([width]), name='b4')
#     Hidden Layer 5
    W5 = tf.Variable(tf.random_normal([width, n_outputs]), name='W5')
    b5 = tf.Variable(tf.random_normal([n_outputs]), name='b5')
    
    #graph
    l1 = tf.add(tf.matmul(tf_x,W1),b1)
    l1_val = tf.nn.tanh(l1)
    
    l2 = tf.add(tf.matmul(l1_val,W2),b2)
    l2_val = tf.nn.tanh(l2)
    
    l3 = tf.add(tf.matmul(l2_val,W3),b3)
    l3_val = tf.nn.tanh(l3)
    
    l4 = tf.add(tf.matmul(l3_val,W4),b4)
    l4_val = tf.nn.tanh(l4)
    
    out = tf.add(tf.matmul(l4_val,W5),b5)
    out_val = tf.nn.tanh(out)
    
    loss = tf.reduce_sum( tf.nn.sparse_softmax_cross_entropy_with_logits( labels=tf_y, logits=out_val))
    
    opt = tf.train.AdamOptimizer()
    tf_train_op = opt.minimize(loss)
    
    return {'tf_x': tf_x, 'tf_y': tf_y, 'tf_loss': loss, 'logits':out_val, 'tf_train_op': tf_train_op}


# In[611]:


def build_nine_layer_graph_nn_xavier(width):
    tf.reset_default_graph()
    
    n_inputs = 4
    n_outputs = 2
    nodes_width = width
    
    #place holders
    tf_x = tf.placeholder( dtype=np.float32, shape=[None, n_inputs])
    tf_y = tf.placeholder( dtype=np.int32, shape = [None,])
    
#     Hidden Layer 1
    W1 = tf.Variable(tf.random_normal([n_inputs, width]), name='W1')
    b1 = tf.Variable(tf.random_normal([width]), name='b1')
#     Hidden Layer 2
    W2 = tf.Variable(tf.random_normal([width, width]), name='W2')
    b2 = tf.Variable(tf.random_normal([width]), name='b2')
#     Hidden Layer 3
    W3 = tf.Variable(tf.random_normal([width, width]), name='W3')
    b3 = tf.Variable(tf.random_normal([width]), name='b3')
#     Hidden Layer 4
    W4 = tf.Variable(tf.random_normal([width, width]), name='W4')
    b4 = tf.Variable(tf.random_normal([width]), name='b4')
#     Hidden Layer 5
    W5 = tf.Variable(tf.random_normal([width, width]), name='W5')
    b5 = tf.Variable(tf.random_normal([width]), name='b5')
#     Hidden Layer 6
    W6 = tf.Variable(tf.random_normal([width, width]), name='W6')
    b6 = tf.Variable(tf.random_normal([width]), name='b6')
#     Hidden Layer 7
    W7 = tf.Variable(tf.random_normal([width, width]), name='W7')
    b7 = tf.Variable(tf.random_normal([width]), name='b7')
#     Hiden Layer 8
    W8 = tf.Variable(tf.random_normal([width, width]), name='W8')
    b8 = tf.Variable(tf.random_normal([width]), name='b8')
#     Hidden Layer 9
    W9 = tf.Variable(tf.random_normal([width, n_outputs]), name='W9')
    b9 = tf.Variable(tf.random_normal([n_outputs]), name='b9')
    
    #graph
    l1 = tf.add(tf.matmul(tf_x,W1),b1)
    l1_val = tf.nn.tanh(l1)
    
    l2 = tf.add(tf.matmul(l1_val,W2),b2)
    l2_val = tf.nn.tanh(l2)
    
    l3 = tf.add(tf.matmul(l2_val,W3),b3)
    l3_val = tf.nn.tanh(l3)
    
    l4 = tf.add(tf.matmul(l3_val,W4),b4)
    l4_val = tf.nn.tanh(l4)
    
    l5 = tf.add(tf.matmul(l4_val,W5),b5)
    l5_val = tf.nn.tanh(l5)
    
    l6 = tf.add(tf.matmul(l5_val,W6),b6)
    l6_val = tf.nn.tanh(l6)
    
    l7 = tf.add(tf.matmul(l6_val,W7),b7)
    l7_val = tf.nn.tanh(l7)
    
    l8 = tf.add(tf.matmul(l7_val,W8),b8)
    l8_val = tf.nn.tanh(l8)
    
    out = tf.add(tf.matmul(l8_val,W9),b9)
    out_val = tf.nn.tanh(out)
    
    loss = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits( logits=out_val, labels=tf_y ))
    
    opt = tf.train.AdamOptimizer()
    tf_train_op = opt.minimize(loss)
    
    return {'tf_x': tf_x, 'tf_y': tf_y, 'tf_loss': loss, 'logits':out_val, 'tf_train_op': tf_train_op}
    


# In[614]:


width = [5,10,25,50,100]
for index in width:
    print("Current Width: " + str(index))
    print(" ")
    nn3 = build_three_layer_graph_nn_xavier(index) 
    sess =tf.Session()
    sess.run( tf.global_variables_initializer())
    print("Three Layer Neural Network of 10 Epochs Training/Testing Error using Xavier")
    hist = train_model_nn( sess, nn3,x_train, y_train, x_test, y_test,epochs=10, batch_size=100)
    
    print(" ")
    nn5 = build_five_layer_graph_nn_xavier(index)
    sess =tf.Session()
    sess.run( tf.global_variables_initializer())
    print("Five Layer Neural Network of 10 Epochs Training/Testing Error using Xavier")
    hist5 = train_model_nn( sess, nn5,x_train, y_train, x_test, y_test,epochs=10, batch_size=100)
    
    print(" ")
    nn9 = build_nine_layer_graph_nn_xavier(index)
    sess =tf.Session()
    sess.run( tf.global_variables_initializer())
    print("Nine Layer Neural Network of 10 Epochs Training/Testing Error using Xavier")
    hist9 = train_model_nn( sess, nn9,x_train, y_train, x_test, y_test,epochs=10, batch_size=100)
    
    print(" ")

