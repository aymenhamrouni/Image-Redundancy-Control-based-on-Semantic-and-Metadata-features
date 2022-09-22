import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from keras.losses import binary_crossentropy
from tensorflow.keras.losses import cosine_similarity 

tf.compat.v1.disable_eager_execution()

import numpy as np

def display_single_subplot(img, n_row, n_col, cell_num):
	ax = plt.subplot(n_row, n_col, cell_num)
	plt.imshow(img)
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

def get_sorted_similarity_idx(encoder, img_to_find_idx, dataset, loss='binary_crossentropy'):
	encoded_images = encoder.predict(dataset)
	encoded_images = encoded_images.reshape(encoded_images.shape[0], -1)

	#initializing vars to pass into tensorflow
	X_selected = [encoded_images[img_to_find_idx].tolist() for _ in range(encoded_images.shape[0])]
	X_all = encoded_images.tolist()

	X_selected_tf = tf.Variable(X_selected, tf.float32)
	X_all_tf = tf.Variable(X_all, tf.float32)
	if loss=='binary_crossentropy':
		loss_tf = binary_crossentropy(X_selected_tf, X_all_tf)
	elif loss=='cosine_proximity':
		loss_tf = cosine_similarity(X_selected_tf, X_all_tf)
	else:
		print('Unknown loss, using binary_crossentropy.')
		loss_tf = binary_crossentropy(X_selected_tf, X_all_tf)
	init_op = tf.global_variables_initializer()

	similarity = []
	with tf.Session() as sess:
	    sess.run(init_op)
	    similarity = sess.run(loss_tf)
	similarity_sorted = np.argsort(np.array(similarity))[1:] #the same figure appears in X_all too, so remove it
	return similarity_sorted
    
    
    
def get_sorted_similarity_idx_2(encoder, decoder, img_to_find_idx, dataset, loss='binary_crossentropy'):
#     encoder.compile()
#     encoder.run_eagerly = True


    Encoded=encoder.predict(dataset)
    Encoded=np.delete(Encoded,2,axis=0)
    encoded_images=decoder.predict(Encoded[0])




    
    encoded_images = encoded_images.reshape(encoded_images.shape[0], -1)

    
    #initializing vars to pass into tensorflow
    X_selected = [encoded_images[img_to_find_idx].tolist() for _ in range(encoded_images.shape[0])]
    X_all = encoded_images.tolist()

    X_selected_tf = tf.Variable(X_selected, tf.float32)
    X_all_tf = tf.Variable(X_all, tf.float32)
    if loss=='binary_crossentropy':
        loss_tf = binary_crossentropy(X_selected_tf, X_all_tf)
    elif loss=='cosine_proximity':
        loss_tf = cosine_similarity(X_selected_tf, X_all_tf)
    else:
        print('Unknown loss, using binary_crossentropy.')
        loss_tf = binary_crossentropy(X_selected_tf, X_all_tf)
    init_op = tf.compat.v1.global_variables_initializer()


    similarity = []
    with tf.Session() as sess:
        sess.run(init_op)
        similarity = sess.run(loss_tf)
    similarity_sorted = np.argsort(np.array(similarity))[1:] #the same figure appears in X_all too, so remove it
    return similarity_sorted