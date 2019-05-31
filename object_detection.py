
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--im_path', type=str, help='path to input image')
#parser.add_argument('--save_path', type=str, help='save path to output image')
args = parser.parse_args()
IM_PATH = args.im_path

def read_image(imPath):
	img = cv2.imread(imPath)
	return img
def non_max_suppression_fast(boxes, overlapThresh):
	if len(boxes) == 0:
		return []

	# boundng boxes co-ordinates

	x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

 
 	# computing areas for all bounding boxes and sorting them
 	# according to the lowest lying box (in y)
 	area = (x2 - x1 + 1) * (y2 - y1 + 1)
 	sortedIndices = np.argsort(y2)
 
 	# we will begin looping from topmost box and find out the 
 	# biggest box which encloses that box

 	# we will also maintain a list of final indices
 	# which are to be considered
 	final = []

	while len(sortedIndices) > 0:
		# grab the top index in the indexes list and add the
		# index value to the list of finaled indexes
		top = len(sortedIndices) - 1
		i = sortedIndices[top]
		final.append(i)
 
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box

		# to get the biggest box enclosing, we would need to find out maximum
		# value of x1, y1 and the minimum value of x2, y2

		x1Max = np.maximum(x1[i], x1[sortedIndices[:top]])
		y1Max = np.maximum(y1[i], y1[sortedIndices[:top]])

		x2Min = np.minimum(x2[i], x2[sortedIndices[:top]])
		y2Min = np.minimum(y2[i], y2[sortedIndices[:top]])

		# calculate the width and height of the boxes 

		w = np.maximum(0, x2Min - x1Max + 1)
		h = np.maximum(0, y2Min - y1Max + 1)

		# Now here's the trick. We are going to calculate overlap ratio (or IoU)
		# for the bb under consideration with all other boxes
		# and if the bb under consideration has overlap ratio more than threshold
		# we will ignore the box under consideration
 
		overlap = (w * h) / area[sortedIndices[:top]]
 
		# removing the boxes which have overlap more than threshold
		sortedIndices = np.delete(sortedIndices, np.concatenate(([top],
			np.where(overlap > overlapThresh)[0])))
 

	return boxes[final].astype("int")


# the path to checkpoint file
FROZEN_GRAPH_FILE = 'frozen_inference_graph.pb'  #path to frozen graph

# load the model

# making an empty graph
graph = tf.Graph()
with graph.as_default():

	serialGraph = tf.GraphDef()
	# we will create a serialized graph as the Protobuf (for which the extension of file is .pb)
	# needs to be read serially in a serial graph
	# we will transfer it later to the empty graph created

	with tf.gfile.GFile(FROZEN_GRAPH_FILE, 'rb') as f:
		serialRead = f.read()
		serialGraph.ParseFromString(serialRead)
		tf.import_graph_def(serialGraph, name = '')

sess = tf.Session(graph = graph)

# scores and num_detections is useless

for dirs in os.listdir(IM_PATH):
	if not dirs.startswith('.'):
		for im in os.listdir(os.path.join(IM_PATH, dirs)):
			if im.endswith('.jpeg'):

				image = read_image(os.path.join(IM_PATH, dirs, im))
				if image is None:
					print('image read as None')
				print('image name: ', im)

				# here we will bring in the tensors from the frozen graph we loaded,
				# which will take the input through feed_dict and output the bounding boxes

				imageTensor = graph.get_tensor_by_name('image_tensor:0')

				bboxs = graph.get_tensor_by_name('detection_boxes:0')

				classes = graph.get_tensor_by_name('detection_classes:0')

				(outBoxes, classes) = sess.run([bboxs, classes],feed_dict={imageTensor: np.expand_dims(image, axis=0)})


				# visualise
				cnt = 0
				imageHeight, imageWidth = image.shape[:2]

				classes = np.squeeze(classes)
				boxes = np.squeeze(outBoxes)

				boxes = np.stack((boxes[:,1] * imageWidth, boxes[:,0] * imageHeight,
								boxes[:,3] * imageWidth, boxes[:,2] * imageHeight),axis=1).astype(np.int)

				boxes = non_max_suppression_fast(boxes, 0.6)

				for i, bb in enumerate(boxes):

					cv2.rectangle(image, (bb[0], bb[1]), (bb[2], bb[3]), (100,100,255), thickness = 1)
				
				cv2.imshow('detected', image)
				cv2.waitKey()
				#plt.figure(figsize = (10, 10))
				#plt.imshow(image)
				#plt.show()

				cv2.imwrite(os.path.join(IM_PATH, dirs, 'a_' + im), image)
