import caffe
import cv2
import numpy as np
from collections import Counter
# common variables definition
num_to_emotion = {0:'an', 1:'co', 2:'di', 3:'fe', 4:'ha', 5:'ne', 6:'sa', 7:'su'}
batch_size = 10
channel = 1
width = 224
height = 224
open_list = open('$TESTDATADIR/open_list.txt', 'r')
open_result = open('open_result.txt', 'w')
open_data = []
while True:
	line = open_list.readline()
	if line:
		open_data.append(line[:-1])
	else:
		break
self_list = open('$TESTDATADIR/self_list.txt', 'r')
self_result = open('self_result.txt', 'w')
self_data = []
while True:
	line = self_list.readline()
	if line:
		self_data.append(line[:-1])
	else:
		break
open_batch_count = 185
self_batch_count = 48
# net weights definition
net1_weights = '$PROJECTDIR/finetuning/models/resnet_final_finetuning_iter_159000.caffemodel'
net2_weights = '$PROJECTDIR/finetuning/models/resnet_final_finetuning_iter_254000.caffemodel'
net3_weights = '$PROJECTDIR/finetuning/models/resnet_final_finetuning_iter_90000.caffemodel'
net4_weights = '$PROJECTDIR/finetuning/models/resnet_final_finetuning_iter_196000.caffemodel'
net5_weights = '$PROJECTDIR/finetuning/models/resnet_final_finetuning_iter_314000.caffemodel'
# model structure definition
model_def = '$PROJECTDIR/finetuning/deploy.prototxt'



# set caffe
caffe.set_mode_gpu()
caffe.set_device(0)
# define net 1
net1 = caffe.Net(model_def, net1_weights, caffe.TEST)
open_predict1 = []
self_predict1 = []
# predict on open data
for i in range(0, self_batch_count):
	batch_data = []
	for j in range(0, batch_size):
		img = cv2.imread('$TESTDATADIR/' + self_data[i * batch_size + j], 0)
		img = cv2.resize(img, (width, height))
		img = np.reshape(img, (channel, width, height))
		batch_data.append(img)
	batch_data = np.array(batch_data, dtype = int)
	net1.blobs['data'].data[...] = batch_data
	out1 = net1.forward()
	for j in range(0, batch_size):
		self_predict1.append(net1.blobs['fc8'].data[j].argmax())
# predict on self data
for i in range(0, open_batch_count):
	batch_data = []
	for j in range(0, batch_size):
		img = cv2.imread('$TESTDATADIR/' + open_data[i * batch_size + j], 0)
		img = cv2.resize(img, (width, height))
		img = np.reshape(img, (channel, width, height))
		batch_data.append(img)
	batch_data = np.array(batch_data, dtype = int)
	net1.blobs['data'].data[...] = batch_data
	out1 = net1.forward()
	for j in range(0, batch_size):
		open_predict1.append(net1.blobs['fc8'].data[j].argmax())
# predict the rest 6 images
batch_data = []
for j in range(0, 6):
	img = cv2.imread('$TESTDATADIR/' + open_data[1850 + j], 0)
	img = cv2.resize(img, (width, height))
	img = np.reshape(img, (channel, width, height))
	batch_data.append(img)
for j in range(0, batch_size - 6):
	batch_data.append(np.zeros((1, 224, 224)))
batch_data = np.array(batch_data, dtype = int)
net1.blobs['data'].data[...] = batch_data
out1 = net1.forward()
for j in range(0, 6):
	open_predict1.append(net1.blobs['fc8'].data[j].argmax())



# set caffe
caffe.set_mode_gpu()
caffe.set_device(1)
# define net 2
net2 = caffe.Net(model_def, net2_weights, caffe.TEST)
open_predict2 = []
self_predict2 = []
# predict on open data
for i in range(0, self_batch_count):
	batch_data = []
	for j in range(0, batch_size):
		img = cv2.imread('$TESTDATADIR/' + self_data[i * batch_size + j], 0)
		img = cv2.resize(img, (width, height))
		img = np.reshape(img, (channel, width, height))
		batch_data.append(img)
	batch_data = np.array(batch_data, dtype = int)
	net2.blobs['data'].data[...] = batch_data
	out2 = net2.forward()
	for j in range(0, batch_size):
		self_predict2.append(net2.blobs['fc8'].data[j].argmax())
# predict on self data
for i in range(0, open_batch_count):
	batch_data = []
	for j in range(0, batch_size):
		img = cv2.imread('$TESTDATADIR/' + open_data[i * batch_size + j], 0)
		img = cv2.resize(img, (width, height))
		img = np.reshape(img, (channel, width, height))
		batch_data.append(img)
	batch_data = np.array(batch_data, dtype = int)
	net2.blobs['data'].data[...] = batch_data
	out2 = net2.forward()
	for j in range(0, batch_size):
		open_predict2.append(net2.blobs['fc8'].data[j].argmax())
# predict rest 6 images
batch_data = []
for j in range(0, 6):
	img = cv2.imread('$TESTDATADIR/' + open_data[1850 + j], 0)
	img = cv2.resize(img, (width, height))
	img = np.reshape(img, (channel, width, height))
	batch_data.append(img)
for j in range(0, batch_size - 6):
	batch_data.append(np.zeros((1, 224, 224)))
batch_data = np.array(batch_data, dtype = int)
net2.blobs['data'].data[...] = batch_data
out2 = net2.forward()
for j in range(0, 6):
	open_predict2.append(net2.blobs['fc8'].data[j].argmax())



# set caffe
caffe.set_mode_gpu()
caffe.set_device(2)
# define net 3
net3 = caffe.Net(model_def, net3_weights, caffe.TEST)
open_predict3 = []
self_predict3 = []
# predict on open data
for i in range(0, self_batch_count):
	batch_data = []
	for j in range(0, batch_size):
		img = cv2.imread('$TESTDATADIR/' + self_data[i * batch_size + j], 0)
		img = cv2.resize(img, (width, height))
		img = np.reshape(img, (channel, width, height))
		batch_data.append(img)
	batch_data = np.array(batch_data, dtype = int)
	net3.blobs['data'].data[...] = batch_data
	out3 = net3.forward()
	for j in range(0, batch_size):
		self_predict3.append(net3.blobs['fc8'].data[j].argmax())
# predict on self data
for i in range(0, open_batch_count):
	batch_data = []
	for j in range(0, batch_size):
		img = cv2.imread('$TESTDATADIR/' + open_data[i * batch_size + j], 0)
		img = cv2.resize(img, (width, height))
		img = np.reshape(img, (channel, width, height))
		batch_data.append(img)
	batch_data = np.array(batch_data, dtype = int)
	net3.blobs['data'].data[...] = batch_data
	out3 = net3.forward()
	for j in range(0, batch_size):
		open_predict3.append(net3.blobs['fc8'].data[j].argmax())
# predict rest 6 images
batch_data = []
for j in range(0, 6):
	img = cv2.imread('$TESTDATADIR/' + open_data[1850 + j], 0)
	img = cv2.resize(img, (width, height))
	img = np.reshape(img, (channel, width, height))
	batch_data.append(img)
for j in range(0, batch_size - 6):
	batch_data.append(np.zeros((1, 224, 224)))
batch_data = np.array(batch_data, dtype = int)
net3.blobs['data'].data[...] = batch_data
out3 = net3.forward()
for j in range(0, 6):
	open_predict3.append(net3.blobs['fc8'].data[j].argmax())



# set caffe
caffe.set_mode_gpu()
caffe.set_device(1)
# define net 4
net4 = caffe.Net(model_def, net4_weights, caffe.TEST)
open_predict4 = []
self_predict4 = []
# predict on open data
for i in range(0, self_batch_count):
	batch_data = []
	for j in range(0, batch_size):
		img = cv2.imread('$TESTDATADIR/' + self_data[i * batch_size + j], 0)
		img = cv2.resize(img, (width, height))
		img = np.reshape(img, (channel, width, height))
		batch_data.append(img)
	batch_data = np.array(batch_data, dtype = int)
	net4.blobs['data'].data[...] = batch_data
	out4 = net4.forward()
	for j in range(0, batch_size):
		self_predict4.append(net4.blobs['fc8'].data[j].argmax())
# predict on self data
for i in range(0, open_batch_count):
	batch_data = []
	for j in range(0, batch_size):
		img = cv2.imread('$TESTDATADIR/' + open_data[i * batch_size + j], 0)
		img = cv2.resize(img, (width, height))
		img = np.reshape(img, (channel, width, height))
		batch_data.append(img)
	batch_data = np.array(batch_data, dtype = int)
	net4.blobs['data'].data[...] = batch_data
	out4 = net4.forward()
	for j in range(0, batch_size):
		open_predict4.append(net4.blobs['fc8'].data[j].argmax())
# predict rest 6 images
batch_data = []
for j in range(0, 6):
	img = cv2.imread('$TESTDATADIR/' + open_data[1850 + j], 0)
	img = cv2.resize(img, (width, height))
	img = np.reshape(img, (channel, width, height))
	batch_data.append(img)
for j in range(0, batch_size - 6):
	batch_data.append(np.zeros((1, 224, 224)))
batch_data = np.array(batch_data, dtype = int)
net4.blobs['data'].data[...] = batch_data
out4 = net4.forward()
for j in range(0, 6):
	open_predict4.append(net4.blobs['fc8'].data[j].argmax())



# set caffe
caffe.set_mode_gpu()
caffe.set_device(0)
# define net 5
net5 = caffe.Net(model_def, net5_weights, caffe.TEST)
open_predict5 = []
self_predict5 = []
# predict on open data
for i in range(0, self_batch_count):
	batch_data = []
	for j in range(0, batch_size):
		img = cv2.imread('$TESTDATADIR/' + self_data[i * batch_size + j], 0)
		img = cv2.resize(img, (width, height))
		img = np.reshape(img, (channel, width, height))
		batch_data.append(img)
	batch_data = np.array(batch_data, dtype = int)
	net5.blobs['data'].data[...] = batch_data
	out5 = net5.forward()
	for j in range(0, batch_size):
		self_predict5.append(net5.blobs['fc8'].data[j].argmax())
# predict on self data
for i in range(0, open_batch_count):
	batch_data = []
	for j in range(0, batch_size):
		img = cv2.imread('$TESTDATADIR/' + open_data[i * batch_size + j], 0)
		img = cv2.resize(img, (width, height))
		img = np.reshape(img, (channel, width, height))
		batch_data.append(img)
	batch_data = np.array(batch_data, dtype = int)
	net5.blobs['data'].data[...] = batch_data
	out5 = net5.forward()
	for j in range(0, batch_size):
		open_predict5.append(net5.blobs['fc8'].data[j].argmax())
# predict rest 6 images
batch_data = []
for j in range(0, 6):
	img = cv2.imread('$TESTDATADIR/' + open_data[1850 + j], 0)
	img = cv2.resize(img, (width, height))
	img = np.reshape(img, (channel, width, height))
	batch_data.append(img)
for j in range(0, batch_size - 6):
	batch_data.append(np.zeros((1, 224, 224)))
batch_data = np.array(batch_data, dtype = int)
net5.blobs['data'].data[...] = batch_data
out5 = net5.forward()
for j in range(0, 6):
	open_predict5.append(net5.blobs['fc8'].data[j].argmax())

print open_predict1[:10]
print open_predict2[:10]
print open_predict3[:10]
print open_predict4[:10]
print open_predict5[:10]

print self_predict1[:10]
print self_predict2[:10]
print self_predict3[:10]
print self_predict4[:10]
print self_predict5[:10]

vote = []
for i in range(0, len(open_predict1)):
	vote.append(open_predict1[i])
	vote.append(open_predict2[i])
	vote.append(open_predict3[i])
	vote.append(open_predict4[i])
	vote.append(open_predict5[i])
	count = Counter(vote)
	most = count.most_common(1)[0]
	open_result.write(open_data[i][5:9] + '.jpg ' + num_to_emotion[most[0]] + '\n')
	vote = []
for i in range(0, len(self_predict1)):
	vote.append(self_predict1[i])
	vote.append(self_predict2[i])
	vote.append(self_predict3[i])
	vote.append(self_predict4[i])
	vote.append(self_predict5[i])
	count = Counter(vote)
	most = count.most_common(1)[0]
	self_result.write(self_data[i][5:9] + '.jpg ' + num_to_emotion[most[0]] + '\n')
	vote = []
