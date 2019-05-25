'comment: for v2, useful data start from 5th element; while for tiny-v2, useful data start from 6th.'

import numpy as np
import tensorflow as tf
import cv2
from skimage.transform import rescale
imgSrc = cv2.imread('./img/sample_car.png', 1)
(H, W, _) = imgSrc.shape
img = np.float32(cv2.cvtColor(imgSrc, cv2.COLOR_BGR2RGB)/255)
#img = np.float32(imgSrc/255)
imgSize = 416
img = cv2.resize(img, (imgSize, imgSize))[np.newaxis, :,:,:]
tiny = np.fromfile('yolov3-tiny.weights', np.float32)[5:]
anchors = np.array([10,14,  23,27,  37,58,  81,82,  135,169,  344,319]).reshape([6,2])

def resize(img, size):
    return rescale(img[0], 2, mode='symmetric')[np.newaxis,:,:,:].astype(np.float32)
#    return cv2.resize(img[0], (int(size[0]), int(size[1])), interpolation=0)[np.newaxis,:,:,:]

def decode(out, imgSize, num_classes=80, anchorTotal=None):
    bboxesTotal = []; obj_probsTotal = []; class_probsTotal = []
    for i, detection_feat in enumerate(out):
        _, H, W, _ = detection_feat.get_shape().as_list()
        anchors = anchorTotal[(1-i)*3:(1-i)*3+3, :]
        num_anchors = len(anchors)
        
        detetion_results = tf.reshape(detection_feat, [-1, H * W, num_anchors, num_classes + 5])
        
        bbox_xy = tf.nn.sigmoid(detetion_results[:, :, :, :2])
        bbox_wh = tf.exp(detetion_results[:, :, :, 2:4])
        obj_probs = tf.nn.sigmoid(detetion_results[:, :, :, 4])
        class_probs = tf.nn.sigmoid(detetion_results[:, :, :, 5:])
    
        anchors = tf.constant(anchors, dtype=tf.float32)
    
        height_ind = tf.range(H, dtype=tf.float32)
        width_ind = tf.range(W, dtype=tf.float32)
        
        x_offset, y_offset = tf.meshgrid(height_ind, width_ind)
        x_offset = tf.reshape(x_offset, [1, -1, 1])
        y_offset = tf.reshape(y_offset, [1, -1, 1])
        
        bbox_x = (bbox_xy[:, :, :, 0] + x_offset) / W
        bbox_y = (bbox_xy[:, :, :, 1] + y_offset) / H
        bbox_w = bbox_wh[:, :, :, 0] * anchors[:, 0] / imgSize * 0.5
        bbox_h = bbox_wh[:, :, :, 1] * anchors[:, 1] / imgSize * 0.5

        bboxes = tf.stack([bbox_x - bbox_w, bbox_y - bbox_h, bbox_x + bbox_w, bbox_y + bbox_h], axis=3)
        bboxesTotal.append(bboxes)
        obj_probsTotal.append(obj_probs)
        class_probsTotal.append(class_probs)
    return tf.concat(bboxesTotal, axis=1), tf.concat(obj_probsTotal, axis=1), tf.concat(class_probsTotal, axis=1)

def conv2d(x, infilters, outfilters, name, ind, tiny, size=3, stride=1, batchnorm=True):
    if batchnorm == True:
        beta, gamma, mean, var = tiny[ind:ind+4*outfilters].reshape([4, outfilters])##
        ind = ind+4*outfilters
    else:
        b = tiny[ind:ind+outfilters]
        bias = tf.Variable(b, name = 'b'+name)
        ind = ind + outfilters
    num = size*size*infilters*outfilters
    w = np.transpose(tiny[ind:ind+num].reshape([outfilters, infilters, size, size]), (2,3,1,0))
    Weights = tf.Variable(w, name = 'W'+name)
    ind = ind + num
    if batchnorm == True:
        xx = tf.nn.conv2d(x, Weights, [1,stride,stride,1], 'SAME')
        xx = tf.contrib.layers.batch_norm(xx, scale=True, param_initializers={'beta':tf.constant_initializer(beta), 'gamma':tf.constant_initializer(gamma), 'moving_mean':tf.constant_initializer(mean), 'moving_variance':tf.constant_initializer(var)}, is_training=False)
        return tf.nn.leaky_relu(xx, 0.1), ind
    else:
        return tf.nn.conv2d(x, Weights, [1, stride,stride,1], 'SAME') + bias, ind
def max_pool(x, size=2, stride=2):
    if stride == 2:
        return tf.nn.max_pool(x, [1,size,size,1], [1,stride,stride,1], 'VALID')
    else:
        return tf.nn.max_pool(x, [1,size,size,1], [1,stride,stride,1], 'SAME')
def route(x1, x2):
    [_, H, W, _]= x1.get_shape().as_list()
    x1 = tf.image.resize_nearest_neighbor(x1, [H*2, W*2])
#    x1 = tf.image.resize_bilinear(x1, [H*2, W*2], True)
#    size = tf.constant([2*W, 2*H], tf.float32)
#    x1 = tf.py_func(resize, [x1, size], tf.float32)
    
#    x1 = tf.pad(x1, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='SYMMETRIC')
#    new_height = 2 * H + 4
#    new_width = 2 * W + 4
#    x1 = tf.image.resize_nearest_neighbor(x1, (new_height, new_width))
#    x1 = x1[:, 2:-2, 2:-2, :]
    return tf.concat([x1, x2], axis=3)
ind = 0
x = tf.placeholder(tf.float32, [None,imgSize,imgSize,3])
out = []
net, ind = conv2d(x, 3, 16, '_conv1', ind, tiny)
net = max_pool(net)
net, ind = conv2d(net, 16, 32, '_conv2', ind, tiny)
net = max_pool(net)
net, ind = conv2d(net, 32, 64, '_conv3', ind, tiny)
net = max_pool(net)
net, ind = conv2d(net, 64, 128, '_conv4', ind, tiny)
net = max_pool(net)
#route2 = net
net, ind = conv2d(net, 128, 256, '_conv5', ind, tiny)
route2 = net
net = max_pool(net)
net, ind = conv2d(net, 256, 512, '_conv6', ind, tiny)
net = max_pool(net, stride=1)
net, ind = conv2d(net, 512, 1024, '_conv7', ind, tiny)

#det1
net, ind = conv2d(net, 1024, 256, '_det1_1', ind, tiny, size=1)
route1 = net
net, ind = conv2d(net, 256, 512, '_det1_2', ind, tiny)
out1, ind = conv2d(net, 512, 255, '_det1_3', ind, tiny, size=1, batchnorm=False)
out.append(out1)

#det2
net, ind = conv2d(route1, 256, 128, '_det2_1', ind, tiny, size=1)
net = route(net, route2)
net, ind = conv2d(net, 384, 256, '_det2_2', ind, tiny)
out2, ind = conv2d(net, 256, 255, '_det2_3', ind, tiny, size=1, batchnorm=False)
out.append(out2)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

bbox, obj_probs, class_probs = sess.run(decode(out, imgSize, anchorTotal=anchors), feed_dict={x:img})

class_probs = np.max(class_probs, axis = 3).reshape([-1])
obj_probs = obj_probs.reshape([-1])
bbox = bbox.reshape([-1,4])
confidence = class_probs * obj_probs
confidence[confidence<0.5] = 0

det_indTotal = []
while (np.max(confidence)!=0):
    det_ind = np.argmax(confidence)
    bbox1_area = (bbox[det_ind,3]-bbox[det_ind,1])*(bbox[det_ind,2]-bbox[det_ind,0])
    sign = 0
    for coor in det_indTotal:
        if coor == None:
            det_indTotal.append(det_ind)
        else:
            xi1 = max(bbox[det_ind, 0], bbox[coor, 0])
            yi1 = max(bbox[det_ind, 1], bbox[coor, 1])
            xi2 = min(bbox[det_ind, 2], bbox[coor, 2])
            yi2 = min(bbox[det_ind, 3], bbox[coor, 3])
            int_area = (yi2-yi1)*(xi2-xi1)
            bbox2_area = (bbox[coor,3]-bbox[coor,1])*(bbox[coor,2]-bbox[coor,0])
            uni_area = bbox1_area + bbox2_area - int_area
            iou = int_area/uni_area
            if iou>0.4:
                sign = 1
                break
    if sign==0:
        det_indTotal.append(det_ind) 
    confidence[det_ind] = 0
    
depict = []
for det_ind in det_indTotal:
    x1,y1,x2,y2 = bbox[det_ind]
    x1 = int(x1*W); x2 = int(x2*W); y1 = int(y1*H); y2 = int(y2*H)
    depict.append([x1,y1,x2,y2])
    cv2.rectangle(imgSrc, (x1,y1), (x2,y2), (255,0,0), 2)
#    imgSrc = cv2.resize(imgSrc, (int(W/4),int(H/4)))
    cv2.imshow('v3-tiny', imgSrc)
