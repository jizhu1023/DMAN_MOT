#!/usr/bin/env python
import cv2
import os
import numpy as np
import scipy.io as sio
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras import backend as K
from os.path import expanduser
from spatial_temporal_attention_network import generate_model
import socket

# determine whether runing on MOT training set or test set
dataset = 'train' # or 'test'
 
# communicate with the matlab program using the socket
host = '127.0.0.1' 
port = 65431 
socket_tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
socket_tcp.bind((host, port))
socket_tcp.listen(5)
print('The python socket server is ready. Waiting for the signal from the matlab socket client ...')
connection, adbboxess = socket_tcp.accept()

try:
    K.set_learning_phase(0)
    model = generate_model()
    weights = 'model/spatial_temporal_attention_model.h5'
    model.load_weights(weights, by_name=True)
    print 'load weights done!'

    home = expanduser("~")
    while 1:
        flag = connection.recv(1024)
        if not flag:
            break
        elif flag != 'client ok':
            print(flag)
        else:
            print(flag)
            mat = sio.loadmat('mot_py.mat') # saved by the matlab program
            seq_name = mat['seq_name'][0].encode('ascii', 'ignore')
            traj_dir = mat['traj_dir'][0].encode('ascii', 'ignore')
            frame_id = int(mat['frame_id_double'][0, 0])
            target_id = traj_dir.split('/')[-2]

            x_det = mat['bboxes']['x'][0, 0]
            y_det = mat['bboxes']['y'][0, 0]
            w_det = mat['bboxes']['w'][0, 0]
            h_det = mat['bboxes']['h'][0, 0]
            num_det = x_det.shape[0]

            time_steps = 8
            frame_path = home + '/data/MOT16/' + dataset + '/' + seq_name + '/img1/' + '{:06d}.jpg'.format(frame_id)
            data = np.zeros((1, time_steps, 224, 224, 6), dtype=np.float32)
            img_frame = image.load_img(frame_path)
            img_w = img_frame.size[0]
            img_h = img_frame.size[1]
            subfiles = os.listdir(traj_dir)
            subfiles.sort()
            img_traj_list = []

            for subfile in subfiles:
                if subfile[-3:] == 'jpg':
                    img_traj_list.append(subfile)
            num_traj = len(img_traj_list)
            if num_traj < time_steps:
                tmp_list = img_traj_list[::-1]
                while len(img_traj_list) < time_steps:
                    img_traj_list += tmp_list
                img_traj_list = img_traj_list[0:time_steps]
            else:
                gap = num_traj / time_steps
                mod = num_traj % time_steps
                tmp_list = img_traj_list
                img_traj_list = []
                for i in range(mod, num_traj, gap):
                    img_traj_list.append(tmp_list[i])

            for i in range(time_steps):
                img = image.load_img(traj_dir + img_traj_list[i])
                img_traj = image.img_to_array(img.resize((224, 224)))
                img_traj = np.expand_dims(img_traj, axis=0)
                img_traj = preprocess_input(img_traj)
                data[0, i, :, :, 3:] = img_traj.copy()

            prediction = np.zeros(num_det, dtype = np.float32)
            for i in range(num_det):
                x1 = int(x_det[i, 0])
                y1 = int(y_det[i, 0])
                w = int(w_det[i, 0])
                h = int(h_det[i, 0])
                x2 = x1 + w
                y2 = y1 + h
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img_w, x2)
                y2 = min(img_h, y2)

                img_det = image.img_to_array(img_frame.crop((x1, y1, x2, y2)).resize((224, 224)))
                img_det = np.tile(img_det, (time_steps, 1, 1, 1))
                data[0, :, :, :, 0:3] = preprocess_input(img_det)

                output = model.predict_on_batch(data)
                prediction[i] = output[2][0, 1]

            sio.savemat('similarity.mat', {'similarity': prediction})
            connection.sendall('server ok')
            print('server ok')
finally:
    connection.close()
    socket_tcp.close()
    print('python server closed.')
