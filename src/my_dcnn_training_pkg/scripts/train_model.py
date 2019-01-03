#!/usr/bin/env python

# We load Python stuff first because afterwards it will be removed to avoid error with openCV
import sys
sys.path.append('/usr/lib/python2.7/dist-packages')
import rospy
import rospkg
import time
import csv
import math
print("Start Module Loading...Remove CV ROS-Kinetic version due to incompatibilities")
#print(sys.path)
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except ValueError:
    print ("Its already removed..../opt/ros/kinetic/lib/python2.7/dist-packages")
#print(sys.path)
import cv2
import numpy as np
from keras import Model
from keras.applications.mobilenetv2 import MobileNetV2, preprocess_input
from keras_applications.mobilenet_v2 import _inverted_res_block
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.layers import MaxPooling2D, Conv2D, Reshape
from keras.utils import Sequence
from keras.optimizers import Adam
import keras.backend as K

import os
import shutil

class DataSequence(Sequence):

    def __load_images(self, dataset):
        return np.array([cv2.imread(f) for f in dataset], dtype='f')

    def __init__(self, csv_file, batch_size=32, inmemory=False, number_of_elements_to_be_output=2):
        self._number_of_elements_to_be_output = number_of_elements_to_be_output
        
        assert (self._number_of_elements_to_be_output == 2), "We only support 2 output model XY"
        
        self.paths = []
        self.batch_size = batch_size
        self.inmemory = inmemory

        with open(csv_file, "r") as file:
            """
            Which element do we have interest in?
            [absolute_original_path, width, height, x_com, y_com, z_com, quat_x, quat_y, quat_z, quat_w, class_name, class_names[class_name]]
            We only want the dcnn to learn to give us the XYZ 3D position of the ObjectTo learn.
            x_com
            y_com
            z_com
            """
            self.y = np.zeros((sum(1 for line in file), self._number_of_elements_to_be_output))
            file.seek(0)

            reader = csv.reader(file, delimiter=",")
            for index, (scaled_img_path, _, _, x_com, y_com, z_com, _, _, _, _, _, _) in enumerate(reader):
                """
                if self._number_of_elements_to_be_output == 3:
                    self.y[index][0] = x_com
                    self.y[index][1] = y_com
                    self.y[index][2] = z_com
                """
                
                if self._number_of_elements_to_be_output == 2:
                    self.y[index][0] = x_com
                    self.y[index][1] = y_com
                    
                self.paths.append(scaled_img_path)

            print (str(self.y))
            #print (str(self.paths))

        if self.inmemory:
            self.x = self.__load_images(self.paths)
            self.x = preprocess_input(self.x)

    def __len__(self):
        return math.ceil(len(self.y) / self.batch_size)

    def __getitem__(self, idx):
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        if self.inmemory:
            batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
            return batch_x, batch_y

        batch_x = self.paths[idx * self.batch_size:(idx + 1) * self.batch_size]

        images = self.__load_images(batch_x)
        images = preprocess_input(images)

        return images, batch_y

def create_model(size, alpha, number_of_elements_to_be_output):
    model = MobileNetV2(input_shape=(size, size, 3), include_top=False, alpha=alpha)

    # to freeze layers
    # for layer in model.layers:
    #     layer.trainable = False

    x = model.layers[-1].output
    if size == 96:
        kernel_size_adapt = 3
    elif size == 128:
        kernel_size_adapt = 4
    elif size == 160:
        kernel_size_adapt = 5
    elif size == 192:
        kernel_size_adapt = 6
    elif size == 224:
        kernel_size_adapt = 7
    else:
        kernel_size_adapt = 1
    
    #number_of_elements_to_be_output = 3 # XYZ 3D position of Interest Object
    x = Conv2D(number_of_elements_to_be_output, kernel_size=kernel_size_adapt, name="coords")(x)
    
    from keras.utils import plot_model
    plot_model(model, to_file='./model.png', show_shapes=True)
    
    x = Reshape((number_of_elements_to_be_output,))(x)
    
    if (number_of_elements_to_be_output == 3):
        rospy.logwarn("Created MODEL 3D XYX Output")
    if number_of_elements_to_be_output == 2:
        rospy.logwarn("CREATED MODEL 2D XY Output")

    return Model(inputs=model.input, outputs=x)


def iou(y_true, y_pred):
    # https://keras.io/backend/, $HOME/.keras/keras.json
    # /home/user/.keras/models/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5
    number_of_elements_to_be_output = 2
   
    if number_of_elements_to_be_output == 3:
        d_x = y_true[...,0] - y_pred[...,0]
        d_y = y_true[...,1] - y_pred[...,1]
        d_z = y_true[...,2] - y_pred[...,2]

        # Euclidian distance between the True Point3D and the Predicted
        simple_magn = K.sqrt(K.square(d_x) + K.square(d_y) + K.square(d_z))
        
    if number_of_elements_to_be_output == 2:
        d_x = y_true[...,0] - y_pred[...,0]
        d_y = y_true[...,1] - y_pred[...,1]

        # Euclidian distance between the True Point3D and the Predicted
        simple_magn = K.sqrt(K.square(d_x) + K.square(d_y))    
    
    unitary_tensor = K.ones(K.shape(simple_magn))
    # We consider that 1 meter - MagnitudeBetWeenPoints is the metric. 0 means that the magnitude is
    # bigger than 1 and 1 is the best because it means that we
    result = K.clip( unitary_tensor - simple_magn, 0, 1)
    return result


def train(model, epochs, batch_size, patience, threads, train_csv, validation_csv, models_weight_checkpoints_folder, logs_folder, model_unique_id, load_weight_starting_file=None, number_of_elements_to_be_output=2, initial_learning_rate=0.0001, min_learning_rate = 1e-8):
    train_datagen = DataSequence(train_csv, batch_size,False, number_of_elements_to_be_output)
    validation_datagen = DataSequence(validation_csv, batch_size, False,number_of_elements_to_be_output)

    if load_weight_starting_file:
        rospy.logwarn("Preload Weights, to continue prior training...."+str(load_weight_starting_file))
        model.load_weights(load_weight_starting_file)
    else:
        rospy.logerr("Starting from empty weights.......")

    #model.compile(loss="mean_squared_error", optimizer="adam", metrics=[iou])
    #model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])
    adam_optim = Adam(lr=initial_learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss="mean_squared_error", optimizer=adam_optim, metrics=["accuracy"])

    full_model_file_name = "model-"+ model_unique_id + "-{val_loss:.8f}.h5"
    model_file_path = os.path.join(models_weight_checkpoints_folder, full_model_file_name)
    checkpoint = ModelCheckpoint(model_file_path, monitor="val_loss", verbose=1, save_best_only=True,
                                 save_weights_only=True, mode="auto", period=1)
    
    stop = EarlyStopping(monitor="val_loss", patience=patience*5, mode="auto")
    
    # https://rdrr.io/cran/kerasR/man/ReduceLROnPlateau.html
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=patience, min_lr=min_learning_rate, verbose=1, mode="auto")

    tensorboard_clb = TensorBoard(log_dir=logs_folder, histogram_freq=0,
                                write_graph=True, write_images=True)

    model.summary()

    model.fit_generator(generator=train_datagen,
                        epochs=epochs,
                        validation_data=validation_datagen,
                        callbacks=[checkpoint, reduce_lr, stop, tensorboard_clb],
                        workers=threads,
                        use_multiprocessing=True,
                        shuffle=True,
                        verbose=1)


def main():


    rospy.init_node('generate_database_node', anonymous=True, log_level=rospy.WARN)
    rospy.logwarn("Train Model...START")
    
    if len(sys.argv) < 13:
        rospy.logfatal("usage: train_model.py image_size ALPHA EPOCHS BATCH_SIZE PATIENCE THREADS training_name weight_file_name number_of_elements_to_be_output initial_learning_rate min_learning_rate path_to_database_training_package")
    else:
        image_size = int(sys.argv[1])
        ALPHA = float(sys.argv[2])
        EPOCHS = int(sys.argv[3])
        BATCH_SIZE = int(sys.argv[4])
        PATIENCE = int(sys.argv[5])
        THREADS = int(sys.argv[6])
        training_name = sys.argv[7]
        weight_file_name = sys.argv[8]
        number_of_elements_to_be_output = int(sys.argv[9])
        initial_learning_rate = float(sys.argv[10])
        min_learning_rate = float(sys.argv[11])
        path_to_database_training_package = sys.argv[12]
        
        rospy.logwarn("image_size to training:"+str(image_size))
        rospy.logwarn("ALPHA to training:"+str(ALPHA))
        rospy.logwarn("EPOCHS to training:"+str(EPOCHS))
        rospy.logwarn("BATCH_SIZE to training:"+str(BATCH_SIZE))
        rospy.logwarn("PATIENCE to training:"+str(PATIENCE))
        rospy.logwarn("THREADS to training:"+str(THREADS))
        rospy.logwarn("training_name to training:"+str(training_name))
        rospy.logwarn("weight_file_name to training:"+str(weight_file_name))
        rospy.logwarn("number_of_elements_to_be_output to training:"+str(number_of_elements_to_be_output))
        rospy.logwarn("initial_learning_rate to training:"+str(initial_learning_rate))
        rospy.logwarn("min_learning_rate to training:"+str(min_learning_rate))
        rospy.logwarn("path_to_database_training_package to training:"+str(path_to_database_training_package))
        
        
        rospy.logwarn("Retrieving Paths...")
    
        if path_to_database_training_package == "None":
            rospack = rospkg.RosPack()
            # get the file path for dcnn_training_pkg
            path_to_database_training_package = rospack.get_path('my_dcnn_training_pkg')
            rospy.logwarn("Training Databse Path NOt Found, setting default:"+str(path_to_database_training_package))
        else:
            rospy.logwarn("Training Databse FOUND, setting default:"+str(path_to_database_training_package))
    
        csv_folder_path = os.path.join(path_to_database_training_package, "dataset_gen_csv")
        train_csv_output_file = os.path.join(csv_folder_path, "train.csv")
        validation_csv_output_file = os.path.join(csv_folder_path, "validation.csv")
        
        models_weight_checkpoints_folder = os.path.join(path_to_database_training_package, "model_weight_checkpoints_gen")
        logs_folder = os.path.join(path_to_database_training_package, "logs_gen")
        
        # We clean up the training folders
        if os.path.exists(models_weight_checkpoints_folder):
            shutil.rmtree(models_weight_checkpoints_folder)
        os.makedirs(models_weight_checkpoints_folder)
        print("Created folder=" + str(models_weight_checkpoints_folder))
    
        if os.path.exists(logs_folder):
            shutil.rmtree(logs_folder)
        os.makedirs(logs_folder)
        print("Created folder=" + str(logs_folder))
    
        print ("Start Create Model")
        model = create_model(image_size, ALPHA, number_of_elements_to_be_output)
        
        
        model_unique_id = training_name +"-"+ str(image_size) +"-"+ str(ALPHA) +"-"+ str(EPOCHS) +"-"+ str(BATCH_SIZE) + "-TIME-" + str(time.time())
        
        
        if weight_file_name != "None":
            rospack = rospkg.RosPack()
            # get the file path for rospy_tutorials
            path_to_package = rospack.get_path('my_dcnn_training_pkg')
            backup_models_weight_checkpoints_folder = os.path.join(path_to_package, "bk")
            load_weight_starting_file = os.path.join(backup_models_weight_checkpoints_folder, weight_file_name)
        else:
            rospy.logerr("No load_weight_starting_file...We will start from scratch!")
            load_weight_starting_file = None
        
        
        train(model,
              EPOCHS,
              BATCH_SIZE,
              PATIENCE,
              THREADS,
              train_csv_output_file,
              validation_csv_output_file,
              models_weight_checkpoints_folder,
              logs_folder,
              model_unique_id,
              load_weight_starting_file,
              number_of_elements_to_be_output,
              initial_learning_rate,
              min_learning_rate)
        
        rospy.logwarn("Train Model...END")

if __name__ == "__main__":
    main()

