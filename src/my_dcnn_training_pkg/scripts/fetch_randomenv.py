#!/usr/bin/env python
import rospy
import time
import random
import gym
import math
import rospkg
import os
import copy
from geometry_msgs.msg import Pose
from my_randomgazebomanager_pkg.rviz_markers import MarkerBasics
#from my_randomgazebomanager_pkg.rviz_markers import PickObjectMenu
from my_randomgazebomanager_pkg.move_fetch_client import MoveFetchClient
from sensor_msgs.msg import Image
from get_model_gazebo_pose import GazeboModel
from std_srvs.srv import Empty, EmptyRequest

# Dont put anything ros related after this import because it removes ROS from imports to 
# import the cv2 installed and not the ROS version
from rgb_camera_python3 import RGBCamera
import sys
#print(sys.path)
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except ValueError:
    ImportError
#print(sys.path)
import cv2
import numpy as np

from train_model import create_model
from keras.applications.mobilenetv2 import preprocess_input

import xml.etree.ElementTree as ET


class FetchDCNN():
    def __init__(self, environment_name, weight_file_name, image_size, ALPHA, grasp_activated, number_of_elements_to_be_output):
        
        
        self._image_size = image_size
        self._number_of_elements_to_be_output = number_of_elements_to_be_output
        
        self.init_rviz_markers()
        
        # Init service to move fetch, this is because moveit doenst work in python 3
        self._grasp_activated = grasp_activated
        if self._grasp_activated == True:
            self.fetch_move_client = MoveFetchClient()
            self.fetch_move_client.go_to_safe_arm_pose()
        
        # Init camera RGB object
        self.rgb_camera_object = RGBCamera("/dynamic_objects/camera/raw_image")
        
        # This are the models that we will generate information about.
        self.model_to_track_name = "demo_spam1"
        self.table_to_track_name = "demo_table1"
        
        model_to_track_list = [self.model_to_track_name, self.table_to_track_name]
        self.gz_model_obj = GazeboModel(model_to_track_list)
        
        # We start the model in Keras
        self.model = create_model(self._image_size, ALPHA, self._number_of_elements_to_be_output)
        
        rospack = rospkg.RosPack()
        # get the file path for rospy_tutorials
        path_to_package = rospack.get_path('my_dcnn_training_pkg')
        models_weight_checkpoints_folder = os.path.join(path_to_package, "bk")
        model_file_path = os.path.join(models_weight_checkpoints_folder, weight_file_name)
        
        print (model_file_path)
        
        self.model.load_weights(model_file_path)
        
        self.testing_unscaled_img_folder = os.path.join(path_to_package, "testing/dataset_gen/images")
        self.testing_unscaled_anotations_folder = os.path.join(path_to_package, "testing/dataset_gen_annotations")
        
        
        # We reset the environent to a random state
        print("Starting Service to Reset World Randomly....")
        self.dynamic_world_service_call = rospy.ServiceProxy('/dynamic_world_service', Empty)
        self.change_env_request = EmptyRequest()
        self.dynamic_world_service_call(self.change_env_request)
        print("Starting Service to Reset World Randomly....DONE")
        
        

    def init_rviz_markers(self):
        """
        We initialise the markers used to visualise the predictions vs the 
        real data.
        """
        # We initialise the Markers for Center Of Mass estimation and the Object real position
        marker_type = "cube"
        namespace = "demo_table"
        mesh_package_path = ""
        self.demo_table_position = MarkerBasics(type=marker_type,
                                          namespace=namespace,
                                          index=0,
                                          red=0.0,
                                          green=0.0,
                                          blue=1.0,
                                          alfa=1.0,
                                          scale=[0.6,0.6,0.6],
                                          mesh_package_path=mesh_package_path)
    
    
        marker_type = "mesh"
        namespace = "spam_real_position"
        mesh_package_path = "dynamic_objects/models/demo_spam/meshes/nontextured_fixed.stl"
    
        self.spam_real_position = MarkerBasics(type=marker_type,
                                            namespace=namespace,
                                            index=0,
                                            red=1.0,
                                            green=0.0,
                                            blue=0.0,
                                            alfa=1.0,
                                            scale=[1.0, 1.0, 1.0],
                                            mesh_package_path=mesh_package_path)
    
        marker_type = "sphere"
        mesh_package_path = ""
        namespace = "com_prediction"
        self.com_prediction_marker = MarkerBasics(type=marker_type,
                                            namespace=namespace,
                                            index=0,
                                            red=0.0,
                                            green=1.0,
                                            blue=0.0,
                                            alfa=1.0,
                                            scale=[0.1, 0.1, 0.1],
                                            mesh_package_path=mesh_package_path)
                                            
        #self.pick_object_menu = PickObjectMenu()

    
    def publish_markers_new_data(self,pred, reality, reality_table):
        
        spam_real_pose = Pose()
        spam_real_pose.position.x = reality[0]
        spam_real_pose.position.y = reality[1]
        spam_real_pose.position.z = reality[2]

        com_prediction_pose = Pose()
        com_prediction_pose.position.x = pred[0]
        com_prediction_pose.position.y = pred[1]
        com_prediction_pose.position.z = pred[2]

        demo_table_pose = Pose()
        demo_table_pose.position.x = reality_table[0]
        demo_table_pose.position.y = reality_table[1]
        demo_table_pose.position.z = reality_table[2]

        self.spam_real_position.publish_marker(spam_real_pose)
        self.com_prediction_marker.publish_marker(com_prediction_pose)
        self.demo_table_position.publish_marker(demo_table_pose)
    
    """ 
    def publish_new_grasp_state(self, grasp_state):
        
        if (grasp_state == "grasp_success"):
            print(">>>>grasp_success....")
            self.pick_object_menu.update_menu(index=1)
        elif (grasp_state == "grasp_fail"):
            print(">>>>grasp_fail....")
            self.pick_object_menu.update_menu(index=2)
        else:
            print(">>>>nothing....")
            self.pick_object_menu.update_menu(index=0)
    """        
        
    
    
    def predict_image(self,model,cv2_img=None, path=None):
        
        if cv2_img.all() != None:
            im = cv2_img
        else:
            if path != None:
                im = cv2.imread(path)
            else:
                print ("Predict Image had no path image or image CV2")
                return None
        
        if im.shape[0] != self._image_size:
            im = cv2.resize(im, (self._image_size, self._image_size))
            """
            self.rgb_camera_object.display_image(   image_display=im,
                                                    life_time_ms=50,
                                                    name="ResizedCAM"
                                                )
            """
    
        image = np.array(im, dtype='f')
        image = preprocess_input(image)
        
        
        self.rgb_camera_object.display_image(   image_display=image,
                                                    life_time_ms=50,
                                                    name="ImagePredict"
                                                )
    
        prediction = model.predict(x=np.array([image]))[0]
    
        return prediction
    
    def show_image(self,path, wait_time=500):
        image = cv2.imread(path)
        cv2.imshow("image", image)
        print ("Waiting " + str(wait_time) + "ms...")
        cv2.waitKey(wait_time)
        print ("Waiting "+str(wait_time)+"ms...END")
        cv2.destroyAllWindows()
    
    def get_xyz_from_xml(self,path_xml_file):
    
        tree = ET.parse(path_xml_file)
    
        x_com = float(tree.findtext("./object/pose3d/x_com"))
        y_com = float(tree.findtext("./object/pose3d/y_com"))
        z_com = float(tree.findtext("./object/pose3d/z_com"))
    
        return [x_com, y_com, z_com]
        
    def get_xyz_from_world(self, model_name):
        """
        Retrieves the position of an object from the world
        """
        pose_now = self.gz_model_obj.get_model_pose(model_name)
        
        XYZ = [pose_now.position.x,pose_now.position.y,pose_now.position.z]
        
        return XYZ
    
    
    def start_prediction_test(self):
        
        print("\nTrying out unscaled image")
        for k in os.listdir(self.testing_unscaled_img_folder):
            print ("Name File==>" + str(k))
            
            rospy.logwarn("We Reset Simulation")
            
            init_joints_config = [0.0] * 7
            self.move_joints(init_joints_config)
            
            if "png" in k:
                
                img_path = os.path.join(self.testing_unscaled_img_folder, k)
                pred = self.predict_image(img_path, self.model)
    
                base_name_file = os.path.splitext(k)[0]
                annotation_file = base_name_file + ".xml"
                img_anotation_path = os.path.join(self.testing_unscaled_anotations_folder, annotation_file)
                reality = self.get_xyz_from_xml(img_anotation_path)
    
                print ("Class Prediction=>" + str(pred))
                print ("Class Reality File Image=>" + str(reality))
    
                self.publish_markers_new_data(pred, reality)
    
                self.show_image(img_path, wait_time=5000)
                
                if self._grasp_activated == True:
                    self.start_grasp_sequence(pred)
                
    
        rospy.loginfo("Start Prediction DONE...")
        
    
    def start_camera_rgb_prediction(self, number_of_tests=1, camera_period = 5.0, wait_reset_period=3.0, go_dustbin=False):
        
        for i in range(number_of_tests):
            print ("Number of Image=>" + str(i))
            
            self.dynamic_world_service_call(self.change_env_request)
            print ("Waiting for Reset Env to settle=>")
            rospy.sleep(wait_reset_period)
            print ("Waiting for Reset Env to settle...DONE")
            
            cv2_img = self.rgb_camera_object.get_latest_image()
            pred = self.predict_image(self.model,cv2_img)
            
            # Add Z Axis value
            z_value = 0.6 + 0.042
            pred_mod = [pred[0],pred[1],z_value]
            
    
            reality = self.get_xyz_from_world(self.model_to_track_name)
            reality_table = self.get_xyz_from_world(self.table_to_track_name)
            
            print ("Class Prediction=>" + str(pred))
            print ("Class Prediction Corrected=>" + str(pred_mod))
            print ("Class Reality SIMULATION=>" + str(reality))
            print ("Class RealityTabel SIMULATION=>" + str(reality_table))
            self.publish_markers_new_data(pred_mod, reality, reality_table)
    
            
            #self.rgb_camera_object.display_latest_image(int(camera_period*1000))
            
            if self._grasp_activated == True:
                if go_dustbin:
                    self.start_grasp_sequence_leave_dustbin(pred_mod)
                else:
                    self.start_grasp_sequence(pred_mod)
        
        rospy.loginfo("Start Prediction DONE...")
    
        
    def start_camera_rgb_prediction_continuous(self, image_freq= 20.0):
        """
        It continuously make predictions
        """
        # We reset the world Once
        self.dynamic_world_service_call(self.change_env_request)
        print ("Waiting for Reset Env to settle=>")
        wait_reset_period = 3.0
        rospy.sleep(wait_reset_period)
        print ("Waiting for Reset Env to settle...DONE")
        
        rate = rospy.Rate(image_freq)
        life_time_ms = int((1.0/ image_freq)*1000)
        while not rospy.is_shutdown():
            
            cv2_img = self.rgb_camera_object.get_latest_image()
            
            self.rgb_camera_object.display_image(   image_display=cv2_img,
                                                    life_time_ms=life_time_ms
                                                )
                                                
            pred = self.predict_image(self.model,cv2_img)
            # Add Z Axis value
            z_value = 0.6 + 0.042
            pred_mod = [pred[0],pred[1],z_value]
    
            reality = self.get_xyz_from_world(self.model_to_track_name)
            reality_table = self.get_xyz_from_world(self.table_to_track_name)
            
    
            print ("Class Prediction=>" + str(pred))
            print ("Class PredictionMod=>" + str(pred_mod))
            print ("Class Reality SIMULATION=>" + str(reality))
            self.publish_markers_new_data(pred_mod, reality, reality_table)
            rate.sleep()
                
        rospy.loginfo("Start Prediction DONE...")
        
    def move_endeffector(self, position_XYZ):
        """
        Move the EndEffector to the position given, orientation set
        """
        rospy.logwarn("START Move ENd Effector")
        
        orientation_XYZW = [-0.707, 0.000, 0.707, 0.001]
        
        result = self.fetch_move_client.move_endeffector(position_XYZ, orientation_XYZW)
        
        rospy.logwarn("END Move End Effector, result="+str(result))
        
        return result
        
    def move_joints(self, joints_config):
        """
        Move the joints to the specified configuration.
        """
        rospy.logwarn("START Move Joints")

        result = self.fetch_move_client.move_joints(joints_config)
        
        rospy.logwarn("END Move Joints, result="+str(result))
        
        return result
    
    
    def open_close_gripper(self, open_or_close):
        """
        Choose if you want to open or close
        """
        if open_or_close == "open":
            # Open
            gripper_x = 0.4
            max_effort = 10.0
        else:
            # Close
            gripper_x = 0.05
            max_effort = 20.0
        
        self.fetch_move_client.move_gripper(gripper_x, max_effort)
    
    
    def check_if_object_grasped(self, tcp_xyz, wait_time=2.0, max_delta=0.2):
        """
        It checks if the object to graps has been lifted from the table
        """
        
        model_to_track_reality = self.get_xyz_from_world(self.model_to_track_name)
        
        print ("model_to_track_reality=>" + str(model_to_track_reality))
        print ("tcp_xyz=>" + str(tcp_xyz))
        
        z_model = model_to_track_reality[2]
        z_tcp = tcp_xyz[2]
        
        delta = z_tcp - z_model
        
        rospy.logwarn("delta=="+str(delta)+" <= "+str(max_delta))
        
        if delta <= max_delta:
            grasp_state = "grasp_success"
        else:
            grasp_state = "grasp_fail"

        #self.publish_new_grasp_state(grasp_state)
        
        rospy.sleep(wait_time)
        
        return grasp_state == "grasp_success"
        
    
    def start_grasp_sequence(self, predicted_position_XYZ):
        
        #Init Pose
        self.fetch_move_client.go_to_safe_arm_pose()
        
        # Set optimum torso height
        height = 0.2
        result = self.fetch_move_client.move_torso(height)
        
        height_delta = 0.3
        position_XYZ = [predicted_position_XYZ[0],
                        predicted_position_XYZ[1],
                        predicted_position_XYZ[2] + height_delta ]
        orientation_XYZW = [-0.707, 0.000, 0.707, 0.001]
        self.fetch_move_client.move_endeffector(position_XYZ, orientation_XYZW)
        
        # Test Gripper
        self.open_close_gripper(open_or_close="open")
        self.open_close_gripper(open_or_close="close")
        self.open_close_gripper(open_or_close="open")
        
        # Lower ARM
        height_delta = 0.19
        position_XYZ = [predicted_position_XYZ[0],
                        predicted_position_XYZ[1],
                        predicted_position_XYZ[2] + height_delta ]
        orientation_XYZW = [-0.707, 0.000, 0.707, 0.001]
        self.fetch_move_client.move_endeffector(position_XYZ, orientation_XYZW)
        
        # Close Gripper to grasp
        self.open_close_gripper(open_or_close="close")
        
        # Go Up with the object
        height_delta = 0.3
        position_XYZ = [predicted_position_XYZ[0],
                        predicted_position_XYZ[1],
                        predicted_position_XYZ[2] + height_delta ]
        orientation_XYZW = [-0.707, 0.000, 0.707, 0.001]
        self.fetch_move_client.move_endeffector(position_XYZ, orientation_XYZW)
        
        
        # We check if we grapsed the object
        self.check_if_object_grasped(tcp_xyz=position_XYZ)
        
        # Go down with the object
        height_delta = 0.19
        position_XYZ = [predicted_position_XYZ[0],
                        predicted_position_XYZ[1],
                        predicted_position_XYZ[2] + height_delta ]
        orientation_XYZW = [-0.707, 0.000, 0.707, 0.001]
        self.fetch_move_client.move_endeffector(position_XYZ, orientation_XYZW)
        
        # Release the object
        self.open_close_gripper(open_or_close="open")
        
        # We reset the Graps Marker
        #self.publish_new_grasp_state(grasp_state="nothing")
        
        # Up again
        height_delta = 0.3
        position_XYZ = [predicted_position_XYZ[0],
                        predicted_position_XYZ[1],
                        predicted_position_XYZ[2] + height_delta ]
        orientation_XYZW = [-0.707, 0.000, 0.707, 0.001]
        self.fetch_move_client.move_endeffector(position_XYZ, orientation_XYZW)
        
        # Go to Init Safe pose
        self.fetch_move_client.go_to_safe_arm_pose()
        
    def start_grasp_sequence_leave_dustbin(self, predicted_position_XYZ):
        
        #Init Pose
        self.fetch_move_client.go_to_safe_arm_pose()
        
        # Set optimum torso height
        height = 0.2
        result = self.fetch_move_client.move_torso(height)
        
        height_delta = 0.3
        position_XYZ = [predicted_position_XYZ[0],
                        predicted_position_XYZ[1],
                        predicted_position_XYZ[2] + height_delta ]
        orientation_XYZW = [-0.707, 0.000, 0.707, 0.001]
        self.fetch_move_client.move_endeffector(position_XYZ, orientation_XYZW)
        
        # Test Gripper
        self.open_close_gripper(open_or_close="open")
        self.open_close_gripper(open_or_close="close")
        self.open_close_gripper(open_or_close="open")
        
        # Lower ARM
        height_delta = 0.19
        position_XYZ = [predicted_position_XYZ[0],
                        predicted_position_XYZ[1],
                        predicted_position_XYZ[2] + height_delta ]
        orientation_XYZW = [-0.707, 0.000, 0.707, 0.001]
        self.fetch_move_client.move_endeffector(position_XYZ, orientation_XYZW)
        
        # Close Gripper to grasp
        self.open_close_gripper(open_or_close="close")
        
        # Go Up with the object
        height_delta = 0.3
        position_XYZ = [predicted_position_XYZ[0],
                        predicted_position_XYZ[1],
                        predicted_position_XYZ[2] + height_delta ]
        orientation_XYZW = [-0.707, 0.000, 0.707, 0.001]
        self.fetch_move_client.move_endeffector(position_XYZ, orientation_XYZW)
        
        
        # We check if we grapsed the object
        grasp_success = self.check_if_object_grasped(tcp_xyz=position_XYZ)
        
        if grasp_success:
            # The Graps Succeeeded We place object in DustBin
            # Move to dustbin Pose
            self.fetch_move_client.go_to_dustbin_arm_pose()
            
        # Release the object
        self.open_close_gripper(open_or_close="open")
        
        # We reset the Graps Marker
        #self.publish_new_grasp_state(grasp_state="nothing")
        
        # Go to Init Safe pose
        self.fetch_move_client.go_to_safe_arm_pose()
        
        
        
if __name__ == '__main__':
    rospy.init_node('fetch_randomenv_node', anonymous=True, log_level=rospy.INFO)
    
    
    if len(sys.argv) < 10:
        rospy.logfatal("usage: fetch_randomenv.py environment_name weight_file_name grasp_activated number_of_tests camera_period image_size ALPHA number_of_elements_to_be_output go_dustbin")
    else:
        
        rospy.logwarn(str(sys.argv))
        
        environment_name = sys.argv[1]
        weight_file_name = sys.argv[2]
        grasp_activated = bool(sys.argv[3] == "True")
        number_of_tests = int(sys.argv[4])
        camera_period = float(sys.argv[5])
        image_size = int(sys.argv[6])
        ALPHA = float(sys.argv[7])
        number_of_elements_to_be_output = int(sys.argv[8])
        go_dustbin = bool(sys.argv[9] == "True")
        
        rospy.logwarn("environment_name:"+str(environment_name))
        rospy.logwarn("weight_file_name:"+str(weight_file_name))
        rospy.logwarn("grasp_activated:"+str(grasp_activated))
        rospy.logwarn("number_of_tests:"+str(number_of_tests))
        rospy.logwarn("camera_period:"+str(camera_period))
        rospy.logwarn("image_size:"+str(image_size))
        rospy.logwarn("ALPHA:"+str(ALPHA))
        rospy.logwarn("number_of_elements_to_be_output:"+str(number_of_elements_to_be_output))
    
        agent = FetchDCNN(  environment_name,weight_file_name,
                            image_size,
                            ALPHA,
                            grasp_activated,
                            number_of_elements_to_be_output)
                                    
        agent.start_camera_rgb_prediction(number_of_tests=number_of_tests, camera_period = camera_period, wait_reset_period=3.0, go_dustbin=go_dustbin)
        #agent.start_camera_rgb_prediction_continuous()

