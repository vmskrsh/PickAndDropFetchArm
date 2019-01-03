#!/usr/bin/env python

import sys
import rospkg
import shutil
import rospy
import os
import datetime
import time
import numpy
from rgb_camera_python3_v2 import RGBCamera
from XMLGenerator import XMLGenerator, XMLObjectTags
from get_model_gazebo_pose import GazeboModel
from std_srvs.srv import Empty, EmptyRequest
from my_randomgazebomanager_pkg.move_fetch_client import MoveFetchClient

try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except ValueError:
    ImportError
#print(sys.path)
import cv2

def dataset_generator(models_of_interest_list, models_to_be_list, full_path_to_dataset_gen='./dataset_gen/', number_of_dataset_elements=10, frequency=1, show_images=False, env_change_settle_time=2.0, move_fetch_arm=False, move_time=2.0, index_img=0):

    # We first wait for the service for RandomEnvironment change to be ready
    rospy.loginfo("Waiting for service /dynamic_world_service to be ready...")
    rospy.wait_for_service('/dynamic_world_service')
    rospy.loginfo("Service /dynamic_world_service READY")
    dynamic_world_service_call = rospy.ServiceProxy('/dynamic_world_service', Empty)
    change_env_request = EmptyRequest()

    dynamic_world_service_call(change_env_request)

    # Init the FetchClient to move the robot arm
    move_fetch_client_object = MoveFetchClient()
    
    x_range = [0.35,0.35]
    y_range = [0.1,0.5]
    z_range = [0.64+0.3,0.64+0.3]
    
    orientation_XYZW = [-0.707, 0.000, 0.707, 0.001]
    
    #Init Pose
    move_fetch_client_object.go_to_safe_arm_pose()
    
    # Set optimum torso height
    height = 0.2
    move_fetch_client_object.move_torso(height)
        

    # Period show the images in miliseconds
    period = int((1 / frequency) * 1000)

    # We first check if the path given exists, if not its generated the directory
    if os.path.exists(full_path_to_dataset_gen):
        shutil.rmtree(full_path_to_dataset_gen)

    os.makedirs(full_path_to_dataset_gen)
    rospy.logfatal("Created folder=" + str(full_path_to_dataset_gen))

    
    # Init camera RGB object
    camera_topic = "/dynamic_objects/camera/raw_image"
    camera_info_topic = "/dynamic_objects/camera/info_image"
    camera_obj = RGBCamera(camera_topic,camera_info_topic)

    # This are the models that we will generate information about.
    gz_model_obj = GazeboModel(models_to_be_list)

    # XML generator :
    xml_generator_obj = XMLGenerator(path_out=full_path_to_dataset_gen)

    # index to name the pictures :
    now = datetime.datetime.now()
    str_date = now.strftime("%Y_%m_%d_%H_%M_")

    # create the images folder :
    path_img_dir = os.path.join(full_path_to_dataset_gen, 'images/')

    if not os.path.exists(path_img_dir):
        os.makedirs(path_img_dir)


    # We generate as many dataset elements as it was indicated.
    for i in range(number_of_dataset_elements):

        # We Randomise Environment
        dynamic_world_service_call(change_env_request)
        rospy.logwarn("Waiting for new env to settle...")
        rospy.sleep(env_change_settle_time)
        rospy.logwarn("Waiting for new env to settle")

        # We move Fetch Arm to random pose
        if move_fetch_arm:
            rospy.logwarn("Moving to Random Pose TCP")
            move_fetch_client_object.move_tcp_to_random_pose(x_range,y_range,z_range,orientation_XYZW)
            rospy.sleep(move_time)
            rospy.logwarn("Moving to Random Pose TCP DONE")
        else:
            move_time = 0.0


        estimated_time_of_completion = ((number_of_dataset_elements - i) * (env_change_settle_time + (period/1000.0) + move_time)) / 60.0
        rospy.logwarn("Time Estimated of completion [min]==>"+str(estimated_time_of_completion))

        # We take picture
        rospy.logerr("Takeing PICTURE")
        #image = camera_obj.getImage()
        image = camera_obj.get_latest_image()
        rospy.logerr("Takeing PICTURE...DONE")
        #cam_info = camera_obj.get_camera_info()
        cam_info = camera_obj.get_camera_info()

        if image is None:
            rospy.logwarn("Image value was none")
            image = numpy.zeros((cam_info.height, cam_info.width))

        filename = str_date + str(index_img)

        # We retrieve objects position
        pose_models_oi_dict = {}
        for model_name in models_of_interest_list:
            rospy.logdebug("model_name==>" + str(model_name))
            pose_now = gz_model_obj.get_model_pose(model_name)
            pose_models_oi_dict[model_name] = pose_now

        rospy.logdebug("ObjectsOfInterest poses==>" + str(pose_models_oi_dict))
        #raw_input("Press After Validate Position in simulation....")
        # We create the XML list tags for each Model of Interest captured before
        listobj = []
        for model_name, model_3dpose in pose_models_oi_dict.items(): 

            x_com = model_3dpose.position.x
            y_com = model_3dpose.position.y
            z_com = model_3dpose.position.z
            quat_x = model_3dpose.orientation.x
            quat_y = model_3dpose.orientation.y
            quat_z = model_3dpose.orientation.z
            quat_w = model_3dpose.orientation.w

            pose3d = [x_com, y_com, z_com, quat_x, quat_y, quat_z, quat_w]
            listobj.append(XMLObjectTags(name=model_name, pose3d=pose3d))


        image_format_extension = ".png"
        xml_generator_obj.generate(object_tags=listobj,
                                   filename=filename,
                                   extension_file=image_format_extension,
                                   camera_width=cam_info.width,
                                   camera_height=cam_info.height)

        index_img += 1

        file_name_ext = filename + image_format_extension
        path_img_file = os.path.join(path_img_dir, file_name_ext)

        cv2.imwrite(path_img_file, image)

        if show_images:
            cv2.imshow(filename, image)
            cv2.waitKey(period)
            cv2.destroyAllWindows()

def process_names_str_list(names_trs_list):
    """
    It gets a string with names separated by commads and outputs a list of those names
    :param names_trs_list:
    :return:
    """
    return names_trs_list.split(",")

if __name__ == "__main__":

    rospy.init_node('create_training_material_node_new', anonymous=True, log_level=rospy.INFO)

    if len(sys.argv) < 10:
        print("usage: create_training_material.py models_of_interest_list models_to_be_list number_of_elements env_change_settle_time show_images move_fetch_arm move_time init_index_img path_to_dataset_gen")
    else:
        models_of_interest_list = process_names_str_list(sys.argv[1])
        models_to_be_list = process_names_str_list(sys.argv[2])
        number_of_elements = int(sys.argv[3])
        env_change_settle_time = float(sys.argv[4])
        show_images = bool(sys.argv[5] == "True")
        move_fetch_arm = bool(sys.argv[6] == "True")
        move_time = float(sys.argv[7])
        init_index_img = int(sys.argv[8])
        path_to_dataset_gen = sys.argv[9]
        
        
        
        # Models that we want to generate traiing data of
        print (models_of_interest_list)
        # Models that we want the system to check they are in the simulation and wait until they are
        print (models_to_be_list)
        # Number of Images and data XML pairs to create
        print (number_of_elements)
        print (env_change_settle_time)
        print (show_images)
        print (move_fetch_arm)
        print (move_time)
        print (init_index_img)
        print (path_to_dataset_gen)


        rospy.logdebug("Program test START")

        # get an instance of RosPack with the default search paths
        rospack = rospkg.RosPack()
        # get the file path for rospy_tutorials
        if path_to_dataset_gen == "None":
            path_to_dataset_gen = rospack.get_path('my_randomgazebomanager_pkg')
            rospy.logwarn("NO path to dataset gen, setting current package:"+str(path_to_dataset_gen))
        else:
            rospy.logwarn("Found path to dataset gen:"+str(path_to_dataset_gen))
        
        full_path_to_dataset_gen = os.path.join(path_to_dataset_gen, "dataset_gen")
        
        

        
        dataset_generator(models_of_interest_list=models_of_interest_list,
                          models_to_be_list=models_to_be_list,
                          full_path_to_dataset_gen=full_path_to_dataset_gen,
                          number_of_dataset_elements=number_of_elements,
                          frequency=1,
                          show_images=show_images,
                          env_change_settle_time=env_change_settle_time,
                          move_fetch_arm = move_fetch_arm,
                          move_time = move_time,
                          index_img= init_index_img)
        

        rospy.logdebug("Program test END")
    
