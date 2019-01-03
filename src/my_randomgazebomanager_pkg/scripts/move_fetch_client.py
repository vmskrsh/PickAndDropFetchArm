#!/usr/bin/env python
import os
import rospy
import sys
from geometry_msgs.msg import Pose
from my_randomgazebomanager_pkg.srv import FetchMove, FetchMoveResponse, FetchMoveRequest
import copy
from my_randomgazebomanager_pkg.rviz_markers import MarkerBasicsArray
from my_randomgazebomanager_pkg.rviz_markers import MarkerBasics
import yaml
import rospkg
import random

class MoveFetchClient(object):
    
    def __init__(self):
        rospy.logwarn("Waiting for service /fetch_move_service")
        rospy.wait_for_service('/fetch_move_service')
        self.fetch_move_service = rospy.ServiceProxy('/fetch_move_service', FetchMove)
        self.fetch_move_request = FetchMoveRequest()
        

    def move_endeffector(self, position_XYZ, orientation_XYZW):
        """
        Move the arm through the EndEffector to the position given, orientation set
        """
        rospy.logwarn("START Move ENd Effector")
        
        pose_requested = Pose()
        pose_requested.position.x = position_XYZ[0]
        pose_requested.position.y = position_XYZ[1]
        pose_requested.position.z = position_XYZ[2]
        
        pose_requested.orientation.x = orientation_XYZW[0]
        pose_requested.orientation.y = orientation_XYZW[1]
        pose_requested.orientation.z = orientation_XYZW[2]
        pose_requested.orientation.w = orientation_XYZW[3]
        
        self.fetch_move_request.pose = pose_requested
        
        self.fetch_move_request.joints_array = [0.0] * 7
        
        self.fetch_move_request.movement_type.data = "TCP"
        
        result =  self.fetch_move_service(self.fetch_move_request)
        
        rospy.logwarn("END Move End Effector, result="+str(result))
        
        return result
        
    def move_joints(self, joints_config):
        """
        Move the arm joints to the specified configuration.
        """
        rospy.logwarn("START Move Joints")
        
        pose_requested = Pose()
        self.fetch_move_request.pose = pose_requested
        
        self.fetch_move_request.joints_array = joints_config
        
        self.fetch_move_request.movement_type.data = "JOINTS"
        
        result =  self.fetch_move_service(self.fetch_move_request)
        
        rospy.logwarn("END Move Joints, result="+str(result))
        
        return result
        
    def move_torso(self,torso_height):
        """
        Move torso
        """
        rospy.logwarn("START Move Torso")
        
        pose_requested = Pose()
        self.fetch_move_request.pose = pose_requested
        
        self.fetch_move_request.joints_array = [torso_height]
        
        self.fetch_move_request.movement_type.data = "TORSO"
        
        result =  self.fetch_move_service(self.fetch_move_request)
        
        return result
        
    def move_gripper(self,gripper_x, max_effort):
        
        rospy.logwarn("START Move Gripper")
        
        pose_requested = Pose()
        self.fetch_move_request.pose = pose_requested
        
        self.fetch_move_request.joints_array = [gripper_x,max_effort]
        
        self.fetch_move_request.movement_type.data = "GRIPPER"
        
        result =  self.fetch_move_service(self.fetch_move_request)
        
        return result
        
    def go_to_safe_arm_pose(self):
        """
        It takes the robot arm to a position that doesnt obstrude in any way the recognition
        """
        # Go to Init Safe pose
        print ("Going to Sage Grasp Init Pose...")
        init_joints_config  = [-1.57, -0.9, 0, 0.9, 0.0, 1.57, 0.0]
        self.move_joints(init_joints_config)
        
    def go_to_dustbin_arm_pose(self):
        """
        It takes the robot arm to a position where to dump object to dustbin
        """
        # Go to Init Safe pose
        print ("Going to Sage Grasp Init Pose...")
        init_joints_config  = [-1.57, -0.6, 0, 0.6, 0.0, 1.57, 0.0]
        self.move_joints(init_joints_config)
        
    def move_tcp_to_random_pose(self,x_range,y_range,z_range, orientation_XYZW):
        """
        Moves TCP to a random pose inside the range given
        """
        x_random = random.uniform(x_range[0], x_range[1])
        y_random = random.uniform(y_range[0], y_range[1])
        z_random = random.uniform(z_range[0], z_range[1])
        position_XYZ = [x_random,y_random,z_random]
        
        self.move_endeffector(position_XYZ, orientation_XYZW)


def testing_positions():
    
    fetch_move_client = MoveFetchClient()
    
    position_XYZ = [0.598,0.005,0.9]
    orientation_XYZW = [0, 0, 0, 1]
    
    fetch_move_client.move_endeffector(position_XYZ, orientation_XYZW)
    
    position_XYZ = [0.668, 0.658, 0.757]
    orientation_XYZW = [-0.500, 0.500, 0.500, 0.500]
    
    fetch_move_client.move_endeffector(position_XYZ, orientation_XYZW)
    
    
    #arm_joint_positions  = [0.0, -0.9, 0, 0.9, 0.0, 1.57, 0.0]
    #fetch_move_client.move_joints(arm_joint_positions)
    
    """
    rosrun tf tf_echo world gripper_link
    
    Translation: [0.010, 0.001, 0.757]
- Rotation: in Quaternion [-0.707, 0.000, 0.707, 0.001]
            in RPY (radian) [-0.220, 1.570, 2.920]
            in RPY (degree) [-12.627, 89.969, 167.287]
            
            
    Location example detection World frame
    [-0.01606202  0.04932804  0.6449038 ]
    """
    
    #position_XYZ = [0.010, 0.001, 0.9]
    position_XYZ = [-0.01606202,0.04932804, 0.6449038 + 0.3 ]
    orientation_XYZW = [-0.707, 0.000, 0.707, 0.001]
    
    fetch_move_client.move_endeffector(position_XYZ, orientation_XYZW)
    
    
def test_range_world_movements():
    """
    Here we test the range above the table that fetch can move the arm.
    """
    marker_size = 0.1
    marker_array_obj = MarkerBasicsArray(namespace="fetch_test_range", marker_size=marker_size)
    
    marker_type = "cube"
    namespace = "demo_table"
    mesh_package_path = ""
    demo_table_position = MarkerBasics(type=marker_type,
                                      namespace=namespace,
                                      index=0,
                                      red=0.0,
                                      green=0.0,
                                      blue=1.0,
                                      alfa=1.0,
                                      scale=[0.6,0.6,0.6],
                                      mesh_package_path=mesh_package_path)
    
    demo_table_pose = Pose()
    demo_table_pose.position.x = 0.3
    demo_table_pose.position.y = 0.3
    demo_table_pose.position.z = 0.3
    demo_table_position.publish_marker(demo_table_pose)
    
    fetch_move_client = MoveFetchClient()
    
    base_table_height = 0.6
    height_delta = 0.25
    position_XYZ = [0.3, 0.3, base_table_height + height_delta ]
    orientation_XYZW = [-0.707, 0.000, 0.707, 0.001]
    
    fetch_move_client.move_endeffector(position_XYZ, orientation_XYZW)
    
    # Range of accepcted positions XYZ in world frame, Ok range is if they were ok or not
    range_list = []
    range_list_dicts = []
    ok_range_list = []
    
    delta_increment = marker_size
    max_value = 0.6
    
    min_val_X = 0.0
    position_XYZ[0] = min_val_X
    
    min_val_Y = 0.0
    position_XYZ[1] = min_val_Y
    
    while position_XYZ[0] <= max_value:
        
        position_XYZ[1] = min_val_Y
        while position_XYZ[1] <= max_value:
            rospy.logwarn(str(position_XYZ))
            
            result = fetch_move_client.move_endeffector(position_XYZ, orientation_XYZW)
            ok_range_list.append(result.success)
            range_list.append(copy.deepcopy(position_XYZ))
            
            pose_dict = dict(zip(["OK","X","Y","Z"], [ok_range_list[-1],position_XYZ[0],position_XYZ[1],position_XYZ[2]]))
            range_list_dicts.append(pose_dict)

            marker_array_obj.publish_pose_array(range_list,ok_range_list)
            demo_table_position.publish_marker(demo_table_pose)
            position_XYZ[1] += delta_increment

        rospy.logerr("Max Y Value==>"+str(position_XYZ[1]))
        position_XYZ[0] += delta_increment

    rospy.logerr("Max X Value==>"+str(position_XYZ[0]))
    
    print (str(range_list_dicts))
    
    rospack = rospkg.RosPack()
    path_to_package = rospack.get_path('dcnn_training_pkg')
    pose_files_dir = os.path.join(path_to_package, "fetch_limits_files")
    
    if not os.path.exists(pose_files_dir):
        os.makedirs(pose_files_dir)
    
    pose_file_name = "fetch_move_limits.yaml"
    file_to_store = os.path.join(pose_files_dir, pose_file_name)
    
    store_dict_in_yaml(file_to_store, range_list_dicts)


def store_dict_in_yaml(file_to_store, dictionary_to_save):
    
    with open(file_to_store, 'w') as outfile:
        yaml.dump(dictionary_to_save, outfile, default_flow_style=False)
    rospy.logdebug("Data Saved in=>"+str(file_to_store))

          
def test_torso_movements():
    """
    Here we test the range above the table that fetch can move the arm.
    """
    fetch_move_client = MoveFetchClient()
    
    height = 0.2
    result = fetch_move_client.move_torso(height)


def test_grasp():
    
    
    fetch_move_client = MoveFetchClient()
    
    height = 0.2
    result = fetch_move_client.move_torso(height)
    
    height_delta = 0.3
    position_XYZ = [0.0, 0.0, 0.6449038 + height_delta ]
    orientation_XYZW = [-0.707, 0.000, 0.707, 0.001]
    
    fetch_move_client.move_endeffector(position_XYZ, orientation_XYZW)
    
    
    gripper_x = 0.1
    max_effort = 10.0
    fetch_move_client.move_gripper(gripper_x, max_effort)
    
    gripper_x = 0.0
    max_effort = 10.0
    fetch_move_client.move_gripper(gripper_x, max_effort)
    
    gripper_x = 0.2
    max_effort = 20.0
    fetch_move_client.move_gripper(gripper_x, max_effort)
    
    
    height_delta = 0.19
    position_XYZ = [0.0, 0.0, 0.6449038 + height_delta ]
    orientation_XYZW = [-0.707, 0.000, 0.707, 0.001]
    
    fetch_move_client.move_endeffector(position_XYZ, orientation_XYZW)
    
    gripper_x = 0.05
    max_effort = 20.0
    fetch_move_client.move_gripper(gripper_x, max_effort)
    
    
    height_delta = 0.3
    position_XYZ = [0.0, 0.0, 0.6449038 + height_delta ]
    orientation_XYZW = [-0.707, 0.000, 0.707, 0.001]
    
    fetch_move_client.move_endeffector(position_XYZ, orientation_XYZW)
    
    height_delta = 0.19
    position_XYZ = [0.0, 0.0, 0.6449038 + height_delta ]
    orientation_XYZW = [-0.707, 0.000, 0.707, 0.001]
    
    fetch_move_client.move_endeffector(position_XYZ, orientation_XYZW)
    
    
    gripper_x = 0.2
    max_effort = 20.0
    fetch_move_client.move_gripper(gripper_x, max_effort)
    
    
    height_delta = 0.3
    position_XYZ = [0.0, 0.0, 0.6449038 + height_delta ]
    orientation_XYZW = [-0.707, 0.000, 0.707, 0.001]
    
    fetch_move_client.move_endeffector(position_XYZ, orientation_XYZW)
    
    


if __name__ == '__main__':
    rospy.init_node('fetch_move_client_node', anonymous=True, log_level=rospy.INFO)
    
    
    #test_grasp()
    test_range_world_movements()
