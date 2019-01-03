#!/usr/bin/env python
import os
import rospy
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
import numpy as np
import copy
import sys
#print(sys.path)
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except ValueError:
    ImportError
#print(sys.path)
import cv2
from cv_bridge import CvBridge, CvBridgeError




class RGBCamera(object):

    def __init__(self, rgb_camera_topic_name, camera_info_topic):
        rospy.loginfo("Start RGB Camera Init process...")
        
        self._rgb_camera_topic_name = rgb_camera_topic_name
        self._camera_info_topic = camera_info_topic

        self.bridge_object = CvBridge()
        
        
        rospy.loginfo("Start camera suscriber...")
        self._check_camera_working()
        self._check_camera_info_ready()
        self.listener_camera_info = rospy.Subscriber(self._camera_info_topic, CameraInfo, self.callbackCameraInfo)
        self.image_sub = rospy.Subscriber(self._rgb_camera_topic_name,Image,self.camera_callback)
        rospy.loginfo("RGB Camera Init process...Ready")
        
    def __del__(self):
        self.video_capture = None
        cv2.destroyAllWindows()
    
    def _check_camera_working(self):
        self.video_capture = None
        rospy.logdebug("Waiting for "+self._rgb_camera_topic_name+" to be READY...")
        while self.video_capture is None and not rospy.is_shutdown():
            try:
                image_msg = rospy.wait_for_message(self._rgb_camera_topic_name, Image, timeout=5.0)
                self.video_capture = self.bridge_object.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
                rospy.logdebug("Current "+self._rgb_camera_topic_name+" READY=>")

            except Exception as e: 
                print(e)
                rospy.logerr("Current "+self._rgb_camera_topic_name+" not ready yet, retrying...")
    
    def _check_camera_info_ready(self):
        self.camerainfo = None
        rospy.logdebug("Waiting for "+self._camera_info_topic+" to be READY...")
        while self.camerainfo is None and not rospy.is_shutdown():
            try:
                self.camerainfo = rospy.wait_for_message(self._camera_info_topic, CameraInfo, timeout=5.0)
                rospy.logdebug("Current "+self._camera_info_topic+" READY=>")
            except Exception as e:
                s = str(e)
                rospy.logerr("Error in _check_camera_info_ready = " + s)
                rospy.logerr(self._camera_info_topic+" not ready yet, retrying")
        
    def camera_callback(self,msg):
        
        # We select bgr8 because its the OpenCV encoding by default
        self.video_capture = self.bridge_object.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        
    def callbackCameraInfo(self, camerainfo):
        self.camerainfo = camerainfo
        if self.camerainfo is not None:
            self.K = np.matrix(np.reshape(self.camerainfo.K, (3, 3)))
            self.P = np.matrix(np.reshape(self.camerainfo.P, (3, 4)))
        
    def display_latest_image(self, life_time_ms=1):
        # Display the resulting image
        cv2.imshow("RGB CAM", self.video_capture)
        cv2.waitKey(life_time_ms)
        
    def display_image(self, image_display, life_time_ms=1, name="RGB CAM"):
        # Display the resulting image
        cv2.imshow(name, image_display)
        cv2.waitKey(life_time_ms)
        
    def resize_image(self, original_image, ratio=0.5):
        resized_image = cv2.resize(original_image, (0, 0), fx=ratio, fy=ratio)
        return resized_image
        
    def get_latest_image(self):
        if self.camerainfo is not None:
            return copy.deepcopy(self.video_capture)
        else:
            return None
    
    def get_camera_info(self):
        if self.camerainfo is not None:
            return copy.deepcopy(self.camerainfo)
        else:
            return None
        

def test():
    rospy.init_node('rgb_cam_node', anonymous=True)
   
    rgb_camera_object = RGBCamera("/dynamic_objects/camera/raw_image")

    camera_period = 1.0
    rate = rospy.Rate(1/camera_period)
    for i in range(100):
        rospy.loginfo("New Image..."+str(i))
        rgb_camera_object.display_latest_image(int(camera_period*1000))
        rate.sleep()
    
if __name__ == '__main__':
    test()
