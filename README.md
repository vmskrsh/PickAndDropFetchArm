# PickAndDropFetchArm
 Please clone the below repos into the src
  git clone https://bitbucket.org/theconstructcore/domain_randomization_dynamic_objects.git
  git clone https://bitbucket.org/theconstructcore/fetch_tc.git


Use datagen to first generate the data with the following commands

cd /home/user/catkin_ws
rm -rf build/ devel/
source /home/user/.catkin_ws_python3/dnn_venv/bin/activate
source /home/user/.catkin_ws_python3/devel/setup.bash
catkin_make
source devel/setup.bash
rospack profile
roslaunch my_randomgazebomanager_pkg create_training_material_1object.launch
