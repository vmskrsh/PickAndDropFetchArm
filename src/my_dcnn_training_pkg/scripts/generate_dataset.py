#!/usr/bin/env python

# We load Python stuff first because afterwards it will be removed to avoid error with openCV
import sys
sys.path.append('/usr/lib/python2.7/dist-packages')
import rospy
import rospkg


rospy.logdebug("Start Module Loading...Remove CV ROS-Kinetic version due to incompatibilities")
import csv

rospy.logdebug(sys.path)
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except ValueError:
    rospy.logdebug("Its already removed..../opt/ros/kinetic/lib/python2.7/dist-packages")
rospy.logdebug(sys.path)
import cv2
import glob
import os
import xml.etree.ElementTree as ET
import shutil

SPLIT_RATIO = 0.8

AUGMENTATION = False
AUGMENTATION_DEBUG = False
AUGMENTATION_PER_IMAGE = 25

try:
    import imgaug as ia
    from imgaug import augmenters as iaa
except ImportError:
    rospy.logdebug("Augmentation disabled")
    AUGMENTATION = False

rospy.logdebug("Loaded Modules...")



def generate_database_now(path_to_source_training_package, image_size, path_to_database_training_package):
    
    rospy.loginfo("Start main...")
    
    ######### START OF Directories Setup

    dataset_images_folder = os.path.join(path_to_source_training_package, "dataset_gen/images")
    dataset_annotations_folder = os.path.join(path_to_source_training_package, "dataset_gen_annotations")
    
    train_images_folder = os.path.join(path_to_database_training_package, "train")
    validation_images_folder = os.path.join(path_to_database_training_package, "validation")
    
    csv_folder_path = os.path.join(path_to_database_training_package, "dataset_gen_csv")
    train_csv_output_file = os.path.join(csv_folder_path, "train.csv")
    validation_csv_output_file = os.path.join(csv_folder_path, "validation.csv")

    if not os.path.exists(dataset_images_folder):
        rospy.logfatal("Dataset not found==>"+str(dataset_images_folder)+", please run rosrun my_randomgazebomanager_pkg create_training_material.py")
        return False
    else:
        rospy.loginfo("Trainin Images path found ==>"+str(dataset_images_folder))

    # We clean up the training folders
    if os.path.exists(train_images_folder):
        shutil.rmtree(train_images_folder)
    os.makedirs(train_images_folder)
    rospy.loginfo("Created folder=" + str(train_images_folder))

    if os.path.exists(validation_images_folder):
        shutil.rmtree(validation_images_folder)
    os.makedirs(validation_images_folder)
    rospy.loginfo("Created folder=" + str(validation_images_folder))

    if os.path.exists(csv_folder_path):
        shutil.rmtree(csv_folder_path)
    os.makedirs(csv_folder_path)
    rospy.loginfo("Created folder=" + str(csv_folder_path))

    ######### END OF Directories Setup
    rospy.loginfo("END OF Directories Setup")

    class_names = {}
    k = 0
    output = []

    rospy.loginfo("Retrieving the xml files")
    xml_files = glob.glob("{}/*xml".format(dataset_annotations_folder))
    rospy.loginfo("END Retrieving the xml files")

    rospy.loginfo("Reading XML Anotation Files, this could take a while depending on the number of files, please be patient...")
    for i, xml_file in enumerate(xml_files):
        tree = ET.parse(xml_file)

        path = os.path.join(dataset_images_folder, tree.findtext("./filename"))
        rospy.logdebug("path from XML==>"+str(path))

        height = int(tree.findtext("./size/height"))
        width = int(tree.findtext("./size/width"))

        x_com = float(tree.findtext("./object/pose3d/x_com"))
        y_com = float(tree.findtext("./object/pose3d/y_com"))
        z_com = float(tree.findtext("./object/pose3d/z_com"))
        quat_x = float(tree.findtext("./object/pose3d/quat_x"))
        quat_y = float(tree.findtext("./object/pose3d/quat_y"))
        quat_z = float(tree.findtext("./object/pose3d/quat_z"))
        quat_w = float(tree.findtext("./object/pose3d/quat_w"))

        class_name = tree.findtext("./object/name")
        if class_name not in class_names:
            class_names[class_name] = k
            k += 1

        output.append((path, width, height, x_com, y_com, z_com, quat_x, quat_y, quat_z, quat_w, class_name, class_names[class_name]))
        
        print("{}/{}".format(i, xml_file), end="\r")

    rospy.logdebug(str(output))

    # Get the Number of elements of the same class, in this case all, because we only train one item.
    lengths = []
    i = 0
    last = 0
    rospy.loginfo("Getting Object elements...")
    for j, row in enumerate(output):
        if last == row[-1]:
            i += 1
        else:
            rospy.logdebug("class {}: {} images".format(output[j-1][-2], i))
            lengths.append(i)
            i = 1
            last += 1

    lengths.append(i)
    rospy.loginfo("Object elements==>"+str(lengths))


    ## START CSV generation
    rospy.loginfo("Starting CSV database Generation...")
    with open(train_csv_output_file, "w") as train, open(validation_csv_output_file, "w") as validate:
        csv_train_writer = csv.writer(train, delimiter=",")
        csv_validate_writer = csv.writer(validate, delimiter=",")

        s = 0
        for c in lengths:
            for i in range(c):
                print("{}/{}".format(s + 1, sum(lengths)), end="\r")

                path, width, height, x_com, y_com, z_com, quat_x, quat_y, quat_z, quat_w, class_name, class_id = output[s]
                absolute_original_path = os.path.abspath(path)
                data_list = [absolute_original_path, width, height, x_com, y_com, z_com, quat_x, quat_y, quat_z, quat_w, class_name, class_names[class_name]]

                # We decide if it goes to train folder or to validate folder

                if i <= c * SPLIT_RATIO:
                    basename = os.path.basename(data_list[0])
                    train_scaled_img_path = os.path.join(train_images_folder, basename)
                    data_list[0] = os.path.abspath(train_scaled_img_path)
                    csv_train_writer.writerow(data_list)
                else:
                    basename = os.path.basename(data_list[0])
                    validate_scaled_img_path = os.path.join(validation_images_folder, basename)
                    data_list[0] = os.path.abspath(validate_scaled_img_path)
                    csv_validate_writer.writerow(data_list)

                image = cv2.imread(absolute_original_path)

                cv2.imwrite(data_list[0], cv2.resize(image, (image_size, image_size)))

                s += 1

    ## END CSV generation

    rospy.loginfo("\nDone!")

    return True


def main():
    
    rospy.init_node('generate_database_node', anonymous=True, log_level=rospy.INFO)
    rospy.logwarn("Generate Database...START")
    
    if len(sys.argv) < 4:
        rospy.logfatal("usage: generate_dataset.py path_to_source_training_package image_size path_to_database_training_package")
    else:
        path_to_source_training_package = sys.argv[1]
        image_size = int(sys.argv[2])
        path_to_database_training_package = sys.argv[3]
        
        if path_to_database_training_package == "None":
            rospack = rospkg.RosPack()
            # get the file path for dcnn_training_pkg
            path_to_database_training_package = rospack.get_path('my_dcnn_training_pkg')
            rospy.logwarn("NOT Found path_to_database_training_package, getting default:"+str(path_to_database_training_package))
        else:
            rospy.logwarn("Found path_to_database_training_package:"+str(path_to_database_training_package))
        
        rospy.logwarn("Path to Training Original Material:"+str(path_to_source_training_package))
        rospy.logwarn("image_size to generate database:"+str(image_size))
        rospy.logwarn("image_size to generate database:"+str(path_to_database_training_package))
        
        generate_database_now(  path_to_source_training_package,
                                image_size,
                                path_to_database_training_package)
        
        rospy.logwarn("Generate Database...END")


if __name__ == "__main__":
    main()
