#!/usr/bin/env python
import rospy
import cv2
import mediapipe as mp
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

class MediapipePoseNode:
    def __init__(self):
        rospy.init_node('mediapipe_pose_node')
        self.sub = rospy.Subscriber('/camera/color/image_raw', Image, self.callback)
        self.pub1 = rospy.Publisher('/image_with_pose', Image, queue_size=1 )
        # self.pub2 = rospy.Publisher('pose_vector', )
        self.current_pose = None

        self.bridge = CvBridge()

        self.model_path = '/home/mob-plat-dev/Mediapipe models/pose_landmarker_lite.task' 
        # self.model_path = '/home/mob-plat-dev/Mediapipe models/pose_landmarker_full.task' 
        # self.model_path = '/home/mob-plat-dev/Mediapipe models/pose_landmarker_heavy.task' 
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
        VisionRunningMode = mp.tasks.vision.RunningMode
        options = PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=self.model_path),running_mode=VisionRunningMode.IMAGE)
        self.detector = vision.PoseLandmarker.create_from_options(options)


    def callback(self, Image):
        try:
            # Convert the ROS Image message to a OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(Image, 'bgr8')
        except CvBridgeError as e:
            rospy.logerr("CvBridgeError: %s" % str(e))

        # Convert img from OpenCV to a MediaPipe’s Image object
        mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = cv_image)
        
        # Extract pose landmarks by running the detector
        detector_result = self.detector.detect(mp_image)

        # Annotate image with pose and convert to ROS img msg
        annotated_mp_image = self.draw_landmarks_on_image(mp_image.numpy_view(), detector_result)
        corrected_mp_image = annotated_mp_image[:, :, ::-1] # Swap Reds and Blues
        annotated_cv_image = cv2.cvtColor(corrected_mp_image, cv2.COLOR_RGB2BGR)
        annotated_ros_image = self.bridge.cv2_to_imgmsg(annotated_cv_image, 'bgr8')

        # Publish the img with pose
        self.pub1.publish(annotated_ros_image)

        # Print pose
        print("Pose", detector_result.pose_landmarks[0])

        # calculate vector for diff in pose
        # if self.current_pose == None:
        #     self.current_pose = detector_result.pose_landmarks
        # else:
        #     diff = 

        # publish vector

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        pose_landmarks_list = detection_result.pose_landmarks
        annotated_image = np.copy(rgb_image)

        # Loop through the detected poses to visualize.
        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]

            # Draw the pose landmarks.
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks])
            solutions.drawing_utils.draw_landmarks(annotated_image, pose_landmarks_proto, solutions.pose.POSE_CONNECTIONS, solutions.drawing_styles.get_default_pose_landmarks_style())
        
        return annotated_image


def main():
    mediapipe_pose_node = MediapipePoseNode()
    rospy.spin()

if __name__ == main():
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.logwarn("Node Not Executed")