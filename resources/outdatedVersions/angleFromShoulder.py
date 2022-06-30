import cv2
import mediapipe as mp
import numpy as np

# rotation matrix helper functions

def vec_length(v: np.array):
    return np.sqrt(sum(i ** 2 for i in v))


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def look_at(eye: np.array, target: np.array):
    axis_z = normalize((eye - target))
    if vec_length(axis_z) == 0:
        axis_z = np.array((0, -1, 0))

    axis_x = np.cross(np.array((0, 0, 1)), axis_z)
    if vec_length(axis_x) == 0:
        axis_x = np.array((1, 0, 0))

    axis_y = np.cross(axis_z, axis_x)
    rot_matrix = np.matrix([axis_x, axis_y, axis_z]).transpose()
    return rot_matrix

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# For static images:
IMAGE_FILES = []
BG_COLOR = (192, 192, 192) # gray
with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5) as pose:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    image_height, image_width, _ = image.shape
    # Convert the BGR image to RGB before processing.
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
      continue
    print(
        f'Nose coordinates: ('
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})'
    )

    annotated_image = image.copy()
    # Draw segmentation on the image.
    # To improve segmentation around boundaries, consider applying a joint
    # bilateral filter to "results.segmentation_mask" with "image".
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    bg_image = np.zeros(image.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR
    annotated_image = np.where(condition, annotated_image, bg_image)
    # Draw pose landmarks on the image.
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
    # Plot pose world landmarks.
    mp_drawing.plot_landmarks(
        results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)


cap = cv2.VideoCapture(0)
angles = []

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:

  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      break

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    # rotation matrix calculation + quaternion conversion
    if results.pose_landmarks != None :
        orient = look_at(np.array([results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].z]),
                         np.array([results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].z]))
        # print(orient)  # convert each value from radians to degrees

        vec1 = np.array(orient[0], dtype=float)
        vec3 = np.array(orient[1], dtype=float)
        vec4 = np.array(orient[2], dtype=float)
        # normalize to unit length
        vec1 = vec1 / np.linalg.norm(vec1)
        vec3 = vec3 / np.linalg.norm(vec3)
        vec4 = vec4 / np.linalg.norm(vec4)

        M1 = np.zeros((3, 3), dtype=float)  # rotation matrix

        # rotation matrix setup
        M1[:, 0] = vec1
        M1[:, 1] = vec3
        M1[:, 2] = vec4

        # cartesian quaternion
        a = np.math.sqrt(np.math.sqrt((float(1) + M1[0, 0] + M1[1, 1] + M1[2, 2]) ** 2)) * 0.5
        b1 = (M1[2, 1] - M1[1, 2]) / (4 * a)
        b2 = (M1[0, 2] - M1[2, 0]) / (4 * a)
        b3 = (M1[1, 0] - M1[0, 1]) / (4 * a)

        # polar quaternion
        A = np.math.sqrt((a ** 2) +(b1 ** 2) + (b2 ** 2) + (b3 ** 2))
        theta = np.math.acos(a / A)
        B = np.math.sqrt((A ** 2) - (a ** 2))
        cosphi1 = b1 / B
        cosphi2 = b2 / B
        cosphi3 = b3 / B

        realAngle = (((np.rad2deg(theta) / 45) - 1) * 180) # accounting for observed errors

        # print("{0:.2f} + ".format(a), "{0:.2f}i + ".format(b1), "{0:.2f}j + ".format(b2), "{0:.2f}k".format(b3))
        # print('{0:.2f}'.format(A) + 'cos({0:.2f}) + '.format(np.rad2deg(theta)) + '{0:.2f}'.format(A) + 'sin({0:.2f})'.format(np.rad2deg(theta)) + '({0:.2f}i'.format(cosphi1) + ' + {0:.2f}j + '.format(cosphi2) + '{0:.2f}k)'.format(cosphi3))

        cv2.putText(image, 'Angle : {0:.2f}'.format(realAngle), (50, 70), cv2.FONT_HERSHEY_PLAIN, 3,(255, 255, 0), 5)
        angles.append(realAngle)

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Pose', image) # cv2.flip(image, 1)

    if cv2.waitKey(1) == ord('q'):
      break

cap.release()
print()
print('Maximum recorded angle = ' + str(max(angles)))
print('Minimum recorded angle = ' + str(min(angles)))
