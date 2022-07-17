"""

singlePedestrianPathIntersectionv1.1.py

- This script extrapolates the predicted path in completion assuming no sudden changes in fuzzy state for each frame based on data from the last defined one.
- This extrapolation is done over a frame containing a pre defined path overlay for the robot that the camera is assumed to be mounted on.
- While it is mentioned in the code that the prediction is made 100 seconds into the future, this is merely an elegant hack to generate a full path from the location prediction source codes in the other two scripts given in the directory.
- This script doesn't highlight the likelihood of collision. Refer to main .py for the same.

"""

# importing the necessary libraries
import os
import cv2
import time
import numpy as np
import mediapipe as mp


# rotation matrix helper functions

# function to return the magnitude of a vector
def vec_length(v: np.array):
    return np.sqrt(sum(i ** 2 for i in v))

# function to process a vector parameter and return a normalized vector
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

# function to calculate and return a rotation matrix for quaternion generation
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


# filter(s)

# Kalman filter in one dimension implemented as a class
class Kalman:

    def __init__(self, windowSize=10, n=5):
        # x: predicted angle
        # p: predicted angle uncertainty
        # n: number of iterations to run the filter for
        # dt: time interval for updates
        # q: process noise variance (uncertainty in the system's dynamic model)
        # r: measurement uncertainty
        # Z: list of position estimates derived from sensor measurements

        # initializing with static values due to very low variance in testing
        self.x = 0
        self.p = 0.5
        self.windowSize = windowSize
        self.n = n # must be smaller than windowSize
        self.Z = []

        self.q = 0 # assuming dynamic model uncertainty to be 0 (perfect system)
        self.dt = 0.05 # average latency is 50ms
        self.r = 0.5 # angle measurement uncertainty (determine experimentally based on test case)

    # prediction stage
    def predict(self):
        # prediction assuming a dynamic model
        self.x = self.x   # state transition equation
        self.p = self.p + self.q  # predicted covariance equation

    # measurement stage
    def measure(self, z):

        if len(self.Z) < self.windowSize:
            self.Z.append(z)
        else:
            self.Z.pop(0)
            self.Z.append(z)

        return np.mean(self.Z)

    # updation stage
    def update(self, z):
        k = self.p / (self.p + self.r)  # Kalman gain
        self.x = self.x + k * (z - self.x)  # state update
        self.p = (1 - k) * self.p  # covariance update

    # iterative processing stage
    def process(self, i):

        for j in range(1, self.n):
            self.predict()
            z = self.measure(i)
            self.update(z)

        return self.x

# streaming moving average filter in one dimension implemented as a class
class StreamingMovingAverage:

    def __init__(self, window_size):
        self.window_size = window_size # size of the window of values
        self.values = [] # list to hold said window
        self.sum = 0 # initializing the sum for the moving average

    # processing the average
    def process(self, value):
        self.values.append(value)
        self.sum += value
        if len(self.values) > self.window_size:
            self.sum -= self.values.pop(0)
        return float(self.sum) / len(self.values)

# empty filter class implemented for comparative testing
class noFilter:

    def __init__(self):
        pass

    def process(self, value):
        return value

# pose detector class for mediapipe
class PoseDetector:

    """
    Estimates Pose points of a human body using the mediapipe library.
    """

    def __init__(self, mode=False, smooth=True, detectionCon=0.5, trackCon=0.5):

        self.mode = mode
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode,
                                     smooth_landmarks=self.smooth,
                                     min_detection_confidence=self.detectionCon,
                                     min_tracking_confidence=self.trackCon)

    # a method to initialize the filters used in objects of this class (i.e.; single human obstacles detected by the mediapipe model)
    def filterSettings(self, xFilter, yFilter, angleFilter):

        self.xFilter = xFilter
        self.yFilter = yFilter
        self.angleFilter = angleFilter

    # a method to detect and draw the landmarks detected by the model on the input frame
    def findPose(self, img, draw=True):


        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True, bboxWithHands=False):

        self.lmList = []
        self.bboxInfo = {}
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy, cz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                self.lmList.append([id, cx, cy, cz])

            # bounding box generation
            ad = abs(self.lmList[12][1] - self.lmList[11][1]) // 2
            if bboxWithHands:
                x1 = self.lmList[16][1] - ad
                x2 = self.lmList[15][1] + ad
            else:
                x1 = self.lmList[12][1] - ad
                x2 = self.lmList[11][1] + ad

            y2 = self.lmList[29][2] + ad
            y1 = self.lmList[1][2] - ad
            bbox = (x1, y1, x2 - x1, y2 - y1)
            cx, cy = bbox[0] + (bbox[2] // 2), \
                     (bbox[1] + bbox[3] // 2) - 40

            self.bboxInfo = {"bbox": bbox, "center": (cx, cy)}

            if draw:
                cv2.rectangle(img, bbox, (255, 0, 255), 3)
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        return self.lmList, self.bboxInfo

    # a method to extract the angle of orientation of the detected human obstacle by quaternion generation from the projection of two landmarks
    def angleOfOrientation(self, p1, p2):

        if self.results.pose_landmarks != None:
            # calculating the rotation matrix
            orient = look_at(np.array([p1[1], p1[2], p1[3]]), np.array([p2[1], p2[2], p2[3]]))

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

            # obtaining the quaternion in cartesian form
            a = np.math.sqrt(np.math.sqrt((float(1) + M1[0, 0] + M1[1, 1] + M1[2, 2]) ** 2)) * 0.5
            b1 = (M1[2, 1] - M1[1, 2]) / (4 * a)
            b2 = (M1[0, 2] - M1[2, 0]) / (4 * a)
            b3 = (M1[1, 0] - M1[0, 1]) / (4 * a)

            # converting quaternion to polar form
            A = np.math.sqrt((a ** 2) + (b1 ** 2) + (b2 ** 2) + (b3 ** 2))
            theta = np.math.acos(a / A)
            # B = np.math.sqrt((A ** 2) - (a ** 2))
            # cosphi1 = b1 / B
            # cosphi2 = b2 / B
            # cosphi3 = b3 / B

            realAngle = ((np.rad2deg(theta) / 45) - 1) * 180

            # filtering the reading
            realAngle = self.angleFilter.process(realAngle)

            return realAngle

    # fuzzy classification of the predicted direction of movement into 9 different cases
    def futureXY(self, img, lmls, lmrs, init, angleOfApproach, centerXApproachSpeed, centerYApproachSpeed, timeToFuture, err, draw=True):

        # the case covering movements toward the top right of the frame (i.e.; away and to the right of the single camera's perspective)
        if (angleOfApproach > 0) and (angleOfApproach < 90) and (centerXApproachSpeed > 0) and (centerYApproachSpeed < 0):
            futureX = self.xFilter.process(init[0] + np.math.sqrt(
                ((centerXApproachSpeed * timeToFuture) * np.math.cos(angleOfApproach)) ** 2))
            futureY = self.yFilter.process(init[1] - np.math.sqrt(
                ((centerYApproachSpeed * timeToFuture) * np.math.cos(angleOfApproach)) ** 2))

            # visualizing the direction of movement
            if draw == True:
                cv2.drawMarker(img, (int(futureX), int(futureY)), color=(0, 255, 0), markerType=cv2.MARKER_CROSS,thickness=2)
                cv2.line(img, (lmls[1], lmls[2]), (int(futureX), int(futureY)), (255, 255, 255), 2)
                cv2.line(img, (lmrs[1], lmrs[2]), (int(futureX), int(futureY)), (255, 255, 255), 2)
                cv2.putText(img, 'CASE 1', (200, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 0), 1, cv2.LINE_AA)

        # the case covering movements toward the bottom left of the frame (i.e.; toward and to the left of the single camera's perspective)
        elif (angleOfApproach > 0) and (angleOfApproach < 90) and (centerXApproachSpeed < 0) and (centerYApproachSpeed > 0):
            futureX = self.xFilter.process(init[0] - np.math.sqrt(
                ((centerXApproachSpeed * timeToFuture) * np.math.cos(angleOfApproach)) ** 2))
            futureY = self.yFilter.process(init[1] + np.math.sqrt(
                ((centerYApproachSpeed * timeToFuture) * np.math.cos(angleOfApproach)) ** 2))

            # visualizing the direction of movement
            if draw == True:
                cv2.drawMarker(img, (int(futureX), int(futureY)), color=(0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=2)
                cv2.line(img, (lmls[1], lmls[2]), (int(futureX), int(futureY)), (255, 255, 255), 2)
                cv2.line(img, (lmrs[1], lmrs[2]), (int(futureX), int(futureY)), (255, 255, 255), 2)
                cv2.putText(img, 'CASE 2', (200, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 0), 1, cv2.LINE_AA)

        # the case covering movements toward the bottom right of the frame (i.e.; toward and to the right of the single camera's perspective)
        elif (angleOfApproach > 90) and (angleOfApproach < 180) and (centerXApproachSpeed > 0) and (centerYApproachSpeed > 0):
            futureX = self.xFilter.process(init[0] + np.math.sqrt(
                ((centerXApproachSpeed * timeToFuture) * np.math.cos(angleOfApproach)) ** 2))
            futureY = self.yFilter.process(init[1] + np.math.sqrt(
                ((centerYApproachSpeed * timeToFuture) * np.math.cos(angleOfApproach)) ** 2))

            # visualizing the direction of movement
            if draw == True:
                cv2.drawMarker(img, (int(futureX), int(futureY)), color=(0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=2)
                cv2.line(img, (lmls[1], lmls[2]), (int(futureX), int(futureY)), (255, 255, 255), 2)
                cv2.line(img, (lmrs[1], lmrs[2]), (int(futureX), int(futureY)), (255, 255, 255), 2)
                cv2.putText(img, 'CASE 3', (200, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 0), 1, cv2.LINE_AA)

        # the case covering movements toward the top left of the frame (i.e.; away and to the left of the single camera's perspective)
        elif (angleOfApproach > 90) and (angleOfApproach < 180) and (centerXApproachSpeed < 0) and (centerYApproachSpeed < 0):
            futureX = self.xFilter.process(init[0] - np.math.sqrt(
                ((centerXApproachSpeed * timeToFuture) * np.math.cos(angleOfApproach)) ** 2))
            futureY = self.yFilter.process(init[1] - np.math.sqrt(
                ((centerYApproachSpeed * timeToFuture) * np.math.cos(angleOfApproach)) ** 2))

            # visualizing the direction of movement
            if draw == True:
                cv2.drawMarker(img, (int(futureX), int(futureY)), color=(0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=2)
                cv2.line(img, (lmls[1], lmls[2]), (int(futureX), int(futureY)), (255, 255, 255), 2)
                cv2.line(img, (lmrs[1], lmrs[2]), (int(futureX), int(futureY)), (255, 255, 255), 2)
                cv2.putText(img, 'CASE 4', (200, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 0), 1, cv2.LINE_AA)

        # the case covering movements toward the right of the frame (i.e.; to the right of the single camera's perspective)
        elif (((angleOfApproach > 0 - err) and (angleOfApproach < 0 + err)) or ((angleOfApproach > 180 - err) and (angleOfApproach < 180 + err))) and (centerXApproachSpeed > 0) and ((centerYApproachSpeed > 0 - err) and (centerYApproachSpeed < 0 + err)):
            futureX = self.xFilter.process(init[0] + np.math.sqrt(
                ((centerXApproachSpeed * timeToFuture) * np.math.cos(angleOfApproach)) ** 2))
            futureY = self.yFilter.process(init[1])

            # visualizing the direction of movement
            if draw == True:
                cv2.drawMarker(img, (int(futureX), int(futureY)), color=(0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=2)
                cv2.line(img, (lmls[1], lmls[2]), (int(futureX), int(futureY)), (255, 255, 255), 2)
                cv2.line(img, (lmrs[1], lmrs[2]), (int(futureX), int(futureY)), (255, 255, 255), 2)
                cv2.putText(img, 'CASE 5', (200, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 0), 1, cv2.LINE_AA)

        # the case covering movements toward the left of the frame (i.e.; to the left of the single camera's perspective)
        elif (((angleOfApproach > 0 - err) and (angleOfApproach < 0 + err)) or ((angleOfApproach > 180 - err) and (angleOfApproach < 190 - err))) and (centerXApproachSpeed < 0) and ((centerYApproachSpeed > 0 - err) and (centerYApproachSpeed < 0 + err)):
            futureX = self.xFilter.process(init[0] - np.math.sqrt(
                ((centerXApproachSpeed * timeToFuture) * np.math.cos(angleOfApproach)) ** 2))
            futureY = self.yFilter.process(init[1])

            # visualizing the direction of movement
            if draw == True:
                cv2.drawMarker(img, (int(futureX), int(futureY)), color=(0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=2)
                cv2.line(img, (lmls[1], lmls[2]), (int(futureX), int(futureY)), (255, 255, 255), 2)
                cv2.line(img, (lmrs[1], lmrs[2]), (int(futureX), int(futureY)), (255, 255, 255), 2)
                cv2.putText(img, 'CASE 6', (200, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 0), 1, cv2.LINE_AA)

        # the case covering movements toward the top of the frame (i.e.; away from the single camera's perspective)
        elif ((angleOfApproach > 90 - err) and (angleOfApproach < 90 + err)) and ((centerXApproachSpeed > 0 - err) and (centerXApproachSpeed < 0 + err)) and (centerYApproachSpeed > 0):
            futureX = self.xFilter.process(init[0])
            futureY = self.yFilter.process(init[1] - np.math.sqrt(
                ((centerYApproachSpeed * timeToFuture) * np.math.cos(angleOfApproach)) ** 2))

            # visualizing the direction of movement
            if draw == True:
                cv2.drawMarker(img, (int(futureX), int(futureY)), color=(0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=2)
                cv2.line(img, (lmls[1], lmls[2]), (int(futureX), int(futureY)), (255, 255, 255), 2)
                cv2.line(img, (lmrs[1], lmrs[2]), (int(futureX), int(futureY)), (255, 255, 255), 2)
                cv2.putText(img, 'CASE 7', (200, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 0), 1, cv2.LINE_AA)

        # the case covering movements toward the bottom of the frame (i.e.; towards the single camera's perspective)
        elif ((angleOfApproach > 90 - err) and (angleOfApproach < 90 + err)) and ((centerXApproachSpeed > 0 - err) and (centerXApproachSpeed < 0 + err)) and (centerYApproachSpeed < 0):
            futureX = self.xFilter.process(init[0])
            futureY = self.yFilter.process(init[1] + np.math.sqrt(
                ((centerYApproachSpeed * timeToFuture) * np.math.cos(angleOfApproach)) ** 2))

            # visualizing the direction of movement
            if draw == True:
                cv2.drawMarker(img, (int(futureX), int(futureY)), color=(0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=2)
                cv2.line(img, (lmls[1], lmls[2]), (int(futureX), int(futureY)), (255, 255, 255), 2)
                cv2.line(img, (lmrs[1], lmrs[2]), (int(futureX), int(futureY)), (255, 255, 255), 2)
                cv2.putText(img, 'CASE 8', (200, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 0), 1, cv2.LINE_AA)

        # the edge case for a lack of significant movement requiring prediction of a location seperate from the future reality
        else:
            futureX = self.xFilter.process(init[0])
            futureY = self.yFilter.process(init[1])

            # visualizing the direction of movement
            if draw == True:
                cv2.drawMarker(img, (int(futureX), int(futureY)), color=(0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=2)
                cv2.line(img, (lmls[1], lmls[2]), (int(futureX), int(futureY)), (255, 255, 255), 2)
                cv2.line(img, (lmrs[1], lmrs[2]), (int(futureX), int(futureY)), (255, 255, 255), 2)
                cv2.putText(img, 'CASE 9', (200, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 0), 1, cv2.LINE_AA)

        return futureX, futureY

def main(path):

    # initializing the frame and overlay settings
    pathOverlay = cv2.imread('resources/overlays/pathOverlayBlack.png') # getting the overlay image
    cap = cv2.VideoCapture(path) # initializing the test video path
    cap.set(3, 768) # setting the width of the frame
    cap.set(4, 432) # setting the height of the frame

    # FPS initializations
    curTime = time.time() # initializing the starting time
    lastTime = curTime # initializing the time in the last frame to the current time itself
    fps = 0 # initializing the frame rate variable
    frameNumber = 0 # initializing the frame number to zero

    # general initializations
    lastXCenter = 0 # location of the center's X coordinate in the last frame
    lastYCenter = 0 # location of the center's Y coordinate in the last frame
    occupiedHeight = 0 # a ratio of the height of the target on the frame to the frame width itself as a measure of nearness of the obstacle to the camera
    centerXApproachSpeed = 0 # the X component of the velocity of the detected obstacle
    centerYApproachSpeed = 0 # the Y component of the velocity of the detected obstacle
    angleOfApproach = 0 # the recorded angle of orientation of the obstacle
    lastDeltaY = 0 # the height difference between the highest and lowest points of the obstacle on the last frame

    # future definitions
    futureX = 0  # the predicted X coordinate of the location of the obstacle
    futureY = 0  # the predicted Y coordinate of the location of the obstacle

    # past frame definitions
    currentFrame = 0  # the number of the current frame
    frameWindow = 4  # the window size of frames past the present for initializations as the last state of the system

    # pose detector settings and variables that visibly impact output
    detector = PoseDetector()
    # setting the filter options for the pose detector class
    # filter options : StreamingMovingAverage(10), Kalman(windowSize=20, n=10), noFilter()
    detector.filterSettings(xFilter=StreamingMovingAverage(5),
                            yFilter=StreamingMovingAverage(5),
                            angleFilter=Kalman(windowSize=25, n=10))
    timeToFuture = 100 # all collision predictions are made for these many seconds into the future
    futureErrorThresholds = 10# the error thresholds for the fuzzy states for both linear and angular measurements as a lower proportional error margin is to be tolerated for angular variations than linear ones
    drawState = True # the boolean determining whether or not landmarks and their associated line segments are to be drawn on the frame image

    while True:

        # reading the image
        success, img = cap.read()

        # adding the video break conditions
        if (cv2.waitKey(1) == ord('q')) or (not success):
            break

        # resizing the image to fit the frame
        img = cv2.resize(img, (768, 432))
        # flipping the image to get a real depiction of the scene
        img = cv2.flip(img, 1)

        # finding the landmarks and visualizing them
        img = detector.findPose(img, draw=drawState)

        # resizing & adding a standard path overlay
        pathOverlay = cv2.resize(pathOverlay, (768, 432))
        img = cv2.addWeighted(img,0.7,pathOverlay,0.3,0)

        # getting a list of landmarks and bounding box information
        lmList, bboxInfo = detector.findPosition(img, draw=False, bboxWithHands=False)

        # code to be executed if a human obstacle is detected (i.e.; if ia bounding box is generatable)
        if bboxInfo:

            # finding the center of the target pose
            center = bboxInfo["center"]

            yLocations = []
            for lm in lmList:
                yLocations.append(lm[2])

                # getting the landmarks corresponding to the shoulders
                if (lm[0] == 12):
                    lmrs = lm
                elif (lm[0] == 11):
                    lmls = lm

            # finding the difference between highest landmark and lowest landmark in pixels
            deltaY = max(yLocations) - min(yLocations)

            occupiedHeight = deltaY / 432 # an indicator of pedestrian's apparent height (in terms of percentage of the Y axis occupied)

            # calculating the velocity vector components of the obstacle with respect to the frame
            centerXApproachSpeed = (center[0] - lastXCenter) / (time.time() - lastTime)
            centerYApproachSpeed = (center[1] - lastYCenter) / (time.time() - lastTime)

            # angle of approach reporting currently accurate only between the range of 30 and 160 degrees
            angleOfApproach = detector.angleOfOrientation(lmls, lmrs)

            # filtering, predicting & drawing the future location of the target pedestrian
            futureX, futureY = detector.futureXY(img, lmls, lmrs, center, angleOfApproach, centerXApproachSpeed, centerYApproachSpeed, timeToFuture, futureErrorThresholds, draw=drawState)

            # generating a representation of the path taken by the obstacle
            # this visualization was accomplished by constructing extrapolated parallel straight lines along the line connecting the centroid of the shoulder points and the predicted location coordinates on the frame by origin shifting
            oShiftX = futureX - ((lmls[1] + lmrs[1]) / 2)
            oShiftY = futureY - ((lmls[2] + lmrs[2]) / 2)
            x1 = lmls[1] + oShiftX
            y1 = lmls[2] + oShiftY
            x2 = lmrs[1] + oShiftX
            y2 = lmrs[2] + oShiftY

            # printing the conditions for collision prediction
            cv2.putText(img, '{0:.2f}'.format(occupiedHeight), (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 0),1, cv2.LINE_AA)
            cv2.putText(img, '{0:.2f}'.format(angleOfApproach), (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 0), 1, cv2.LINE_AA)

            # green zone - indicating a lack of any risk of collision due to a target percieved to be too far away
            if occupiedHeight < 0.5 :
                cv2.line(img, (lmls[1], lmls[2]), (lmrs[1], lmrs[2]), (255, 255, 255), 2)
                cv2.line(img, (lmls[1], lmls[2]), (int(x1), int(y1)), (0,128,0), 2)
                cv2.line(img, (lmrs[1], lmrs[2]), (int(x2), int(y2)), (0,128,0), 2)

            # yellow zone - indicating a minor risk of collision due to a nearby target which may be on a course that collides with the assumed robot
            elif occupiedHeight > 0.5 and occupiedHeight < 1 :
                cv2.line(img, (lmls[1], lmls[2]), (lmrs[1], lmrs[2]), (255, 255, 255), 2)
                cv2.line(img, (lmls[1], lmls[2]), (int(x1), int(y1)), (0,255,255), 2)
                cv2.line(img, (lmrs[1], lmrs[2]), (int(x2), int(y2)), (0,255,255), 2)

        # red zone - indicating a guaranteed collision when the predicted future coordinates fall within a projection of the future location of the robot on the given frame
            if occupiedHeight > 1 :
                cv2.line(img, (lmls[1], lmls[2]), (lmrs[1], lmrs[2]), (255, 255, 255), 2)
                cv2.line(img, (lmls[1], lmls[2]), (int(x1), int(y1)), (0,0,255), 2)
                cv2.line(img, (lmrs[1], lmrs[2]), (int(x2), int(y2)), (0,0,255), 2)

            # highlighting shoulder points
            cv2.circle(img, (lmls[1], lmls[2]), 5, (255, 255, 255), cv2.FILLED)
            cv2.circle(img, (lmrs[1], lmrs[2]), 5, (255, 255, 255), cv2.FILLED)

            # a series of conditional definitions to define the state of the last frame taken ('last' here being a relative term)
            currentFrame += 1
            frameNumber += 1
            if (currentFrame == frameWindow) and (frameNumber > frameWindow):
                lastXCenter = center[0]
                lastYCenter = center[1]
                lastDeltaY = deltaY
                lastTime = time.time()
                currentFrame = 0

            elif (frameNumber < frameWindow):
                lastXCenter = center[0]
                lastYCenter = center[1]
                lastDeltaY = deltaY
                currentFrame = 0

            else:
                pass

        # FPS calculation
        fps = 1 / (time.time() - curTime)
        curTime = time.time()
        cv2.putText(img, '{0:.2f}'.format(fps), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img, '{0:.2f}'.format((1 / fps) * 1000), (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 0), 1, cv2.LINE_AA)

        # scale
        cv2.putText(img, '50px: ', (675, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 255, 0), 1, cv2.LINE_AA)
        cv2.line(img, (710, 17), (760, 17), (100, 255, 0), 1)
        cv2.line(img, (710, 17), (710, 22), (100, 255, 0), 1)
        cv2.line(img, (760, 17), (760, 22), (100, 255, 0), 1)

        # printing the path of the test video
        cv2.putText(img, str(path), (300, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 0), 1, cv2.LINE_AA)

        # showing the processed frame
        cv2.imshow("img", img)

    # releasing & destroying windows
    cap.release()
    cv2.destroyAllWindows()


# a definition of the main parameters
if __name__ == "__main__":

    # defining the directory to obtain the test videos from
    directory = 'resources\stockTestFootage'

    # listing all the test videos within the directory
    for filename in os.listdir(directory):

        f = os.path.join(directory, filename)

        # checking for the validity of a file path
        if os.path.isfile(f):

            main(f)