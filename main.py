import cv2
import math
import time
import numpy as np
import pandas as pd
import mediapipe as mp
import matplotlib.pyplot as plt


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

# filter(s)
class StreamingMovingAverage:

    def __init__(self, window_size):
        self.window_size = window_size
        self.values = []
        self.sum = 0

    def process(self, value):
        self.values.append(value)
        self.sum += value
        if len(self.values) > self.window_size:
            self.sum -= self.values.pop(0)
        return float(self.sum) / len(self.values)


# pose detector class
class PoseDetector:

    """
    Estimates Pose points of a human body using the mediapipe library.
    """

    def __init__(self, mode=False, smooth=True, detectionCon=0.5, trackCon=0.5):

        """
        :param mode: In static mode, detection is done on each image: slower
        :param upBody: Upper boy only flag
        :param smooth: Smoothness Flag
        :param detectionCon: Minimum Detection Confidence Threshold
        :param trackCon: Minimum Tracking Confidence Threshold
        """

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

    def findPose(self, img, draw=True):

        """
        Find the pose landmarks in an Image of BGR color space.
        :param img: Image to find the pose in.
        :param draw: Flag to draw the output on the image.
        :return: Image with or without drawings
        """

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

            # Bounding Box
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

    def findAngle(self, img, p1, p2, p3, draw=True):

        """
        Finds angle between three points. Inputs index values of landmarks
        instead of the actual points.
        :param img: Image to draw output on.
        :param p1: Point1 - Index of Landmark 1.
        :param p2: Point2 - Index of Landmark 2.
        :param p3: Point3 - Index of Landmark 3.
        :param draw:  Flag to draw the output on the image.
        :return:
        """

        # Get the landmarks
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # Calculate the Angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360

        # Draw
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return angle

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):

        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]

    def angleCheck(self, myAngle, targetAngle, addOn=20):
        return targetAngle - addOn < myAngle < targetAngle + addOn

    # quaternion conversions done in this next function
    def angleOfOrientation(self, p1, p2):

        if self.results.pose_landmarks != None:
            # calculating the rotation matrix
            orient = look_at(np.array([p1[1], p1[2], p1[3]]), np.array([p2[1], p2[2], p2[3]]))
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

            return realAngle

    # implicit fuzzy classification implemented here
    def futureXY(self, img, lmls, lmrs, init, angleOfApproach, centerXApproachSpeed, centerYApproachSpeed, timeToFuture, err, draw=True):

        if (angleOfApproach > 0) and (angleOfApproach < 90) and (centerXApproachSpeed > 0) and (centerYApproachSpeed < 0):
            futureX = init[0] + np.math.sqrt(
                ((centerXApproachSpeed * timeToFuture) * np.math.cos(angleOfApproach)) ** 2)
            futureY = init[1] - np.math.sqrt(
                ((centerYApproachSpeed * timeToFuture) * np.math.cos(angleOfApproach)) ** 2)

            if draw == True:
                cv2.drawMarker(img, (int(futureX), int(futureY)), color=(0, 255, 0), markerType=cv2.MARKER_CROSS,thickness=2)
                cv2.line(img, (lmls[1], lmls[2]), (int(futureX), int(futureY)), (255, 255, 255), 2)
                cv2.line(img, (lmrs[1], lmrs[2]), (int(futureX), int(futureY)), (255, 255, 255), 2)

        elif (angleOfApproach > 0) and (angleOfApproach < 90) and (centerXApproachSpeed < 0) and (centerYApproachSpeed > 0):
            futureX = init[0] - np.math.sqrt(
                ((centerXApproachSpeed * timeToFuture) * np.math.cos(angleOfApproach)) ** 2)
            futureY = init[1] + np.math.sqrt(
                ((centerYApproachSpeed * timeToFuture) * np.math.cos(angleOfApproach)) ** 2)

            if draw == True:
                cv2.drawMarker(img, (int(futureX), int(futureY)), color=(0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=2)
                cv2.line(img, (lmls[1], lmls[2]), (int(futureX), int(futureY)), (255, 255, 255), 2)
                cv2.line(img, (lmrs[1], lmrs[2]), (int(futureX), int(futureY)), (255, 255, 255), 2)

        elif (angleOfApproach > 90) and (angleOfApproach < 180) and (centerXApproachSpeed > 0) and (centerYApproachSpeed > 0):
            futureX = init[0] + np.math.sqrt(
                ((centerXApproachSpeed * timeToFuture) * np.math.cos(angleOfApproach)) ** 2)
            futureY = init[1] + np.math.sqrt(
                ((centerYApproachSpeed * timeToFuture) * np.math.cos(angleOfApproach)) ** 2)

            if draw == True:
                cv2.drawMarker(img, (int(futureX), int(futureY)), color=(0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=2)
                cv2.line(img, (lmls[1], lmls[2]), (int(futureX), int(futureY)), (255, 255, 255), 2)
                cv2.line(img, (lmrs[1], lmrs[2]), (int(futureX), int(futureY)), (255, 255, 255), 2)

        elif (angleOfApproach > 90) and (angleOfApproach < 180) and (centerXApproachSpeed < 0) and (centerYApproachSpeed < 0):
            futureX = init[0] - np.math.sqrt(
                ((centerXApproachSpeed * timeToFuture) * np.math.cos(angleOfApproach)) ** 2)
            futureY = init[1] - np.math.sqrt(
                ((centerYApproachSpeed * timeToFuture) * np.math.cos(angleOfApproach)) ** 2)

            if draw == True:
                cv2.drawMarker(img, (int(futureX), int(futureY)), color=(0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=2)
                cv2.line(img, (lmls[1], lmls[2]), (int(futureX), int(futureY)), (255, 255, 255), 2)
                cv2.line(img, (lmrs[1], lmrs[2]), (int(futureX), int(futureY)), (255, 255, 255), 2)

        elif (((angleOfApproach > 0 - err) and (angleOfApproach < 0 + err)) or ((angleOfApproach > 180 - err) and (angleOfApproach < 180 + err))) and (centerXApproachSpeed > 0) and ((centerYApproachSpeed > 0 - err) and (centerYApproachSpeed < 0 + err)):
            futureX = init[0] + np.math.sqrt(
                ((centerXApproachSpeed * timeToFuture) * np.math.cos(angleOfApproach)) ** 2)
            futureY = init[1]

            if draw == True:
                cv2.drawMarker(img, (int(futureX), int(futureY)), color=(0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=2)
                cv2.line(img, (lmls[1], lmls[2]), (int(futureX), int(futureY)), (255, 255, 255), 2)
                cv2.line(img, (lmrs[1], lmrs[2]), (int(futureX), int(futureY)), (255, 255, 255), 2)

        elif (((angleOfApproach > 0 - err) and (angleOfApproach < 0 + err)) or ((angleOfApproach > 180 - err) and (angleOfApproach < 190 - err))) and (centerXApproachSpeed < 0) and ((centerYApproachSpeed > 0 - err) and (centerYApproachSpeed < 0 + err)):
            futureX = init[0] - np.math.sqrt(
                ((centerXApproachSpeed * timeToFuture) * np.math.cos(angleOfApproach)) ** 2)
            futureY = init[1]

            if draw == True:
                cv2.drawMarker(img, (int(futureX), int(futureY)), color=(0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=2)
                cv2.line(img, (lmls[1], lmls[2]), (int(futureX), int(futureY)), (255, 255, 255), 2)
                cv2.line(img, (lmrs[1], lmrs[2]), (int(futureX), int(futureY)), (255, 255, 255), 2)

        elif ((angleOfApproach > 90 - err) and (angleOfApproach < 90 + err)) and ((centerXApproachSpeed > 0 - err) and (centerXApproachSpeed < 0 + err)) and (centerYApproachSpeed > 0):
            futureX = init[0]
            futureY = init[1] - np.math.sqrt(
                ((centerYApproachSpeed * timeToFuture) * np.math.cos(angleOfApproach)) ** 2)

            if draw == True:
                cv2.drawMarker(img, (int(futureX), int(futureY)), color=(0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=2)
                cv2.line(img, (lmls[1], lmls[2]), (int(futureX), int(futureY)), (255, 255, 255), 2)
                cv2.line(img, (lmrs[1], lmrs[2]), (int(futureX), int(futureY)), (255, 255, 255), 2)

        elif ((angleOfApproach > 90 - err) and (angleOfApproach < 90 + err)) and ((centerXApproachSpeed > 0 - err) and (centerXApproachSpeed < 0 + err)) and (centerYApproachSpeed < 0):
            futureX = init[0]
            futureY = init[1] + np.math.sqrt(
                ((centerYApproachSpeed * timeToFuture) * np.math.cos(angleOfApproach)) ** 2)

            if draw == True:
                cv2.drawMarker(img, (int(futureX), int(futureY)), color=(0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=2)
                cv2.line(img, (lmls[1], lmls[2]), (int(futureX), int(futureY)), (255, 255, 255), 2)
                cv2.line(img, (lmrs[1], lmrs[2]), (int(futureX), int(futureY)), (255, 255, 255), 2)

        else:
            futureX = init[0]
            futureY = init[1]

        return futureX, futureY

def main():

    pathOverlay = cv2.imread('resources/overlays/pathOverlayBlack.png')
    cap = cv2.VideoCapture('resources/testVideos/test0.mp4')
    cap.set(3, 768)
    cap.set(4, 432)

    detector = PoseDetector()

    # FPS initializations
    curTime = time.time()  # start time
    fps = 0
    frameNumber = 0

    # general initializations
    lastXCenterDisplacement = 0
    lastYCenterDisplacement = 0
    occupiedHeight = 0
    xCenterDisplacement = 0
    centerXApproachSpeed = 0
    centerYApproachSpeed = 0
    angleOfApproach = 0
    lastDeltaY = 0

    # future definitions
    futureX = 0
    futureY = 0
    timeToFuture = 1 # all collision predictions are made for these many time units into the future
    futureErrorThresholds = 10
    # threshold = 10 # collision threshold for futureDeltaY

    # past definitions
    currentFrame = 0
    frameWindow = 2

    # filter settings
    angleFilter = StreamingMovingAverage(10)
    xFilter = StreamingMovingAverage(10)
    yFilter = StreamingMovingAverage(10)

    # data collection settings & initialization
    df = pd.DataFrame()
    df.index.name = 'frameNumber'
    windowSizes = []
    errorThresholds = []
    timesToFuture = []
    leftShoulderX = []
    leftShoulderY = []
    rightShoulderX = []
    rightShoulderY = []
    occupiedHeights = []
    anglesOfApproach = []
    XframeSpeeds = []
    YFrameSpeeds = []
    predictedX = []
    predictedY = []

    while True:

        success, img = cap.read()

        if (cv2.waitKey(1) == ord('q')) or (not success):
            break

        img = cv2.resize(img, (768, 432))
        img = cv2.flip(img, 1)

        img = detector.findPose(img)

        # resizing & adding path overlay
        pathOverlay = cv2.resize(pathOverlay, (768, 432))
        img = cv2.addWeighted(img,0.7,pathOverlay,0.3,0)

        lmList, bboxInfo = detector.findPosition(img, draw=False, bboxWithHands=False)

        if bboxInfo:

            # finding the center of the target pose
            center = bboxInfo["center"]
            cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)

            # finding the difference between highest landmark and lowest landmark in pixels
            yLocations = []
            for lm in lmList:
                yLocations.append(lm[2])
                if (lm[0] == 12):
                    lmrs = lm
                elif (lm[0] == 11):
                    lmls = lm
            deltaY = max(yLocations) - min(yLocations)
            occupiedHeight = deltaY / 432 # indicator of pedestrian's apparent height ( in terms of percentage of Y axis occupied)

            xCenterDisplacement = center[0]
            yCenterDisplacement = center[1]
            centerXApproachSpeed = (xCenterDisplacement - lastXCenterDisplacement) * fps
            centerYApproachSpeed = (yCenterDisplacement - lastYCenterDisplacement) * fps

            # angle of approach reporting currently accurate only between the range of 30 and 160 degrees
            angleOfApproach = detector.angleOfOrientation(lmls, lmrs)

            # filtering, predicting & drawing the future location of the target pedestrian
            initX = xFilter.process(center[0])
            initY = yFilter.process(center[1])
            futureX, futureY = detector.futureXY(img, lmls, lmrs, (initX, initY), angleOfApproach, centerXApproachSpeed, centerYApproachSpeed, timeToFuture, futureErrorThresholds)

            cv2.circle(img, (lmls[1], lmls[2]), 5, (255, 255, 255), cv2.FILLED)
            cv2.circle(img, (lmrs[1], lmrs[2]), 5, (255, 255, 255), cv2.FILLED)

            # displacement updation, will cause first n frames in window to be highly inaccurate
            currentFrame += 1
            frameNumber += 1
            if (currentFrame == frameWindow) and (frameNumber > frameWindow):
                lastXCenterDisplacement = xCenterDisplacement
                lastYCenterDisplacement = yCenterDisplacement
                lastDeltaY = deltaY
                currentFrame = 0

            elif (frameNumber < frameWindow):
                lastXCenterDisplacement = xCenterDisplacement
                lastYCenterDisplacement = yCenterDisplacement
                lastDeltaY = deltaY
                currentFrame = 0

            else:
                pass

            # data collection
            windowSizes.append(frameWindow)
            errorThresholds.append(futureErrorThresholds)
            timesToFuture.append(timeToFuture)
            leftShoulderX.append(lmls[1])
            leftShoulderY.append(lmls[2])
            rightShoulderX.append(lmrs[1])
            rightShoulderY.append(lmrs[2])
            occupiedHeights.append(occupiedHeight)
            anglesOfApproach.append(angleOfApproach)
            XframeSpeeds.append(centerXApproachSpeed)
            YFrameSpeeds.append(centerYApproachSpeed)
            predictedX.append(futureX)
            predictedY.append(futureY)

        # FPS calculation
        fps = 1 / (time.time() - curTime)
        curTime = time.time()
        cv2.putText(img, '{0:.2f}'.format(fps), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img, '{0:.2f}'.format((1 / fps) * 1000), (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow("img", img)
        #time.sleep(0.2)

    # editing & saving the dataframe in .csv format
    df['windowSize'] = windowSizes
    df['XYError'] = errorThresholds
    df['timeToFuture'] = timesToFuture
    df['leftShoulderX'] = leftShoulderX
    df['leftShoulderY'] = leftShoulderY
    df['rightShoulderX'] = rightShoulderX
    df['rightShoulderY'] = rightShoulderY
    df['angleOfApproach'] = anglesOfApproach
    df['XFrameSpeed'] = XframeSpeeds
    df['YFrameSpeed'] = YFrameSpeeds
    df['occupiedHeights'] = occupiedHeights
    df['predictedX'] = predictedX
    df['predictedY'] = predictedY
    df.to_csv('data.csv')

    # releasing & destroying windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()