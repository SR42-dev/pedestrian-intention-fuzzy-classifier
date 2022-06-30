import cv2
import math
import time
import numpy as np
import mediapipe as mp


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
                     bbox[1] + bbox[3] // 2

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
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                             math.atan2(y1 - y2, x1 - x2))
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

    def futureXY(self, img, init, angleOfApproach, centerApproachSpeed, timeToFuture, drawPos=True, drawPath=False):

        futureX = init[0] + ((centerApproachSpeed * timeToFuture) * np.math.cos(angleOfApproach))
        futureY = init[1] # currently doesn't account for horizontal terrain changes on the robot's path

        if drawPos == True :

            # future location on frame
            cv2.drawMarker(img, (int(futureX), int(futureY)), color=(0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=2)

        if drawPath == True:

            # predicted path line
            cv2.line(img, init, (int(futureX), int(futureY)), (255, 0, 255), 3)

        return futureX, futureY

def main():

    # capture frames from a camera
    cap = cv2.VideoCapture('resources/testVideos/test0.mp4')
    cap.set(3, 768)
    cap.set(4, 432)

    detector = PoseDetector()

    curTime = time.time()  # start time
    fps = 0

    lastXCenterDisplacement = 0
    occupiedHeight = 0
    xCenterDisplacement = 0
    centerApproachSpeed = 0
    angleOfApproach = 0
    lastDeltaY = 0

    futureX = 0
    futureY = 0
    timeToFuture = 1 # all collision predictions are made for these many time units into the future
    threshold = 10 # collision threshold for futureDeltaY

    while True:

        success, img = cap.read()
        img = detector.findPose(img)
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
            occupiedHeight = deltaY / cv2.getWindowImageRect('img')[3]  # target variable 1
            xCenterDisplacement = (cv2.getWindowImageRect('img')[2] / 2) - center[0]  # target variable 2
            centerApproachSpeed = (xCenterDisplacement - lastXCenterDisplacement) * fps  # target variable 3

            # relative bot approach speed indicator value
            botApproachSpeed = (deltaY - lastDeltaY) * fps

            # angle of approach reporting currently accurate only between the range of 30 and 160 degrees
            angleOfApproach = detector.angleOfOrientation(lmls, lmrs) # target variable 4
            # print(lmls)
            # print(lmrs)

            # predicting & drawing the future location of the target pedestrian
            futureX, futureY = detector.futureXY(img, center, angleOfApproach, centerApproachSpeed, timeToFuture, drawPath=True)

            # collision prediction wrt botApproachSpeed
            futureDeltaY = botApproachSpeed * timeToFuture  # predicted closeness of the pedestrian to the bot in the future
            if (futureDeltaY > threshold) \
                    and ((futureX > bboxInfo['bbox'][0]) and (futureX < (bboxInfo['bbox'][0] + bboxInfo['bbox'][2])))\
                    and ((futureX > bboxInfo['bbox'][1]) and (futureX < (bboxInfo['bbox'][1] + bboxInfo['bbox'][3]))):
                cv2.putText(img, 'Collision imminent', (50, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                            (255, 255, 0), 5)

            # cv2.putText(img, 'Angle : {0:.2f}'.format(angleOfApproach), (50, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 5)
            cv2.circle(img, (lmls[1], lmls[2]), 5, (255, 255, 255), cv2.FILLED)
            cv2.circle(img, (lmrs[1], lmrs[2]), 5, (255, 255, 255), cv2.FILLED)

            lastXCenterDisplacement = xCenterDisplacement  # displacement updation
            lastDeltaY = deltaY

        # FPS calculation
        fps = 1 / (time.time() - curTime)
        curTime = time.time()
        cv2.putText(img, '{0:.2f}'.format(fps), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 1, cv2.LINE_AA)

        # data output
        print('--------------\n' + 'FPS = {0:.2f}\n'.format(fps)
              + 'yBigness (%) = {0:.2f}\n'.format(occupiedHeight)
              + 'Displacement from center (px) = {0:.2f}\n'.format(xCenterDisplacement)
              + 'Speed of center approach (px/s) = {0:.2f}\n'.format(centerApproachSpeed)
              + 'Angle of approach (px/s) = {0:.2f}\n\n'.format(angleOfApproach)
              + 'Time to future (s) = {0:.2f}\n'.format(timeToFuture)
              + 'Predicted future location (px, px) = {0:.2f}, '.format(futureX) + '{0:.2f} \n'.format(futureY)
              + '--------------\n')

        cv2.imshow("img", img)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()