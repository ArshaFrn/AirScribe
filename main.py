import cv2
import numpy as np
import mediapipe as mp
import time
import math
import datetime

# curve fitting for smoothing
def catmullRomSpline(p0, p1, p2, p3, numPoints=15):
    points = []
    for t in np.linspace(0, 1, numPoints):
        t2 = t * t
        t3 = t2 * t

        x = 0.5 * ((2 * p1[0]) +
                   (-p0[0] + p2[0]) * t +
                   (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 +
                   (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3)
        y = 0.5 * ((2 * p1[1]) +
                   (-p0[1] + p2[1]) * t +
                   (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 +
                   (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3)
        points.append((int(x), int(y)))
    return points

# Sci-fi background grid
def drawSciFiBackground(image):
    h, w, _ = image.shape
    image[:] = (10, 0, 10)
    gridColor = (30, 0, 30)
    for i in range(0, w, 50):
        cv2.line(image, (i, 0), (i, h), gridColor, 1)
    for i in range(0, h, 50):
        cv2.line(image, (0, i), (w, i), gridColor, 1)

# spline stroke drawing
def drawSplineStroke(points):
    if not points or len(points) < 4:
        return points

    points = [points[0]] + points + [points[-1]]

    curvePoints = []
    for i in range(len(points) - 3):
        p0 = points[i]
        p1 = points[i + 1]
        p2 = points[i + 2]
        p3 = points[i + 3]
        curvePoints.extend(catmullRomSpline(p0, p1, p2, p3))

    return curvePoints


def main():
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8, max_num_hands=1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 120)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cv2.namedWindow('AirScribe', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('AirScribe', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    drawColor = (0, 128, 255)
    penThickness = 15
    shapes = []

    drawingMode = 'draw'
    shapeStartPos = None
    currentStroke = []

    smoothedPos = None
    smoothingFactor = 0.2
    isDrawing = False
    pinchCounter = 0
    unpinchCounter = 0
    handLostCounter = 0
    pinchConfirmFrames = 2
    handLostTolerance = 10

    targetFps = 120
    frameTime = 1.0 / targetFps
    lastFrameTime = time.time()

    while cap.isOpened():
        currentTime = time.time()
        timeSinceLast = currentTime - lastFrameTime
        if timeSinceLast < frameTime:
            time.sleep(0.001)
            continue
        lastFrameTime = currentTime

        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.flip(image, 1)
        h, w, _ = image.shape
        drawingCanvas = np.zeros((h, w, 3), dtype=np.uint8)
        drawSciFiBackground(drawingCanvas)

        for shape in shapes:
            if shape['type'] == 'line':
                cv2.line(drawingCanvas, shape['start'], shape['end'], shape['color'], shape['thickness'])
            elif shape['type'] == 'rectangle':
                cv2.rectangle(drawingCanvas, shape['start'], shape['end'], shape['color'], shape['thickness'])
            elif shape['type'] == 'freedraw' and len(shape['points']) > 1:
                cv2.polylines(drawingCanvas, [np.array(shape['points'], dtype=np.int32)], False, shape['color'], shape['thickness'])

        rgbImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgbImage)

        if results.multi_hand_landmarks:
            handLostCounter = 0
            handLandmarks = results.multi_hand_landmarks[0]
            indexFingerTip = handLandmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
            thumbTip = handLandmarks.landmark[mpHands.HandLandmark.THUMB_TIP]
            rawFingerPos = (int(indexFingerTip.x * w), int(indexFingerTip.y * h))

            dx = indexFingerTip.x - thumbTip.x
            dy = indexFingerTip.y - thumbTip.y
            dz = indexFingerTip.z - thumbTip.z
            distance3D = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
            rawIsPinching = distance3D < 0.07

            if rawIsPinching:
                pinchCounter += 1
                unpinchCounter = 0
            else:
                unpinchCounter += 1
                pinchCounter = 0

            if smoothedPos is None:
                smoothedPos = rawFingerPos
            else:
                smoothedPos = (
                    int(smoothingFactor * rawFingerPos[0] + (1 - smoothingFactor) * smoothedPos[0]),
                    int(smoothingFactor * rawFingerPos[1] + (1 - smoothingFactor) * smoothedPos[1])
                )
        else:
            handLostCounter += 1
            pinchCounter = 0
            unpinchCounter = 0
            smoothedPos = None

        if not isDrawing and pinchCounter >= pinchConfirmFrames:
            isDrawing = True
            if drawingMode == 'draw':
                currentStroke = []
            else:
                shapeStartPos = smoothedPos

        if isDrawing and (unpinchCounter >= pinchConfirmFrames or handLostCounter >= handLostTolerance):
            isDrawing = False
            if drawingMode == 'draw' and len(currentStroke) > 1:
                shapes.append({'type': 'freedraw', 'points': drawSplineStroke(currentStroke), 'color': drawColor, 'thickness': penThickness})
            elif drawingMode == 'line' and smoothedPos:
                shapes.append({'type': 'line', 'start': shapeStartPos, 'end': smoothedPos, 'color': drawColor, 'thickness': penThickness})
            elif drawingMode == 'rectangle' and smoothedPos:
                shapes.append({'type': 'rectangle', 'start': shapeStartPos, 'end': smoothedPos, 'color': drawColor, 'thickness': penThickness})
            currentStroke = []
            shapeStartPos = None

        if isDrawing and smoothedPos:
            if drawingMode == 'draw':
                currentStroke.append(smoothedPos)
                cv2.polylines(drawingCanvas, [np.array(currentStroke, dtype=np.int32)], False, drawColor, penThickness)
            elif drawingMode == 'line' and shapeStartPos:
                cv2.line(drawingCanvas, shapeStartPos, smoothedPos, drawColor, penThickness)
            elif drawingMode == 'rectangle' and shapeStartPos:
                cv2.rectangle(drawingCanvas, shapeStartPos, smoothedPos, drawColor, penThickness)

        outputImage = drawingCanvas
        mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        outputImage = np.where(mask > 0, cv2.addWeighted(image, 0.4, outputImage, 0.6, 0), outputImage)

        if smoothedPos:
            cv2.circle(outputImage, smoothedPos, penThickness // 2 + 5, (255, 255, 255), 1)
            cv2.circle(outputImage, smoothedPos, 2, drawColor, -1)

        modeText = f"Mode: {drawingMode.upper()} (1: Draw, 2: Line, 3: Rect)"
        controlsText = "'+/-': Thickness | r,g,b: Color | c: Clear | u: Undo"

        cv2.putText(outputImage, modeText, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(outputImage, controlsText, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('AirScribe', outputImage)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            shapes = []
        elif key == ord('r'):
            drawColor = (50, 50, 255)
        elif key == ord('g'):
            drawColor = (50, 255, 50)
        elif key == ord('b'):
            drawColor = (255, 100, 50)
        elif key in [ord('+'), ord('=')]:
            penThickness = min(100, penThickness + 5)
        elif key == ord('-'):
            penThickness = max(5, penThickness - 5)
        elif key == ord('1'):
            drawingMode = 'draw'
        elif key == ord('2'):
            drawingMode = 'line'
        elif key == ord('3'):
            drawingMode = 'rectangle'
        elif key == ord('u'):
            if shapes:
                shapes.pop()
        elif key == ord('s'):
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"AirScribe_drawing_{timestamp}.png"
            cv2.imwrite(filename, drawingCanvas)
            cv2.putText(outputImage, f"Saved as {filename}", (w // 2 - 250, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)
            cv2.imshow('AirScribe', outputImage)
            cv2.waitKey(1000)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
