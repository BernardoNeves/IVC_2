import cv2
import numpy as np

# initialize HSV color range for green colored objects
thresholdLower = (50, 100, 100)
thresholdUpper = (70, 255, 255)

backSub = cv2.createBackgroundSubtractorMOG2(history = 100, varThreshold = 40)

# Start capturing the video from webcam
video_capture = cv2.VideoCapture(0)
frame_previous = None
roi = None
center_previous = None

def cam():
    global frame # global frame so it can be used in mouse_get_threshold()
    global frame_previous
    global roi
    if not video_capture.isOpened():
        video_capture.open(0)
    # Store the current frame of the video in the variable frame
    ret, frame = video_capture.read()
    # Flip the image to make it right
    frame = cv2.flip(frame,1)
    if roi is None:
        roi = cv2.selectROI('Camera Output', frame, False)    
    
    # Crop image
    frame_roi = frame[roi[1] : roi[1] + roi[3], roi[0] : roi[0] + roi[2]]
    
    # Convert the frame to HSV as it allows better segmentation.
    frame_hsv = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2HSV)
    # Blur the frame using a Gaussian Filter of kernel size 5, to remove excessive noise
    frame_blurred = cv2.GaussianBlur(frame_hsv, (5,5), 0)

    # Create a mask for the frame, showing threshold values
    frame_segmented = cv2.inRange(frame_blurred, thresholdLower, thresholdUpper)
    
    # 3. Set previous frame and continue if there is None
    if frame_previous is None:
        # First frame; there is no previous one yet
        frame_previous = frame_segmented
        return None
    
    
    # calculate difference and update previous frame
    frame_difference = backSub.apply(frame_segmented)
    # frame_difference = cv2.absdiff(src1=frame_previous, src2=frame_segmented)
    
    frame_previous = frame_segmented
    
    # 4. Dilute the image a bit to make differences more seeable; more suitable for contour detection
    frame_difference = cv2.dilate(frame_difference,(5,5), 1)
    
    # 5. Only take different areas that are different enough (>20 / 255)
    frame_thresholded = cv2.threshold(src=frame_difference, thresh=20, maxval=255, type=cv2.THRESH_BINARY)[1]
    
    # Erode the masked output to delete small white dots present in the masked image
    frame_eroded  = cv2.erode(frame_thresholded, None, 10)
    # Dilate the resultant image to restore our target
    frame_masked = cv2.dilate(frame_eroded, None, 10)

    # Draw a contour around the detected motion nad return it's center
    center = draw_contours(frame_roi, frame_masked)

    # Display the masked frame in a window located at (x,y) 
    show_output('Masked Output', frame_masked, 1275, 550) # 300, 200

    # Show the output frame in a window located at (x,y) 
    show_output('Camera Output',frame, 1275, 0) # 950, 200 
    cv2.setMouseCallback('Camera Output',mouse_get_threshold)

    if center is not None:
        paddle_x = center[0] * 610 / roi[2]
        return paddle_x

def show_output(title_output, frame_output, x, y):
    cv2.imshow(title_output,frame_output)
    cv2.moveWindow(title_output, x, y)

def draw_contours(frame_draw, frame_contours):
    
    # cv2.rectangle(frame_draw, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), (255, 0, 0), 2)
    cv2.rectangle(frame_draw, (0, 0), (roi[2]-1, roi[3]-1), (255, 0, 0), 2)
    
    _, frame_contours = cv2.threshold(frame_contours, 254, 255, cv2.THRESH_BINARY)
    # Find all contours in the masked image
    contours, _ = cv2.findContours(
        frame_contours.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    global center_previous
    # Define center of the object to be detected as None
    center = None

    # check if there's at least 1 object with the segmented color
    if len(contours) > 0:
        cv2.rectangle(frame_draw, (0, 0), (roi[2]-1, roi[3]-1), (0, 255, 0), 2)
        
        # Find the contour with maximum area
        contours_max = max(contours, key=cv2.contourArea)

        #  rotated bounding rectangle (https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html)
        rect = cv2.minAreaRect(contours_max)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(frame_draw, [box], 0, (0, 0, 255), 2)

        # Calculate the centroid of the object
        # "formula from (https://learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/)"
        M = cv2.moments(contours_max)
        center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
        center_previous = center
    else:
        center = center_previous
    return center


def mouse_get_threshold(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN: # checks mouse left button down condition
        # convert rgb to hsv format
        colorsHSV = cv2.cvtColor(np.uint8([[[frame[y,x,0] ,frame[y,x,1],frame[y,x,2] ]]]),cv2.COLOR_BGR2HSV)
        
        # create a threshold based on the color values
        tempLower = colorsHSV[0][0][0]  - 10, 100, 100
        tempUpper = colorsHSV[0][0][0] + 10, 255, 255
        
        global thresholdLower
        global thresholdUpper
        # set the threshold
        thresholdLower = np.array(tempLower)
        thresholdUpper = np.array(tempUpper)
    
def main(): # used for debugging the segmentation without the breakout game running
    while(1):
        cam()
        if cv2.waitKey(20) & 0xFF == 27: #  wait for Esc key
            break
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__': # only run main() if executing directly from it's file and not being imported
    main()