import numpy
import cv2

def vid_capture():
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        ret = cap.set(3, 320)
        ret = cap.set(4, 320)

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        colour = cv2.cvtColor(frame, 0)

        # Display the resulting frame
        cv2.imshow('frame', colour)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


