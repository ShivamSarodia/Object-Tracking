import numpy as np
import cv2

class Select:
    """ Class for handling the rectangle select process """
    def __init__(self):
        self.NONE = 0 #no rectangle has been selected
        self.JUST_CLEARED = 4 #user just clicked to start a new rectangle
        self.SELECTING = 1 #user is currently dragging a new rectangle
        self.JUST_SELECTED = 2 #user just released to end the new rectangle
        self.SELECTED = 3 #user has already selected a rectangle

        self.status = self.NONE

        self.p1 = None #one point of the rectangle
        self.p2 = None #opposite corner of the rectangle
    
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.p1 = (x,y)
            self.p2 = (x,y)
            self.status = self.JUST_CLEARED
            print("Just cleared")

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.status == self.JUST_CLEARED or self.status == self.SELECTING:
                self.p2 = (x,y)
        
        elif event == cv2.EVENT_LBUTTONUP:
            if self.status == self.JUST_CLEARED or self.status == self.SELECTING:
                self.p2 = (x,y)
                self.status = self.JUST_SELECTED
                print("Just selected")

    def get_status(self):
        retval = self.status
        
        if self.status == self.JUST_SELECTED:
            self.status = self.SELECTED

        elif self.status == self.JUST_CLEARED:
            self.status = self.SELECTING
            
        return retval

    def get_p1(self):
        return self.p1

    def get_p2(self):
        return self.p2

class Display:
    """ Class for handling the display process """

    def __init__(self, win):
        self.win = win
    
    def tick(self, frame, p1, p2, points):
        frame = cv2.rectangle(frame, p1, p2, 0)
        cv2.imshow(self.win, frame)

        if cv2.waitKey(10) == ord("q"):
            return False

        return True

class Tracker:
    """ Class for handling the tracking process """

    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7 )
    
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def __init__(self, frame, p1, p2):
        self.p1 = p1
        self.p2 = p2
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.points = cv2.goodFeaturesToTrack(gray, mask = None, **feature_params)

    def tick(self, frame):
        pass

    def get_points(self):
        return self.points

    def get_p1(self):
        return self.p1

    def get_p2(self):
        return self.p2

    
win_name = "Display"

display = Display(win_name)
select = Select()

# Set up window and callback
cv2.namedWindow(win_name)
cv2.setMouseCallback(win_name, select.mouse_callback)
cap = cv2.VideoCapture(0)

for i in range(0, 10): cap.read() #run through a few frames because the first couple are black

running = True
while running:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1) #flip the image horizontally because it's more intuitive

    status = select.get_status()
    tracker = None

    if status == select.NONE:
        #no rectangle has been selected yet
        running = display.tick(frame, None, None, None)

    elif status == select.JUST_CLEARED or status == select.SELECTING:
        #rectangle being selected
        tracker = None #clear the tracker
        rect_p1 = select.get_p1()
        rect_p2 = select.get_p2()
        running = display.tick(frame, rect_p1, rect_p2, None)

    elif status == select.JUST_SELECTED or status == select.SELECTED:
        #rectangle has been selected
        if status == select.JUST_SELECTED: #if new selection, create new Tracker
            tracker = Tracker(frame, select.get_p1(), select.get_p2())

        rect_p1 = tracker.get_p1()
        rect_p2 = tracker.get_p2()
        points = tracker.get_points()
        running = display.tick(frame, rect_p1, rect_p2, points)
        
    else: #shouldn't happen
        display.tick(frame, None, None, None)
            
cv2.destroyAllWindows()
cap.release()
