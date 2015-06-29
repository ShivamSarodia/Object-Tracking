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

    circle_params = dict( radius = 5,
                          color = 0,
                          thickness = -1 )

    def __init__(self, win):
        self.win = win
    
    def tick(self, frame, p1, p2, points):
        frame = cv2.rectangle(frame, p1, p2, 0) #draw the rectangle

        for point in points: #draw the points
            frame = cv2.circle(frame, (point[0][0], point[0][1]), **self.circle_params)

        cv2.imshow(self.win, frame) #show the frame

        if cv2.waitKey(10) == ord("q"):
            return False

        return True

class Tracker:
    """ Class for handling the tracking process """

    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.1,
                           minDistance = 7,
                           blockSize = 7 )
    
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def __init__(self, frame, p1, p2):
        self.p1 = p1
        self.p2 = p2
        
        self.old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #make rectangular mask for the image -- todo find better way
        rect_mask = np.zeros(frame.shape, np.uint8)
        px_max = max(p1[0], p2[0])
        py_max = max(p1[1], p2[1])
        px_min = min(p1[0], p2[0])
        py_min = min(p1[1], p2[1])
        rect_mask[px_min:px_max, py_min:py_max] = np.ones(px_max - px_min, py_max - py_min)
        
        self.points = cv2.goodFeaturesToTrack(self.old_gray, mask = rect_mask, **self.feature_params)
        
    def tick(self, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        all_points, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, frame_gray, self.points, None, **self.lk_params)

        #Select good points
        good_new = all_points[st == 1]
        good_old = all_points[st == 0]

        self.old_gray = frame_gray.copy()
        self.points = good_new.reshape(-1, 1, 2)

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
tracker = None

while running:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1) #flip the image horizontally because it's more intuitive

    status = select.get_status()

    if status == select.NONE:
        #no rectangle has been selected yet
        running = display.tick(frame, None, None, [])

    elif status == select.JUST_CLEARED or status == select.SELECTING:
        #rectangle being selected
        tracker = None #clear the tracker
        rect_p1 = select.get_p1()
        rect_p2 = select.get_p2()
        running = display.tick(frame, rect_p1, rect_p2, [])

    elif status == select.JUST_SELECTED or status == select.SELECTED:
        #rectangle has been selected
        if status == select.JUST_SELECTED: #if new selection, create new Tracker
            tracker = Tracker(frame, select.get_p1(), select.get_p2())
        else: #if old selection, tick the tracker
            tracker.tick(frame)
            
        rect_p1 = tracker.get_p1()
        rect_p2 = tracker.get_p2()
        points = tracker.get_points()
        running = display.tick(frame, rect_p1, rect_p2, points)
        
    else: #shouldn't happen
        display.tick(frame, None, None, [])
            
cv2.destroyAllWindows()
cap.release()
