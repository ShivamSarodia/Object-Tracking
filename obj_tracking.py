import numpy as np
import matplotlib.path as path
import cv2

class Rect:
    def __init__(self, p1, p2, p3 = None, p4 = None):
        """ If no p3 or p4, makes upright rect """

        if p3 is None and p4 is None:
            self.set_upright(p1, p2)
        else:
            self.p3 = p3
            self.p4 = p4
            self.__checkRect()

    def set_upright(self, p1, p2):
        """ Set this rectangle to an upright rect with p1 and p2 opposites"""
        self.p1 = p1
        self.p2 = p2
        self.p3 = (self.p1[0], self.p2[1])
        self.p4 = (self.p2[0], self.p1[0])

    def make_mask(self, shape):
        rect_mask = np.zeros(shape, np.uint8)
        rect_mask = cv2.fillConvexPoly(rect_mask,
                                       np.array([self.p1, self.p2, self.p3, self.p4]),
                                       1)
        return rect_mask
        
    def __approx_equal(self, x, y, eps = 0.02):
        if x != 0:
            return abs((x - y)/x) < eps
        elif y != 0:
            return abs((x - y)/y) < eps
        else:
            return True

    def __checkRect(self):
        checks = [__approx_equal(abs(p1[0] - p3[0]), abs(p2[0] - p4[0])),
                  __approx_equal(abs(p1[1] - p3[1]), abs(p2[1] - p4[1])),
                  __approx_equal(abs(p2[0] - p3[0]), abs(p1[0] - p3[0])),
                  __approx_equal(abs(p2[1] - p3[1]), abs(p1[1] - p3[1]))]
        return all(checks)

    def translate(self, transvect):
        self.p1 = (self.p1[0] + transvect[0], self.p1[1] + transvect[1])
        self.p2 = (self.p2[0] + transvect[0], self.p2[1] + transvect[1])
        self.p3 = (self.p3[0] + transvect[0], self.p3[1] + transvect[1])
        self.p4 = (self.p4[0] + transvect[0], self.p4[1] + transvect[1])
                
    def get_p1(self):
        return self.p1
    def get_p2(self):
        return self.p2
    def get_p3(self):
        return self.p3
    def get_p4(self):
        return self.p4        

class Select:
    """ Class for handling the region select process """
    def __init__(self):
        self.NONE = 0 #no rectangle has been selected
        self.JUST_CLEARED = 4 #user just clicked to start a new rectangle
        self.SELECTING = 1 #user is currently dragging a new rectangle
        self.JUST_SELECTED = 2 #user just released to end the new rectangle
        self.SELECTED = 3 #user has already selected a rectangle

        self.status = self.NONE

        self.first_point = None #one point of the rectangle
        self.rect = None
    
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.first_point = (x,y)
            self.rect = Rect(self.first_point, (x, y))
            self.status = self.JUST_CLEARED
            print("Just cleared")

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.status == self.JUST_CLEARED or self.status == self.SELECTING:
                self.rect.set_upright(self.first_point, (x, y))
        
        elif event == cv2.EVENT_LBUTTONUP:
            if self.status == self.JUST_CLEARED or self.status == self.SELECTING:
                self.rect.set_upright(self.first_point, (x, y))
                self.status = self.JUST_SELECTED
                print("Just selected")

    def get_status(self):
        retval = self.status
        
        if self.status == self.JUST_SELECTED:
            self.status = self.SELECTED

        elif self.status == self.JUST_CLEARED:
            self.status = self.SELECTING
            
        return retval

    def get_rect(self):
        return self.rect
    
class Display:
    """ Class for handling the display process """

    circle_params = dict( radius = 5,
                          color = 0,
                          thickness = -1 )

    def __init__(self, win):
        self.win = win
    
    def tick(self, frame, rect, points):
        if rect is not None:
            frame = cv2.rectangle(frame, rect.get_p1(), rect.get_p2(), 0) #draw the rectangle

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

    def __init__(self, frame, rect):
        self.rect = rect

        self.reload_points(frame)

    def tick(self, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        all_points, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, frame_gray, self.points, None, **self.lk_params)

        #Select good points
        good_points = all_points[st == 1]

        self.old_gray = frame_gray.copy()
        self.points = good_points.reshape(-1, 1, 2)
        
    def reload_points(self, frame):
        """Redraws suitable points in the range being observed"""

        self.old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Make a rectangular mask for the goodFeatures func -- todo improve
        rect_mask = self.rect.make_mask(frame.shape[0:2])

        # px_max = max(self.p1[0], self.p2[0])
        # py_max = max(self.p1[1], self.p2[1])
        # px_min = min(self.p1[0], self.p2[0])
        # py_min = min(self.p1[1], self.p2[1])
        # rect_mask[py_min:py_max, px_min:px_max] = ones_array[py_min:py_max, px_min:px_max]

        self.points = cv2.goodFeaturesToTrack(self.old_gray, mask = rect_mask, **self.feature_params)
        
    def get_points(self):
        return self.points

    def get_rect(self):
        return self.rect
    
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
        running = display.tick(frame, None, [])

    elif status == select.JUST_CLEARED or status == select.SELECTING:
        #rectangle being selected
        tracker = None #clear the tracker
        rect = select.get_rect()
        running = display.tick(frame, rect, [])

    elif status == select.JUST_SELECTED or status == select.SELECTED:
        #rectangle has been selected
        if status == select.JUST_SELECTED: #if new selection, create new Tracker
            tracker = Tracker(frame, select.get_rect())
        else: #if old selection, tick the tracker
            tracker.tick(frame)
            

        rect = tracker.get_rect()
        points = tracker.get_points()
        if points is None: points = [] #covers for when points is None on close

        running = display.tick(frame, rect, points)
        
    else: #shouldn't happen
        display.tick(frame, None, None, [])
            
cv2.destroyAllWindows()
cap.release()
