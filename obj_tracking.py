import numpy as np
import cv2 as cv

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
        if event == cv.EVENT_LBUTTONDOWN:
            self.p1 = (x,y)
            self.p2 = (x,y)
            self.status = self.JUST_CLEARED

        elif event == cv.EVENT_MOUSEMOVE:
            if self.status == self.JUST_CLEARED or self.status == self.SELECTING:
                self.p2 = (x,y)
        
        elif event == cv.EVENT_LBUTTONUP:
            if self.status == self.JUST_CLEARED or self.status == self.SELECTING:
                self.p2 = (x,y)
                self.status = self.JUST_SELECTED

    def get_status(self):
        retval = self.status
        
        if self.status == self.JUST_SELECTED:
            self.status = self.SELECTED

        elif self.status == self.JUST_CLEARED:
            self.status = self.CLEARED
            
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
        cv.imshow(self.win, frame)

        if cv.waitKey(1) == ord("q"):
            return False

        return True

class Tracker:
    """ Class for handling the tracking process """

    def __init__(self):
        pass

    
win_name = "Display"

display = Display(win_name)
select = Select()

cv.setMouseCallback(win_name, select.mouse_callback)
cap = cv.VideoCapture(0)
for i in range(0, 10): cap.read() #run through a few frames because the first couple are black

running = True
while running:
    ret, frame = cap.read()

    status = select.get_status()
    rect_p1 = select.get_p1()
    rect_p2 = select.get_p2()

    if status == select.NONE:
        running = display.tick(frame, None, None, None)

    elif status == status.JUST_CLEARED:
        running = display.tick(frame,rect_p1, rect_p2, None)

    elif status == select.SELECTING:
        running = display.tick(frame, rect_p1, rect_p2, None)

    elif status == select.JUST_SELECTED:
        running = display.tick(frame, rect_p1, rect_p2, None) #improve

    elif status == select.SELECTED:
        running = display.tick(frame, rect_p1, rect_p2, None) #improve
        
    
    
cv.destroyAllWindows()
cap.release()
