import requests
import cv2
import urllib.request
import numpy as np
# import numpy as np
import matplotlib.pyplot as plt


class Controller:

    def __init__(self, ip='192.168.0.110', port='8887'):
        """
        Inputs:
            * ip: (default is 192.168.0.110)
            * port: (default is 8887)
        """
        self.url = 'http://'+ip+':'+port
        self.color_filters = {
            'rl': 101,
            'gl': 0,
            'bl': 0,

            'ru': 255,
            'gu': 91,
            'bu': 91
            }
        self.config = {
            'apply_canny': 1,
            'apply_blur' : 1,
            'apply_hough': 1
        }
        self.canny_config = {
            'threshold1': 50,
            'threshold2': 150
        }#[50, 150]
        self.blur_config = {
            'kernel': 3
            }
        self.hough_config = {
            'threshold':     20,
            'minLineLength': 20,
            'maxLineGap':    300
        }
        #[20, 20, 300]

    def drive(self, angle=0, throttle=0, driver_mode="user", recording=False):
        """
        Accelerates the car with a given acceleration and angle.
        Inputs:
            * angle:
            * throttle:
            * driver_mode:
            * recording:
        Outputs:
            HTTP status codes
        """
        request = requests.post(self.url+'/drive', json={
            "angle":angle,
            "throttle":throttle,
            "drive_mode":driver_mode,
            "recording":recording
        })
        return request.status_code
    
    def get_video(self):
        """
        Car video stream
        """
        stream = urllib.request.urlopen(self.url+'/video')
        bytes = b''
        ###################### WEBCAM SETUP ###########################
        # cap = cv2.VideoCapture(0)                                   ###
        cv2.namedWindow('Robocar Cam' , cv2.WINDOW_NORMAL)          ###
        cv2.resizeWindow('Robocar Cam', 500, 700)                   ###
        cv2.namedWindow('Robocar Original' , cv2.WINDOW_NORMAL)     ###
        cv2.resizeWindow('Robocar Original', 700, 500)              ###
        ###################### WEBCAM SETUP ###########################
        cv2.createTrackbar('Red Lower  ', 'Robocar Cam', self.color_filters['rl'], 255, self.red_lower)
        cv2.createTrackbar('Green Lower', 'Robocar Cam', self.color_filters['gl'], 255, self.green_lower)
        cv2.createTrackbar('Blue Lower ', 'Robocar Cam', self.color_filters['bl'], 255, self.blue_lower)
        cv2.createTrackbar('Red Upper  ', 'Robocar Cam', self.color_filters['ru'], 255, self.red_upper)
        cv2.createTrackbar('Green Upper', 'Robocar Cam', self.color_filters['gu'], 255, self.green_upper)
        cv2.createTrackbar('Blue Upper ', 'Robocar Cam', self.color_filters['bu'], 255, self.blue_upper)

        cv2.createTrackbar('Apply Blur ', 'Robocar Cam', self.config['apply_blur'], 1, self.apply_blur)
        cv2.createTrackbar('Blur Kernel', 'Robocar Cam', self.blur_config['kernel'], 200, self.blur_kernel)

        cv2.createTrackbar('Apply Canny', 'Robocar Cam', self.config['apply_canny'], 1, self.apply_canny)
        cv2.createTrackbar('Canny Th 1 ', 'Robocar Cam', self.canny_config['threshold1'], 350, self.canny_threshold1)
        cv2.createTrackbar('Canny Th 2 ', 'Robocar Cam', self.canny_config['threshold2'], 350, self.canny_threshold2)

        cv2.createTrackbar('Apply Hough',  'Robocar Cam', self.config['apply_hough'], 1, self.apply_hough)
        cv2.createTrackbar('Threshold  ',  'Robocar Cam', self.hough_config['threshold'], 300, self.hough_threshold)
        cv2.createTrackbar('MinLineLength','Robocar Cam', self.hough_config['minLineLength'], 800, self.hough_minLineLength)
        cv2.createTrackbar('MaxLineGap  ', 'Robocar Cam', self.hough_config['maxLineGap'], 500, self.hough_maxLineGap)
        while(True):
            bytes += stream.read(1024)
            a = bytes.find(b'\xff\xd8')
            b = bytes.find(b'\xff\xd9')
            if a != -1 and b != -1:
                jpg = bytes[a:b+2]
                bytes = bytes[b+2:]
                frame = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                ###################### WEBCAM SETUP ###########################
                # ret, frame = cap.read()                                     ###
                ###################### WEBCAM SETUP ###########################
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ############### Color Mask ################
                masked_frame = self.mask_frame(rgb_frame)
                ################### Blur ##################
                if self.config['apply_blur'] == 1:
                    kernel = self.blur_config['kernel']
                    if kernel%2 == 0:
                        kernel += 1
                    masked_frame = cv2.GaussianBlur(masked_frame, (kernel, kernel), 0)
                ############### Canny Edges ###############
                if self.config['apply_canny'] == 1:
                    masked_frame = cv2.Canny(masked_frame, self.canny_config['threshold1'], self.canny_config['threshold2'])
                ############### Hough lines ###############
                if self.config['apply_hough'] == 1:
                    if self.config['apply_canny'] == 0:
                        gray_frame = cv2.cvtColor(masked_frame, cv2.COLOR_RGB2GRAY)
                        list_of_lines = self.hough_lines(gray_frame)    
                    else:
                        list_of_lines = self.hough_lines(masked_frame)
                    if list_of_lines is None:
                        list_of_lines = []
                    masked_frame = self.draw_lines(masked_frame, list_of_lines)
                    line = self.lane_lines(masked_frame, list_of_lines[:1])
                    if line is not None:
                        p1, p2 = line
                        x1, y1 = p1
                        x2, y2 = p2

                        x1 = 80
                        line = ((x1, y1), (x2, y2))
                        
                        rad = np.arctan2(y1-y2, x2-x1)
                        
                        frame = cv2.line(frame, *line, color=[0, 255, 0], thickness=10)
                        angle = rad * 180 / np.pi

                        if angle > 90 + 45:
                            angle = 90 + 45
                        if angle < 90 - 45:
                            angle = 90 - 45

                        wheels_angle = -(((angle - 45) / 90) * 2 - 1)

                        angle = np.round(angle, 2)
                        wheels_angle = np.round(wheels_angle, 4)

                        cv2.putText(frame, str(angle), (60,20), cv2.FONT_HERSHEY_PLAIN, 0.8, (0,0,0), thickness=2)
                        cv2.putText(frame, str(angle), (60,20), cv2.FONT_HERSHEY_PLAIN, 0.8, (255,255,255), thickness=1)

                        self.drive(wheels_angle, 0.25)
                    else:
                        line = ((0, 0), (1,1))
                    # line = self.make_line_points(list_of_lines[0])
                    frame = self.draw_lane_lines(frame, line)

                cv2.imshow('Robocar Cam', masked_frame)
                cv2.imshow('Robocar Original', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cv2.destroyAllWindows() 
        # return image
    
    def red_lower(self, new_lower_red):
        self.color_filters['rl'] = new_lower_red

    def green_lower(self, new_lower_green):
        self.color_filters['gl'] = new_lower_green
    
    def blue_lower(self, new_lower_blue):
        self.color_filters['bl'] = new_lower_blue
    
    def red_upper(self, new_upper_red):
        self.color_filters['ru'] = new_upper_red
    
    def green_upper(self, new_upper_green):
        self.color_filters['gu'] = new_upper_green
    
    def blue_upper(self, new_upper_blue):
        self.color_filters['bu'] = new_upper_blue
    
    def apply_canny(self, new_value):
        self.config['apply_canny'] = new_value

    def canny_threshold1(self, new_threshold1):
        self.canny_config['threshold1'] = new_threshold1

    def canny_threshold2(self, new_threshold2):
        self.canny_config['threshold2'] = new_threshold2
    
    def apply_blur(self, new_value):
        self.config['apply_blur'] = new_value

    def blur_kernel(self, new_kernel):
        self.blur_config['kernel'] = new_kernel

    def apply_hough(self, new_value):
        self.config['apply_hough'] = new_value
    
    def hough_threshold(self, new_threshold):
        self.hough_config['threshold'] = new_threshold

    def hough_minLineLength(self, new_minLineLength):
        self.hough_config['minLineLength'] = new_minLineLength
        
    def hough_maxLineGap(self, new_maxLineGap):
        self.hough_config['maxLineGap'] = new_maxLineGap

    def mask_frame(self, frame):         
        lower_rgb = np.uint8([self.color_filters[lower_bound_x] for lower_bound_x in ['rl','gl','bl']]) # Performance issue? Use only in debugg mode
        upper_rgb = np.uint8([self.color_filters[upper_bound_x] for upper_bound_x in ['ru','gu','bu']]) # Performance issue? Use only in debugg mode
        mask = cv2.inRange(frame, lower_rgb, upper_rgb)
        # combine the mask
        masked_frame = cv2.bitwise_and(frame, frame, mask = mask)
        return masked_frame

    def convert_hsv(self, image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    def convert_hls(self, image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    def bgr_to_rgb(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def select_white_yellow(self, image):
        converted = self.convert_hls(image)
        # white color mask
        lower = np.uint8([  0, 200,   0])
        upper = np.uint8([255, 255, 255])
        white_mask = cv2.inRange(converted, lower, upper)
        # yellow color mask
        lower = np.uint8([ 10,   0, 100])
        upper = np.uint8([ 40, 255, 255])
        yellow_mask = cv2.inRange(converted, lower, upper)
        # combine the mask
        mask = cv2.bitwise_or(white_mask, yellow_mask)
        return cv2.bitwise_and(image, image, mask = mask)

    def convert_gray_scale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    def apply_smoothing(self, image, kernel_size=3):
        """
        kernel_size must be postivie and odd
        """
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def detect_edges(self, image, low_threshold=50, high_threshold=150):
        return cv2.Canny(image, low_threshold, high_threshold)

    def hough_lines(self, image):
        """
        `image` should be the output of a Canny transform.
        
        Returns hough lines (not the image with lines)
        """
        return cv2.HoughLinesP(image, rho=1, theta=np.pi/180, threshold=self.hough_config['threshold'],
                                                              minLineLength=self.hough_config['minLineLength'],
                                                              maxLineGap=self.hough_config['maxLineGap'])
        # return cv2.HoughLinesP(image, rho=1, theta=np.pi/180, threshold=20, minLineLength=20, maxLineGap=300)


    def draw_lines(self, image, lines, color=[255, 0, 0], thickness=2, make_copy=True):
        # the lines returned by cv2.HoughLinesP has the shape (-1, 1, 4)
        if make_copy:
            image = np.copy(image) # don't want to modify the original
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(image, (x1, y1), (x2, y2), color, thickness)
        return image

    def average_slope_intercept(self, lines):
        the_lines    = [] # (slope, intercept)
        the_weights  = [] # (length,)
        
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x2==x1:
                    continue # ignore a vertical line
                slope = (y2-y1)/(x2-x1)
                intercept = y1 - slope*x1
                length = np.sqrt((y2-y1)**2+(x2-x1)**2)
       
                the_lines.append((slope, intercept))
                the_weights.append((length))
                
        # add more weight to longer lines    
        the_lane = np.dot(the_weights, the_lines) /np.sum(the_weights) if len(the_weights) >0 else None
        
        return the_lane # (slope, intercept), (slope, intercept)

    def average_slope_intercept2(self, lines):
        left_lines    = [] # (slope, intercept)
        left_weights  = [] # (length,)
        right_lines   = [] # (slope, intercept)
        right_weights = [] # (length,)
        
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x2==x1:
                    continue # ignore a vertical line
                slope = (y2-y1)/(x2-x1)
                intercept = y1 - slope*x1
                length = np.sqrt((y2-y1)**2+(x2-x1)**2)
                if slope < 0: # y is reversed in image
                    left_lines.append((slope, intercept))
                    left_weights.append((length))
                else:
                    right_lines.append((slope, intercept))
                    right_weights.append((length))
        
        # add more weight to longer lines    
        left_lane  = np.dot(left_weights,  left_lines) /np.sum(left_weights)  if len(left_weights) >0 else None
        right_lane = np.dot(right_weights, right_lines)/np.sum(right_weights) if len(right_weights)>0 else None
        
        return left_lane, right_lane # (slope, intercept), (slope, intercept)

    def make_line_points(self, y1, y2, line):
        """
        Convert a line represented in slope and intercept into pixel points
        """
        if line is None:
            return None
        
        slope, intercept = line
        
        if slope == 0:
            slope = 1

        # make sure everything is integer as cv2.line requires it
        x1 = int((y1 - intercept)/slope)
        x2 = int((y2 - intercept)/slope)
        y1 = int(y1)
        y2 = int(y2)
        
        return ((x1, y1), (x2, y2))

    def lane_lines(self, image, lines):
        lane =  self.average_slope_intercept(lines)
        
        y1 = image.shape[0] # bottom of the image
        y2 = y1*0.6         # slightly lower than the middle

        line = self.make_line_points(y1, y2, lane)
        
        return line
    
    def lane_lines2(self, image, lines):
        left_lane, right_lane = self.average_slope_intercept(lines)
        
        y1 = image.shape[0] # bottom of the image
        y2 = y1*0.6         # slightly lower than the middle

        left_line  = self.make_line_points(y1, y2, left_lane)
        right_line = self.make_line_points(y1, y2, right_lane)
        
        return left_line, right_line

    
    def draw_lane_lines(self, image, line, color=[255, 0, 0], thickness=20):
        # make a separate image to draw lines and combine with the orignal later
        line_image = np.zeros_like(image)
        if line is not None:
            cv2.line(line_image, *line,  color, thickness)
        # image1 * α + image2 * β + λ
        # image1 and image2 must be the same shape.
        return cv2.addWeighted(image, 1.0, line_image, 0.95, 0.0)
 
    def draw_lane_lines2(self, image, lines, color=[255, 0, 0], thickness=20):
        # make a separate image to draw lines and combine with the orignal later
        line_image = np.zeros_like(image)
        for line in lines:
            if line is not None:
                cv2.line(line_image, *line,  color, thickness)
        # image1 * α + image2 * β + λ
        # image1 and image2 must be the same shape.
        return cv2.addWeighted(image, 1.0, line_image, 0.95, 0.0)

    def get_video_frame(self):
        """
        Get camera video frame
        """
        pass
 
if __name__ == '__main__':
    robocar = Controller()
    robocar.get_video()