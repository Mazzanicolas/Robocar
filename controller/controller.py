import requests
import cv2
import urllib.request
import numpy as np
# import numpy as np
# import matplotlib.pyplot as plt


class Controller:

    def __init__(self, ip='192.168.0.110', port='8887'):
        """
        Inputs:
            * ip: (default is 192.168.0.110)
            * port: (default is 8887)
        """
        self.url = 'http://'+ip+':'+port

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
        while True:
            bytes += stream.read(1024)
            a = bytes.find(b'\xff\xd8')
            b = bytes.find(b'\xff\xd9')
            if a != -1 and b != -1:
                jpg = bytes[a:b+2]
                bytes = bytes[b+2:]
                i = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                cv2.imshow('Robocar Video Stream', i)
                if cv2.waitKey(1) == 27:
                    exit(0)  
    
    def get_video_frame(self):
        """
        Get camera video frame
        """
        pass
 