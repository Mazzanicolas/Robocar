import requests

class Controller:

    def __init__(self, ip='192.168.0.110', port='8887'):
        self.url = 'http://'+ip+':'+port+'/drive'

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
        request = requests.post(self.url, json={
            "angle":angle,
            "throttle":throttle,
            "drive_mode":driver_mode,
            "recording":recording
        })
        return request.status_code
  

if __name__ == "__main__":
    ip   = '192.168.0.110'
    port = '8887'
    controller = Controller(ip, port)
    result = controller.drive(0,0)
    print(result)