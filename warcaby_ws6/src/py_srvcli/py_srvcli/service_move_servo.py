import rclpy
from rclpy.node import Node
from img_check.srv import MoveServo 
import subprocess
import os

class MoveServoService(Node):
    def __init__(self):
        super().__init__('move_servo_service')
        self.srv = self.create_service(MoveServo, 'service_move_servo', self.callback)

    def callback(self, request, response):
        #ścieżka
        sciezka = os.path.expanduser('~/Desktop/warcaby_ws2/src/py_srvcli/py_srvcli/SCServo_Linux_220329/SCServo_Linux/examples/SMS_STS/MoveServo')
        plik = "./MoveServo"
        port = "/dev/ttyACM0"
        angle = str(request.angle)

        #Sprawdzenie, czy plik istnieje
        full_path = os.path.join(sciezka, plik)
        if not os.path.isfile(full_path):
            response.output = ""
            response.error = f"Nie znaleziono pliku: {full_path}"
            self.get_logger().error(response.error)
            return response

        #Uruchomienie programu
        result = subprocess.run([plik, port, angle], cwd=sciezka, capture_output=True, text=True)

        response.output = result.stdout
        response.error = result.stderr
        self.get_logger().info(f"Uruchomiono MoveServo z kątem {angle}")
        if result.stderr:
            self.get_logger().error(result.stderr)

        return response

def main(args=None):
    rclpy.init(args=args)
    node = MoveServoService()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
