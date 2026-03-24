import rclpy
from rclpy.node import Node
from img_check.srv import ImgCheck

import cv2
import numpy as np
import os
import time
from datetime import datetime

from ament_index_python.packages import get_package_share_directory

import tensorflow as tf
from tensorflow import keras


#==============================
#PARAMETRY
#==============================
BOARD_SIZE = 10
WARP_SIZE = 1000
MODEL_IMG_H = 96
MODEL_IMG_W = 96

TOP_LEFT_IS_DARK = False


class ImgCheckService(Node):
    def __init__(self):
        super().__init__('img_check_service_nn')

        self.srv = self.create_service(
            ImgCheck,
            'img_check_service',
            self.handle_request
        )

        self.get_logger().info("img_check_service (NN) uruchomiony")

        #katalog pakietu
        self.base_dir = get_package_share_directory('py_srvcli')

        self.model_path = os.path.join(self.base_dir, "models", "warcaby_model.keras")
        self.class_names_path = os.path.join(self.base_dir, "models", "warcaby_class_names.npy")
        self.corners_path = os.path.join(self.base_dir, "config", "board_corners.npy")

        self.debug_dir = os.path.join(self.base_dir, "debug_nn")
        os.makedirs(self.debug_dir, exist_ok=True)

        #model
        self.model = keras.models.load_model(self.model_path)
        self.class_names = np.load(self.class_names_path, allow_pickle=True).tolist()

        self.name_to_code = {}
        for n in self.class_names:
            nl = n.lower()
            if "bial" in nl:
                self.name_to_code[n] = 2
            elif "poma" in nl:
                self.name_to_code[n] = 1
            else:
                self.name_to_code[n] = 0

        self.get_logger().info(f"Model: {self.model_path}")
        self.get_logger().info(f"Klasy: {self.class_names}")
        self.get_logger().info(f"Narożniki: {self.corners_path}")

    #==========================================================
    #KAMERA 
    #=========================================================
    def find_camera(self):
        for dev in range(4):
            cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
            if cap.isOpened():
                self.get_logger().info(f"Używam kamery /dev/video{dev}")
                return cap
            cap.release()
        raise RuntimeError("Nie udało się otworzyć kamery")


    def capture_board_image(self):
        cap = self.find_camera()

        #rozdzielczość
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        frame = None
        for _ in range(30):
            ret, frame = cap.read()
            if ret and frame is not None:
                break
            time.sleep(0.05)

        cap.release()

        if frame is None:
            raise RuntimeError("Nie udało się pobrać klatki z kamery")

        return frame

    #==========================================================
    #WARP
    #==========================================================
    def warp_board_with_saved_points(self, frame):
        pts_src = np.load(self.corners_path).astype(np.float32)

        pts_dst = np.float32([
            [0, 0],                        # LT
            [WARP_SIZE - 1, 0],             # RT
            [WARP_SIZE - 1, WARP_SIZE - 1], # RB
            [0, WARP_SIZE - 1]              # LB
        ])

        M = cv2.getPerspectiveTransform(pts_src, pts_dst)
        return cv2.warpPerspective(frame, M, (WARP_SIZE, WARP_SIZE))

    #==========================================================
    #DETEKCJA
    #==========================================================
    def detect_board_nn(self):
        frame = self.capture_board_image()
        board = self.warp_board_with_saved_points(frame)

        cell = WARP_SIZE // BOARD_SIZE
        annotated = board.copy()
        mat = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int32)

        patches = []
        coords = []

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                is_dark = ((r + c) % 2 == 1) if not TOP_LEFT_IS_DARK else ((r + c) % 2 == 0)
                if not is_dark:
                    continue

                y1, y2 = r * cell, (r + 1) * cell
                x1, x2 = c * cell, (c + 1) * cell

                patch = board[y1:y2, x1:x2]
                patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                patch = cv2.resize(patch, (MODEL_IMG_W, MODEL_IMG_H))
                patch = patch.astype(np.float32)

                patches.append(patch)
                coords.append((r, c, x1, y1, x2, y2))

        batch = np.stack(patches, axis=0)
        probs = self.model.predict(batch, verbose=0)
        preds = np.argmax(probs, axis=1)

        for i, (r, c, x1, y1, x2, y2) in enumerate(coords):
            cls = self.class_names[preds[i]]
            mat[r, c] = self.name_to_code[cls]

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(
                annotated,
                f"{cls[:4]} {probs[i, preds[i]]:.2f}",
                (x1 + 3, y1 + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1
            )

        return mat, annotated

    # ==========================================================
    # ROS SERVICE
    # ==========================================================
    def handle_request(self, request, response):
        try:
            mat, img = self.detect_board_nn()
            response.matrix = mat.flatten().astype(int).tolist()

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(os.path.join(self.debug_dir, f"wynik_{ts}.jpg"), img)

        except Exception as e:
            self.get_logger().error(str(e))
            response.matrix = [999] * 100

        return response


def main(args=None):
    rclpy.init(args=args)
    node = ImgCheckService()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
