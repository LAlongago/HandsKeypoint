from video_capture import VideoCapture
from pose_detection import PoseDetection
from gesture_recognition import GestureRecognition
from ui_display import UIDisplay
import config


def main():
    # 初始化各模块
    video_capture = VideoCapture(config.VIDEO_SOURCE)
    pose_detection = PoseDetection(config.MODEL_PATH)
    gesture_recognition = GestureRecognition()
    ui_display = UIDisplay()

    while True:
        # 获取一帧图像
        frame = video_capture.get_frame()
        if frame is None:
            break

        # 进行关键点检测
        results = pose_detection.detect(frame)

        # 进行手势识别
        gestures = gesture_recognition.recognize(results)

        # 显示结果
        ui_display.show(frame, results, gestures)

        # 退出条件
        if ui_display.check_exit():
            break

    video_capture.release()
    ui_display.close()


if __name__ == "__main__":
    main()
