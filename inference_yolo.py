import cv2
from ultralytics import YOLO
import os

# 加载训练好的YOLOv8l pose模型
model = YOLO(r"hands_keypoint_model_l/weights/best.pt")

# 定义视频文件路径
video_path = r"data/test/test2.mp4"
cap = cv2.VideoCapture(video_path)

# 获取视频帧率
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 初始化视频写入器，用于保存带有关键点的输出视频
output_dir = "results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_path = "results/annotated_video.mp4"
postfix = 1
while os.path.exists(output_path):
    postfix += 1
    output_path = f"{output_dir}/annotated_video_{postfix}.mp4"
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# 循环处理视频的每一帧
while cap.isOpened():
    success, frame = cap.read()
    if success:
        # 对当前帧进行YOLOv8推理
        results = model(frame)

        # 获取关键点信息
        for result in results:
            keypoints = result.keypoints  # 获取关键点
            print("关键点信息:", keypoints)

            # 可以进一步处理关键点信息，如保存到文件或进行其他分析

        # 在帧上可视化结果
        annotated_frame = results[0].plot()

        # 显示带有关键点的帧
        cv2.imshow("YOLOv8 Pose Inference", annotated_frame)

        # 写入带有关键点的帧到输出视频
        out.write(annotated_frame)

        # 如果按下'q'键，则退出循环
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# 释放视频捕获对象和关闭窗口
cap.release()
out.release()
cv2.destroyAllWindows()
