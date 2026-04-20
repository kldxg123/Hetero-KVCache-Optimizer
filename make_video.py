import cv2
import numpy as np
import time


def create_dummy_long_video():
    filename = 'long_test_video.mp4'
    fps = 1
    duration_seconds = 600  # 600 秒 = 10 分钟
    width, height = 336, 336  # 这是一个对 Qwen2-VL 的 ViT 非常友好的标准分辨率

    print(f"🎬 开始生成物理压测视频: {filename}")
    print(f"⏳ 目标长度: {duration_seconds} 秒 ({duration_seconds} 帧) ...")

    t0 = time.time()
    # 定义 MP4 编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    for i in range(duration_seconds):
        # 生成一个背景颜色随时间平滑变化的画面
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (int((i / duration_seconds) * 255), 80, 150)

        # 在画面中心打上帧率和时间戳（模拟画面一直在动）
        text = f"Frame: {i + 1} / 600"
        cv2.putText(frame, text, (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        out.write(frame)

    out.release()
    print(f"✅ 视频生成完毕！耗时: {time.time() - t0:.2f} 秒。")
    print(f"🚀 现在你可以去跑 Qwen2-VL 的长视频极限压测了！")


if __name__ == "__main__":
    create_dummy_long_video()