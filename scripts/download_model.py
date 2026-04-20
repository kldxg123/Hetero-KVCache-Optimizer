import os
from modelscope.hub.snapshot_download import snapshot_download


def download_model():
    model_id = "qwen/Qwen2.5-7B-Instruct"

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_dir = os.path.join(project_root, "models", "Qwen2.5-7B-Instruct")

    print(f"🚀 切换至 ModelScope (魔搭社区) 高速通道下载模型: {model_id}")
    print(f"📂 目标路径: {local_dir}")

    os.makedirs(local_dir, exist_ok=True)

    snapshot_download(
        model_id,
        local_dir=local_dir,
        revision='master'
    )
    print("✅ 模型下载完毕！")


if __name__ == "__main__":
    download_model()
EOF

# 4. 再次执行下载脚本
python
scripts / download_model.py