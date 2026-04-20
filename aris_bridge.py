import subprocess
import os


def run_experiment(input_len, sink_size=64):
    """
    运行压测并捕获 OOM 信号，供 ARIS 循环调用
    """
    env = os.environ.copy()
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    cmd = [
        "python", "tests/profiling_script.py",
        "--model_path", "./models/Qwen2-VL-7B",
        "--input_len", str(input_len),
        "--sink_size", str(sink_size),
        "--tail_size", "8192"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)

        # 信号捕获逻辑
        if "OutOfMemoryError" in result.stderr or "Native OOM" in result.stdout:
            print(f"SIGNAL: OOM_DETECTED at {input_len}")
            return False, result.stderr

        print(f"SIGNAL: SUCCESS for {input_len}")
        print(result.stdout)
        return True, result.stdout

    except Exception as e:
        print(f"SIGNAL: CRASH - {str(e)}")
        return False, str(e)


if __name__ == "__main__":
    # 示例：ARIS 可以调用此脚本进行 32k-128k 的自动步进测试
    run_experiment(45000)