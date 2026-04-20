"""
tests/real_qwen_vl_benchmark.py
===============================
真实 Qwen2-VL-7B MLLM 端到端基准测试 - 简化版本
绕过 AutoProcessor 使用直接文本输入
"""

import os
import sys
import json
import time
import gc
import torch
import warnings
warnings.filterwarnings('ignore')

os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

try:
    import transformers.utils.versions as v
    _orig = v.require_version
    def _patched(requirement, hint=None):
        try:
            return _orig(requirement, hint)
        except ImportError:
            pass
    v.require_version = _patched
except Exception:
    pass

# 导入 - 使用 Qwen2VL 专用类
try:
    from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, BitsAndBytesConfig
    MODEL_CLASS = Qwen2VLForConditionalGeneration
    print("[INFO] 使用 Qwen2VLForConditionalGeneration")
except ImportError as e:
    print(f"[错误] 无法导入 Qwen2VLForConditionalGeneration: {e}")
    print("[错误] 请确保 transformers >= 4.40.0")
    raise

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.engine_wrapper import build_fused_cache, ChunkedPrefillEngine


class QwenVLTextOnlyWrapper:
    """
    Qwen2-VL 测试包装器（文本模式，但使用4-bit加载模拟MLLM压力）
    由于transformers版本限制，我们用高文本token数来模拟多模态场景
    """
    
    def __init__(self, model_path="models/Qwen2-VL-7B", device="cuda"):
        self.device = device
        print(f"[ARIS-MULTIMODAL] 加载 Qwen2-VL-7B (4-bit模式)")
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        
        # 加载 Tokenizer
        print("[ARIS-MULTIMODAL] 加载 Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        
        # 4-bit 量化（MLLM实际部署模式）
        print("[ARIS-MULTIMODAL] 配置 4-bit 量化...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        print("[ARIS-MULTIMODAL] 开始加载 Qwen2-VL-7B (4-bit)...")
        load_start = time.time()
        
        # 使用 Qwen2VLForConditionalGeneration 加载
        self.model = MODEL_CLASS.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
        )
        load_time = time.time() - load_start
        
        model_load_memory = torch.cuda.memory_allocated() / (1024**3)
        print(f"[ARIS-MULTIMODAL] 成功加载 4-bit 模型 (耗时 {load_time:.1f}s)")
        print(f"[ARIS-MULTIMODAL] 模型权重占用显存: {model_load_memory:.2f} GB")
        
        self.model.eval()
        
        config = self.model.config
        # Qwen2-VL 配置结构不同，需要访问 llm_config
        llm_config = getattr(config, 'llm_config', config)
        self.num_layers = getattr(llm_config, 'num_hidden_layers', 28)
        self.hidden_size = getattr(llm_config, 'hidden_size', 3584)
        self.num_heads = getattr(llm_config, 'num_attention_heads', 28)
        self.num_kv_heads = getattr(llm_config, 'num_key_value_heads', 4)
        self.head_dim = self.hidden_size // self.num_heads
        
        print(f"[ARIS-MULTIMODAL] 模型配置:")
        print(f"       - Layers: {self.num_layers}")
        print(f"       - Hidden size: {self.hidden_size}")
        print(f"       - KV heads (GQA): {self.num_kv_heads}")
        print(f"       - Head dim: {self.head_dim}")
        
        # CUDA Warmup
        print("\n[ARIS-MULTIMODAL] ====== CUDA Warm-up ======")
        self._cuda_warmup()
        print("[ARIS-MULTIMODAL] ====== Warm-up 完成 ======\n")

    def _cuda_warmup(self):
        """CUDA预热"""
        text = "Hello world"
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        try:
            with torch.no_grad():
                _ = self.model(**inputs)
            torch.cuda.synchronize()
        except Exception as e:
            print(f"[Warmup警告] {e}")
        
        del inputs
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()

    def create_long_context_input(self, target_tokens=4000, simulate_visual_tokens=True):
        """
        创建长上下文输入
        模拟多模态场景：视觉token（约256个/token per image）+ 文本
        """
        print(f"[ARIS-MULTIMODAL] 创建长上下文输入: 目标 {target_tokens} tokens")
        
        # 构造长文本 + 模拟视觉token标记
        if simulate_visual_tokens:
            # 模拟视频帧描述
            visual_context = "<|vision_start|>Frame 1: Red square scene. <|vision_end|> " + \
                           "<|vision_start|>Frame 2: Blue circle scene. <|vision_end|> " + \
                           "<|vision_start|>Frame 3: Green triangle scene. <|vision_end|> "
            base_text = visual_context + "Describe what you see in the video frames in detail. "
        else:
            base_text = "The video shows a sequence of events. "
        
        # 重复文本以达到目标token数
        repeat_count = (target_tokens // 20) + 1
        long_text = (base_text * repeat_count)[:target_tokens * 5]
        
        # 构造chat格式
        messages = [{"role": "user", "content": long_text}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=target_tokens+100).to(self.device)
        
        actual_tokens = inputs.input_ids.shape[1]
        print(f"[ARIS-MULTIMODAL] 实际输入 token 数: {actual_tokens}")
        
        return inputs

    def run_with_hetero_cache(self, inputs, min_new_tokens=20, chunk_size=2048):
        """Hetero-KVCache 运行"""
        input_ids = inputs.input_ids
        seq_len = input_ids.shape[1]
        print(f"\n[ARIS-MULTIMODAL] Hetero-KV 模式 | 输入: {seq_len} tokens")
        
        torch.cuda.empty_cache()
        gc.collect()
        
        cache = build_fused_cache(
            device=self.device,
            sink_tokens=64,
            keep_tail=4096,
            chunk_size=chunk_size,
            group_size=128,
            enable_quant=True,
            enable_prefetch=False,
            enable_triton=False
        )
        
        class ModelAdapter:
            def __init__(self, real_model):
                self.model = real_model
                self.config = real_model.config
            
            def __call__(self, input_ids, past_key_values, use_cache=True, **kwargs):
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=input_ids,
                        past_key_values=past_key_values,
                        use_cache=use_cache,
                        **kwargs
                    )
                return outputs
        
        adapter = ModelAdapter(self.model)
        engine = ChunkedPrefillEngine(model=adapter, cache=cache, chunk_size=chunk_size)
        
        try:
            # Prefill
            print("[ARIS-MULTIMODAL] 开始 Prefill...")
            prefill_start = time.time()
            engine.prefill(input_ids)
            torch.cuda.synchronize()
            prefill_end = time.time()
            
            ttft = prefill_end - prefill_start
            peak_mem_prefill = torch.cuda.max_memory_allocated() / (1024**3)
            
            print(f"[ARIS-MULTIMODAL] Prefill 完成 | TTFT: {ttft:.3f}s | Peak: {peak_mem_prefill:.3f}GB")
            
            # Decode
            print(f"[ARIS-MULTIMODAL] 开始 Decode ({min_new_tokens} tokens)...")
            current_input = input_ids[:, -1:]
            decode_times = []
            
            for i in range(min_new_tokens):
                token_start = time.time()
                
                with torch.no_grad():
                    outputs = adapter(
                        input_ids=current_input,
                        past_key_values=cache,
                        use_cache=True
                    )
                
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
                torch.cuda.synchronize()
                decode_times.append(time.time() - token_start)
                current_input = next_token
            
            tpot = sum(decode_times) / len(decode_times)
            peak_mem_total = torch.cuda.max_memory_allocated() / (1024**3)
            current_mem = torch.cuda.memory_allocated() / (1024**3)
            
            print(f"[ARIS-MULTIMODAL] Decode 完成 | TPOT: {tpot*1000:.2f}ms | Peak: {peak_mem_total:.3f}GB | Steady: {current_mem:.3f}GB")
            
            return {
                "success": True,
                "ttft": ttft,
                "tpot": tpot,
                "peak_memory_gb": peak_mem_total,
                "steady_memory_gb": current_mem,
                "seq_length": cache.get_seq_length(),
            }
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[ARIS-MULTIMODAL] OOM: {str(e)[:100]}")
                return {"success": False, "error": "OOM"}
            raise
        finally:
            del cache, engine, adapter

    def run_native(self, inputs, min_new_tokens=20):
        """Native HF 运行"""
        input_ids = inputs.input_ids
        seq_len = input_ids.shape[1]
        print(f"\n[ARIS-MULTIMODAL] Native HF 模式 | 输入: {seq_len} tokens")
        
        torch.cuda.empty_cache()
        gc.collect()
        
        try:
            print("[ARIS-MULTIMODAL] 开始 Native Prefill...")
            prefill_start = time.time()
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    use_cache=True,
                )
            torch.cuda.synchronize()
            prefill_end = time.time()
            
            ttft = prefill_end - prefill_start
            peak_mem_prefill = torch.cuda.max_memory_allocated() / (1024**3)
            
            print(f"[ARIS-MULTIMODAL] Native Prefill 完成 | TTFT: {ttft:.3f}s | Peak: {peak_mem_prefill:.3f}GB")
            
            # Decode
            print("[ARIS-MULTIMODAL] 开始 Native Decode...")
            past_key_values = outputs.past_key_values
            current_input = input_ids[:, -1:]
            decode_times = []
            
            for i in range(min_new_tokens):
                token_start = time.time()
                
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=current_input,
                        past_key_values=past_key_values,
                        use_cache=True
                    )
                
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
                torch.cuda.synchronize()
                decode_times.append(time.time() - token_start)
                past_key_values = outputs.past_key_values
                current_input = next_token
            
            tpot = sum(decode_times) / len(decode_times)
            peak_mem_total = torch.cuda.max_memory_allocated() / (1024**3)
            current_mem = torch.cuda.memory_allocated() / (1024**3)
            
            print(f"[ARIS-MULTIMODAL] Native Decode 完成 | TPOT: {tpot*1000:.2f}ms | Peak: {peak_mem_total:.3f}GB")
            
            return {
                "success": True,
                "ttft": ttft,
                "tpot": tpot,
                "peak_memory_gb": peak_mem_total,
                "steady_memory_gb": current_mem,
            }
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[ARIS-MULTIMODAL] Native OOM: {str(e)[:100]}")
                return {"success": False, "error": "OOM"}
            raise


def run_mllm_benchmark_suite():
    """运行 MLLM 基准测试"""
    print("="*70)
    print(" ARIS - Qwen2-VL-7B MLLM 基准测试")
    print(" 4-bit量化模式（模拟真实MLLM部署压力）")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[错误] 需要 GPU")
        return {}
    
    gpu_name = torch.cuda.get_device_name(0)
    print(f"[ARIS-MULTIMODAL] GPU: {gpu_name}")
    
    # 测试配置
    test_configs = [
        {"name": "4K context", "target_tokens": 4000},
        {"name": "8K context", "target_tokens": 8000},
        {"name": "12K context", "target_tokens": 12000},
    ]
    
    results = {
        "device": gpu_name,
        "model": "Qwen2-VL-7B-Instruct (4-bit NF4)",
        "test_type": "Long-context MLLM simulation",
        "tests": []
    }
    
    try:
        wrapper = QwenVLTextOnlyWrapper(device=device)
        
        for config in test_configs:
            print(f"\n{'='*70}")
            print(f" 测试: {config['name']}")
            print(f"{'='*70}")
            
            test_result = {
                "name": config['name'],
                "hetero": None,
                "native": None
            }
            
            inputs = wrapper.create_long_context_input(target_tokens=config['target_tokens'])
            test_result['actual_tokens'] = inputs.input_ids.shape[1]
            
            # Hetero 测试
            print("\n" + "-"*50)
            print(" 运行 Hetero-KVCache 测试")
            print("-"*50)
            hetero_result = wrapper.run_with_hetero_cache(inputs, min_new_tokens=20)
            test_result['hetero'] = hetero_result
            
            torch.cuda.empty_cache()
            gc.collect()
            
            # Native 测试
            if config['target_tokens'] <= 8000:
                print("\n" + "-"*50)
                print(" 运行 Native HF 测试")
                print("-"*50)
                native_result = wrapper.run_native(inputs, min_new_tokens=20)
                test_result['native'] = native_result
            else:
                test_result['native'] = {"skipped": True, "reason": "预期OOM"}
            
            results['tests'].append(test_result)
            
            os.makedirs("experiments", exist_ok=True)
            with open("experiments/real_qwen_vl_benchmark.json", "w") as f:
                json.dump(results, f, indent=2)
            
            del inputs
            torch.cuda.empty_cache()
            gc.collect()
            
    except Exception as e:
        print(f"[ARIS-MULTIMODAL] 错误: {e}")
        import traceback
        traceback.print_exc()
        results['error'] = str(e)
    
    with open("experiments/real_qwen_vl_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*70)
    print(" MLLM 基准测试完成")
    print("="*70)
    
    return results


if __name__ == "__main__":
    run_mllm_benchmark_suite()
