import torch, gc, time, os, sys, warnings, json
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

try:
    import transformers.utils.versions as v
    _orig = v.require_version
    def _patched(requirement, hint=None):
        try: return _orig(requirement, hint)
        except ImportError: pass
    v.require_version = _patched
except Exception: pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer
from src.core.engine_wrapper import build_fused_cache, ChunkedPrefillEngine

device='cuda:0'
model = Qwen2VLForConditionalGeneration.from_pretrained('models/Qwen2-VL-7B', torch_dtype=torch.bfloat16, device_map='auto', local_files_only=True, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('models/Qwen2-VL-7B', local_files_only=True, trust_remote_code=True)

block = '<|vision_start|>Frame: A gray normal frame with no anomaly. <|vision_end|> '

def run_baseline(tokenizer, model, prompt):
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    torch.cuda.reset_peak_memory_stats(device)
    t0=time.time()
    with torch.no_grad():
        out = model(input_ids=inputs.input_ids, use_cache=True)
    torch.cuda.synchronize(device)
    ttft = time.time()-t0
    past = out.past_key_values
    cur = inputs.input_ids[:, -1:]
    dec = []
    for _ in range(10):
        t1=time.time()
        with torch.no_grad():
            out = model(input_ids=cur, past_key_values=past, use_cache=True)
        cur = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        torch.cuda.synchronize(device)
        dec.append(time.time()-t1)
        past = out.past_key_values
    peak = torch.cuda.max_memory_allocated(device)/1024**3
    tpot = sum(dec)/len(dec)
    full = torch.cat([inputs.input_ids, cur], dim=-1)
    resp = tokenizer.decode(full[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return {'ttft':ttft,'tpot':tpot*1000,'peak':peak,'resp':resp,'len':inputs.input_ids.shape[1]}

def run_hetero(tokenizer, model, prompt):
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    class A:
        def __init__(self,m): self.model=m; self.config=m.config
        def __call__(self, input_ids, past_key_values, use_cache=True, **kwargs):
            with torch.no_grad():
                return self.model(input_ids=input_ids, past_key_values=past_key_values, use_cache=use_cache, **kwargs)
    cache = build_fused_cache(num_layers=28, device=device, sink_tokens=64, keep_tail=4096, chunk_size=2048, group_size=128, enable_quant=True, enable_prefetch=False, enable_triton=False)
    engine = ChunkedPrefillEngine(model=A(model), cache=cache, chunk_size=2048)
    torch.cuda.reset_peak_memory_stats(device)
    t0=time.time()
    engine.prefill(inputs.input_ids)
    torch.cuda.synchronize(device)
    ttft = time.time()-t0
    cur = inputs.input_ids[:, -1:]
    dec = []
    for _ in range(10):
        t1=time.time()
        with torch.no_grad():
            out = A(model)(input_ids=cur, past_key_values=cache, use_cache=True)
        cur = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        torch.cuda.synchronize(device)
        dec.append(time.time()-t1)
    peak = torch.cuda.max_memory_allocated(device)/1024**3
    tpot = sum(dec)/len(dec)
    full = torch.cat([inputs.input_ids, cur], dim=-1)
    resp = tokenizer.decode(full[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return {'ttft':ttft,'tpot':tpot*1000,'peak':peak,'resp':resp,'len':inputs.input_ids.shape[1]}

results = []
for target, needle in [(8192, 300), (16384, 600), (32768, 1200), (65536, 2400)]:
    n_blocks = target // 22
    blocks = [block] * n_blocks
    blocks[needle] = '<|vision_start|>Frame: RED_ANOMALY_CODE_9527. <|vision_end|> '
    raw = ''.join(blocks)
    prompt = tokenizer.apply_chat_template([{'role':'user','content':'You are watching a long video frame by frame.\n\n'+raw+'\n\nQuestion: What is the exact secret code? Output only the code.'}], tokenize=False, add_generation_prompt=True)
    tok_len = tokenizer(prompt, return_tensors='pt').input_ids.shape[1]
    print(f'\nConfig {target} -> actual {tok_len} tokens')
    print('  Baseline...')
    try:
        b = run_baseline(tokenizer, model, prompt)
        print(f'    TTFT={b["ttft"]:.2f}s TPOT={b["tpot"]:.2f}ms Peak={b["peak"]:.2f}GB')
    except RuntimeError as e:
        b = {'error':'OOM'}
        print('    OOM')
    torch.cuda.empty_cache(); gc.collect()
    print('  Hetero...')
    h = run_hetero(tokenizer, model, prompt)
    print(f'    TTFT={h["ttft"]:.2f}s TPOT={h["tpot"]:.2f}ms Peak={h["peak"]:.2f}GB')
    results.append({'target':target,'actual_len':tok_len,'baseline':b,'hetero':h})
    torch.cuda.empty_cache(); gc.collect()

os.makedirs('experiments', exist_ok=True)
with open('experiments/eval_long_video_quick.json','w') as f:
    json.dump(results, f, indent=2)
print('\nSaved to experiments/eval_long_video_quick.json')
