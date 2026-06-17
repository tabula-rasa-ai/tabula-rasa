"""Quick validation — runs fast tests on CPU in <20s."""
import os
import sys

import torch

sys.path.insert(0, 'src')

PASS = 0; FAIL = 0
def check(c, m):
    global PASS, FAIL
    if c: PASS+=1; print(f'  ✓ {m}')
    else: FAIL+=1; print(f'  ✗ {m}')

print('=== Fast Integration Tests ===')
print()

# 1. Config
from tabula_rasa.config import Config

cfg = Config()
check(cfg.use_entropy_curriculum, 'entropy curriculum ON')
check(cfg.use_gradient_checkpointing, 'gradient checkpointing ON')
check(cfg.mcts_simulations == 16, 'MCTS sims = 16')
check(cfg.rasa_max_relative == 16, 'RASA max_rel = 16')
for preset in ['1M', '5M', '10M']:
    c2 = Config(); c2.apply_preset(preset)
    check(c2.d_model > 0, f'{preset} preset loads')

# 2. Tokenizer
from tabula_rasa.tokenizer import MathTokenizer

tok = MathTokenizer()
check(tok.vocab_size > 0, f'vocab size = {tok.vocab_size}')
for text in ['12+34=46', '7+8=15', '100-1=99']:
    ids = tok.encode(text, add_special_tokens=True)
    decoded = tok.decode(ids, skip_special=True)
    check(decoded == text, f'roundtrip: {text}')

# 3. Model: RASA
from tabula_rasa.model import MathTransformer

cfg = Config(); cfg.d_model=32; cfg.n_layers=2; cfg.n_heads=4; cfg.d_ff=64
cfg.vocab_size = tok.vocab_size; tok.max_seq_len = 48
cfg.use_rasa = True
m = MathTransformer(cfg)
check(m.layers[0].attention.use_rasa, 'RASA enabled')
check(hasattr(m.layers[0].attention, 'rel_bias'), 'RASA rel_bias')
x = torch.randint(0, tok.vocab_size, (2, 8))
l, loss, v = m(x, x)
check(torch.isfinite(loss), f'RASA forward: loss={loss.item():.4f}')

# 4. Model: MoE
cfg.use_rasa = False; cfg.use_moe = True; cfg.num_experts = 4; cfg.top_k = 2
m2 = MathTransformer(cfg)
l2, loss2, v2 = m2(x, x)
aux = m2.get_moe_aux_loss()
check(torch.isfinite(loss2), f'MoE forward: loss={loss2.item():.4f}')
check(torch.isfinite(aux), f'MoE aux loss: {aux.item():.4f}')

# 5. LoRA
from tabula_rasa.lora import apply_lora_to_model, load_lora_adapters, save_lora_adapters

cfg.use_moe = False; cfg.use_rasa = False
m3 = MathTransformer(cfg)
lora = apply_lora_to_model(m3, rank=4, alpha=1.0)
save_lora_adapters(lora, '/tmp/_lora_test.pt')
lora2 = load_lora_adapters(MathTransformer(cfg), '/tmp/_lora_test.pt', rank=4, alpha=1.0)
check(len(lora) == 8, f'LoRA applied: {len(lora)} layers')
check(len(lora2) == 8, 'LoRA loaded')
os.unlink('/tmp/_lora_test.pt')

# 6. Benchmark imports
from tabula_rasa.eval import BOUNDARY_PROBLEMS

check(len(BOUNDARY_PROBLEMS) == 31, f'{len(BOUNDARY_PROBLEMS)} boundary cases')

# 7. Orchestrator
from tabula_rasa.orchestrator import PreparedSlateOrchestrator

orch = PreparedSlateOrchestrator()
check(orch._is_math_query('12+34'), 'math detection')
check(orch._is_greeting('Hello'), 'greeting detection')
check(orch._is_definition('What is AI?'), 'definition detection')
r1 = orch.answer('Hello!'); check(r1.strategy == 'fallback_rule', f'Hello -> {r1.strategy}')
r2 = orch.answer('12+34'); check('not sure' in r2.answer.lower(), 'fallback fallthrough')

# 8. MCTS / RAG structure
from tabula_rasa.mcts_inference import MCTSInference
from tabula_rasa.rag import RAGEngine

m4 = MathTransformer(cfg); m4.eval()
mcts = MCTSInference(m4, tok, num_simulations=4, top_k=5, max_new_tokens=6)
ans, stats = mcts.generate('2+3')
check('num_simulations' in stats, f'MCTS: {stats["total_nodes"]} nodes')

rag = RAGEngine(); n = rag.build_index()
check(n > 0, f'RAG: {n} docs')
res = rag.retrieve('What is a transformer?', top_k=1)
check(len(res) > 0, f'RAG retrieved: {res[0].score:.3f}')

# Summary
total = PASS + FAIL
print(f'\n{"="*50}')
print(f'  {PASS}/{total} passed ({FAIL} failed)')
if FAIL == 0:
    print('  All integration tests OK')
print(f'{"="*50}')
sys.exit(0 if FAIL == 0 else 1)
