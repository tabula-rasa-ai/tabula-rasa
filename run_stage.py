"""
run_stage.py — Run any developmental stage of Tabula Rasa.

Usage:
    python3 run_stage.py --stage 1 --steps 5000    # Train Math
    python3 run_stage.py --stage 2 --steps 3000    # Train Patterns
    python3 run_stage.py --stage 3 --steps 5000    # Train Taxonomy
    python3 run_stage.py --stage 4 --steps 2000    # Train State Tracking
    python3 run_stage.py --stage 5 --steps 5000    # Train Grammar
    python3 run_stage.py --stage 6                 # Run dialectics (ebook-based)
    
    python3 run_stage.py --list                    # List all stages
    python3 run_stage.py --check                   # Run graduation check on current stage
"""

import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
import argparse
import sys
from pathlib import Path


STAGE_INFO = {
    1: {
        'name': 'Math (Numbers & Arithmetic)',
        'specialist': 'math_general',
        'dir': 'specialists/math/general',
        'description': 'Absolute Truth: digits, operators, equality',
        'tokenizer': '24-token char (0-9, +-*/=()%.)',
    },
    2: {
        'name': 'Patterns (String Manipulation)',
        'specialist': 'pattern_specialist',
        'dir': 'specialists/patterns/general',
        'description': 'Bridge to Language: reversals, palindromes, sequences',
        'tokenizer': '47-token char (a-z, 0-9)',
    },
    3: {
        'name': 'Taxonomy (Is-A Logic)',
        'specialist': 'taxonomy_specialist',
        'dir': 'specialists/taxonomy/general',
        'description': 'Semantic graph: noun hierarchies, set theory, Z3-verified',
        'tokenizer': '100-token char (a-z, A-Z, punct)',
    },
    4: {
        'name': 'State & Causality',
        'specialist': 'state_specialist',
        'dir': 'specialists/state/general',
        'description': 'English→Code state tracking via Python sandbox',
        'tokenizer': '512-token BPE',
    },
    5: {
        'name': 'Grammar (Syntax Rules)',
        'specialist': 'grammar_specialist',
        'dir': 'specialists/grammar/general',
        'description': 'AST-like parse tree game, CFG evaluation',
        'tokenizer': '512-token BPE',
    },
    6: {
        'name': 'Dialectics (Socratic Philosophy)',
        'specialist': 'dialectics_specialist',
        'dir': 'specialists/dialectics/general',
        'description': 'Wisdom: Socratic debate on ebooks, ethical reasoning',
        'tokenizer': '1024-token BPE',
    },
}


def main():
    parser = argparse.ArgumentParser(description='Tabula Rasa Stage Runner')
    parser.add_argument('--stage', type=int, choices=range(1, 7),
                        help='Developmental stage to train (1-6)')
    parser.add_argument('--steps', type=int, default=5000,
                        help='Number of training steps')
    parser.add_argument('--batch', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate')
    parser.add_argument('--list', action='store_true',
                        help='List all stages and exit')
    parser.add_argument('--check', action='store_true',
                        help='Run graduation check on current stage')
    parser.add_argument('--spawn', action='store_true',
                        help='Create specialist directory and tokenizer without training')
    
    args = parser.parse_args()
    
    if args.list:
        print('Tabula Rasa — 6-Stage Cognitive Syllabus\n')
        for num, info in STAGE_INFO.items():
            print(f'  Stage {num}: {info["name"]}')
            print(f'    Specialist: {info["specialist"]}')
            print(f'    Dir:        {info["dir"]}')
            print(f'    Goal:       {info["description"]}')
            print(f'    Tokenizer:  {info["tokenizer"]}')
            print()
        return 0
    
    if args.check:
        from egefalos.graduation_daemon import daemon_cycle
        daemon_cycle()
        return 0
    
    if args.stage is None:
        parser.print_help()
        return 1
    
    stage = args.stage
    info = STAGE_INFO[stage]
    steps = args.steps
    batch = args.batch
    lr = args.lr
    
    print(f'\n  ╔══════════════════════════════════════════╗')
    print(f'  ║  STAGE {stage}: {info["name"]:<23s}║')
    print(f'  ╚══════════════════════════════════════════╝')
    print(f'  Steps: {steps} | Batch: {batch} | LR: {lr}')
    print(f'  Specialist: {info["specialist"]}')
    print(f'  Directory: {info["dir"]}')
    print()
    
    # Create directory
    Path(info['dir']).mkdir(parents=True, exist_ok=True)
    
    from tabula_rasa.config import Config
    from tabula_rasa.tokenizer import MathTokenizer
    from tabula_rasa.model import MathTransformer
    
    cfg = Config()
    cfg.d_model = 128
    cfg.n_layers = 4
    cfg.max_seq_len = min(128 + stage * 32, 256)
    cfg.batch_size = batch
    cfg.learning_rate = lr
    
    # Create tokenizer
    tok = MathTokenizer()
    
    # Add stage-appropriate characters
    if stage == 1:
        chars = '0123456789+-*/=()%.'
    elif stage == 2:
        chars = 'abcdefghijklmnopqrstuvwxyz0123456789-:><'
    elif stage == 3:
        chars = 'abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ.,!?-:\'[]()_'
    elif stage == 4:
        chars = 'abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?-:\'[]()_=+-*/; '
    elif stage == 5:
        chars = 'abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ.,!?-:\'[](); '
    else:
        chars = 'abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ.,!?-:\'[]();" '
    
    for c in chars:
        if c not in tok.char_to_id:
            tok.add_token(c)
    
    cfg.vocab_size = tok.vocab_size
    tok.max_seq_len = cfg.max_seq_len
    
    if args.spawn:
        # Just create the tokenizer and model structure
        model = MathTransformer(cfg)
        tok.save(str(Path(info['dir']) / 'tokenizer.json'))
        print(f'  Created specialist: {info["dir"]}/')
        print(f'  Tokenizer: {tok.vocab_size} tokens')
        print(f'  Model: {sum(p.numel() for p in model.parameters()):,} params')
        return 0
    
    # Run the appropriate training
    if stage == 1:
        from train import train
        model, tok = train()
    
    elif stage == 2:
        from egefalos.pattern_specialist import train_pattern_specialist, evaluate_patterns
        model = MathTransformer(cfg)
        tok.save(str(Path(info['dir']) / 'tokenizer.json'))
        results = train_pattern_specialist(model, tok, train_steps=steps,
                                            batch_size=batch, learning_rate=lr)
        acc = evaluate_patterns(model, tok, num_problems=200)
        print(f'\n  Final accuracy: {acc*100:.1f}%')
    
    elif stage == 3:
        from egefalos.taxonomy_specialist import train_taxonomy_specialist, evaluate_taxonomy
        model = MathTransformer(cfg)
        tok.save(str(Path(info['dir']) / 'tokenizer.json'))
        results = train_taxonomy_specialist(model, tok, train_steps=steps,
                                             batch_size=batch, learning_rate=lr)
        acc = evaluate_taxonomy(model, tok, num_problems=200)
        print(f'\n  Final accuracy: {acc*100:.1f}%')
    
    elif stage == 4:
        from egefalos.state_specialist import train_state_specialist, evaluate_state
        model = MathTransformer(cfg)
        tok.save(str(Path(info['dir']) / 'tokenizer.json'))
        results = train_state_specialist(model, tok, train_steps=steps,
                                          batch_size=batch, learning_rate=lr)
        acc = evaluate_state(model, tok, num_problems=50)
        print(f'\n  Final sandbox accuracy: {acc*100:.1f}%')
    
    elif stage == 5:
        from egefalos.grammar_specialist import train_grammar_specialist, evaluate_grammar
        model = MathTransformer(cfg)
        tok.save(str(Path(info['dir']) / 'tokenizer.json'))
        results = train_grammar_specialist(model, tok, train_steps=steps,
                                            batch_size=batch, learning_rate=lr)
        acc = evaluate_grammar(model, tok, num_problems=200)
        print(f'\n  Final accuracy: {acc*100:.1f}%')
    
    elif stage == 6:
        print('Stage 6 uses the Socratic engine. Run: python3 -m egefalos.socratic_stage3')
        return 0
    
    # Save checkpoint
    import torch
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': cfg,
        'stage': stage,
        'steps': steps,
    }, Path(info['dir']) / 'best.pt')
    print(f'  Model saved to {info["dir"]}/best.pt')
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
