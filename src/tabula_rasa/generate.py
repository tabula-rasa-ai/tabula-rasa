"""Interactive generation with trained model."""

import torch
import sys
from pathlib import Path

from tabula_rasa.config import Config
from tabula_rasa.tokenizer import MathTokenizer
from tabula_rasa.model import MathTransformer, count_parameters


def load_model(checkpoint_path):
    cfg = Config()
    tok = MathTokenizer.load(str(Path(checkpoint_path).parent / 'tokenizer.json'))
    cfg.vocab_size = tok.vocab_size  # type: ignore[misc]
    tok.max_seq_len = cfg.max_seq_len

    model = MathTransformer(cfg)
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    return model, tok


def main():
    # Find checkpoint
    ckpt = Path('checkpoints/best.pt')
    if not ckpt.exists():
        ckpt = Path('checkpoints/final.pt')
    if not ckpt.exists():
        print('No checkpoint found. Train first: python3 train.py')
        sys.exit(1)

    print(f'Loading model from {ckpt}')
    model, tok = load_model(ckpt)
    print(f'Model params: {count_parameters(model):,}')
    print(f'Tokenizer vocab: {tok.vocab_size}')
    print()
    print('Enter math expressions followed by =')
    print('Examples: 12+34=   ,   56*78=   ,   (2+3)*4=')
    print('Type "exit" to quit')
    print()

    while True:
        try:
            expr = input('>>> ')
        except (EOFError, KeyboardInterrupt):
            print()
            break

        expr = expr.strip()
        if not expr:
            continue
        if expr.lower() in ('exit', 'quit', 'q'):
            break

        # Ensure it ends with =
        if not expr.endswith('='):
            expr += '='

        generated = model.generate(tok, expr, max_new_tokens=15, temperature=0.3, top_k=3)
        print(f'  {generated}')
        print()


if __name__ == '__main__':
    main()
