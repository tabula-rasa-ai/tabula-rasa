"""
Egefalos (εγκέφαλος) — The Brain Module.

Compatibility shim for backward-compatible imports. All code has moved to
`tabula_rasa.{server,cl,memory,router,reasoning,training,cognitive}`.

Uses a MetaPathFinder to intercept `egefalos.X` imports and redirect them.
"""

import importlib
import sys
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec

# Module name -> new package path
_MODULE_REDIRECT = {
    'tabula_rasa':              'tabula_rasa.server.ai_server',
    'online_ewc':               'tabula_rasa.cl.online_ewc',
    'expert_ewc':               'tabula_rasa.cl.expert_ewc',
    'mas':                      'tabula_rasa.cl.mas',
    'ogd':                      'tabula_rasa.cl.ogd',
    'lwf_gem':                  'tabula_rasa.cl.lwf_gem',
    'hippocampus':              'tabula_rasa.memory.hippocampus',
    'sleep_cycle':              'tabula_rasa.memory.sleep_cycle',
    'replay_buffer':            'tabula_rasa.memory.replay_buffer',
    'memory':                   'tabula_rasa.memory.memory',
    'router_model':             'tabula_rasa.router.router_model',
    'router':                   'tabula_rasa.router.router',
    'router_dataset':           'tabula_rasa.router.router_dataset',
    'router_hippocampus':       'tabula_rasa.router.router_hippocampus',
    'router_sleep_cycle':       'tabula_rasa.router.router_sleep_cycle',
    'task_queue':               'tabula_rasa.training.task_queue',
    'bm25_retrieval':           'tabula_rasa.training.bm25_retrieval',
    'pii_scrubber':             'tabula_rasa.training.pii_scrubber',
    'hard_negative_mining':     'tabula_rasa.training.hard_negative_mining',
    'mult_scratchpad':          'tabula_rasa.training.mult_scratchpad',
    'ppo_trainer':              'tabula_rasa.training.ppo_trainer',
    'self_improve':             'tabula_rasa.training.self_improve',
    'specialist_consolidation': 'tabula_rasa.training.specialist_consolidation',
    'socratic_trainer':         'tabula_rasa.reasoning.socratic_trainer',
    'socratic_stage1':          'tabula_rasa.reasoning.socratic_stage1',
    'socratic_stage2':          'tabula_rasa.reasoning.socratic_stage2',
    'socratic_stage3':          'tabula_rasa.reasoning.socratic_stage3',
    'socratic_critique':        'tabula_rasa.reasoning.socratic_critique',
    'tool_use':                 'tabula_rasa.reasoning.tool_use',
    'mcts':                     'tabula_rasa.reasoning.mcts',
    'math_gym_env':             'tabula_rasa.reasoning.math_gym_env',
    'micro_orchestrator':       'tabula_rasa.reasoning.micro_orchestrator',
    'code_sandbox':             'tabula_rasa.cognitive.code_sandbox',
    'code_specialist':          'tabula_rasa.cognitive.code_specialist',
    'code_curriculum':          'tabula_rasa.cognitive.code_curriculum',
    'language_az':              'tabula_rasa.cognitive.language_az',
    'grammar_engine':           'tabula_rasa.cognitive.grammar_engine',
    'grammar_specialist':       'tabula_rasa.cognitive.grammar_specialist',
    'pattern_specialist':       'tabula_rasa.cognitive.pattern_specialist',
    'taxonomy_specialist':      'tabula_rasa.cognitive.taxonomy_specialist',
    'state_specialist':         'tabula_rasa.cognitive.state_specialist',
    'semantic_game':            'tabula_rasa.cognitive.semantic_game',
    'pythagoras':               'tabula_rasa.cognitive.pythagoras',
    'neocortex':                'tabula_rasa.cognitive.neocortex',
    'logic_verifier':           'tabula_rasa.cognitive.logic_verifier',
    'graduation_daemon':        'tabula_rasa.cognitive.graduation_daemon',
    'p2p_daemon':               'tabula_rasa.cognitive.p2p_daemon',
    'interpret':                'tabula_rasa.cognitive.interpret',
    'interpretability':         'tabula_rasa.cognitive.interpretability',
    'math_call':                'tabula_rasa.cognitive.math_call',
}

# Reverse map: attribute name → (module name, new_path) for from egefalos import X
_ATTR_MAP = {}
for mod_key, new_path in _MODULE_REDIRECT.items():
    try:
        mod = importlib.import_module(new_path)
        for attr_name in dir(mod):
            if not attr_name.startswith('_'):
                _ATTR_MAP[attr_name] = (mod_key, new_path)
    except ImportError:
        pass


class _EgefalosRedirectFinder(MetaPathFinder):
    """Intercepts `egefalos.X` imports and redirects to `tabula_rasa.CATEGORY.X`."""

    def find_spec(self, fullname, path, target=None):
        if not fullname.startswith('egefalos.'):
            return None

        # egefalos.X → redirect to new path
        submod = fullname[len('egefalos.'):]
        if submod in _MODULE_REDIRECT:
            new_path = _MODULE_REDIRECT[submod]
            try:
                real = importlib.import_module(new_path)
                sys.modules[fullname] = real
                return ModuleSpec(fullname, None, origin=f'redirect:{new_path}')
            except ImportError:
                return None

        return None


# Register only once
_installed = False
if not _installed:
    sys.meta_path.insert(0, _EgefalosRedirectFinder())
    _installed = True


def __getattr__(name):
    """Fallback for `from egefalos import X` (attribute-level access)."""
    if name in _ATTR_MAP:
        mod_key, new_path = _ATTR_MAP[name]
        mod = importlib.import_module(new_path)
        return getattr(mod, name)
    raise AttributeError(f"module 'egefalos' has no attribute '{name}'")


__all__ = list(_MODULE_REDIRECT.keys())
