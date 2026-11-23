"""Small synthetic two-step invocation through the real observer adapters.

No model, research input, weight edit or numerical training is performed. This
fixture exercises repeated operation identity and is also exported to MLflow.
"""
from types import SimpleNamespace


def record_two_steps(recorder, wrap):
    supplied = object()
    callbacks = []
    push = wrap(lambda value: value, recorder, 'solver.jvp', detached=True)
    pull = wrap(lambda value: value, recorder, 'solver.vjp', detached=True)

    def solve(*, progress):
        progress(0, 1.)
        assert pull(supplied) is supplied
        assert push(supplied) is supplied
        assert push(supplied) is supplied
        progress(1, .5)
        assert pull(supplied) is supplied
        return SimpleNamespace(status='stationary', iterations=1)

    inverse = wrap(solve, recorder, 'solver.solve', detached=True)
    value = SimpleNamespace(estimate=.25, lower=.25, upper=.25)

    def search(*, evaluate):
        assert evaluate(supplied) is value  # unchanged baseline
        assert evaluate(supplied) is value  # one candidate
        return SimpleNamespace(status='accepted', value=value, length=.5, trials=[{}])

    proposal = wrap(search, recorder, 'proposal.search', detached=True)

    def correct(owner, number, observation):
        inverse(progress=lambda iteration, objective: callbacks.append((number, iteration, objective)))
        proposal(evaluate=lambda weights: value)
        return {'status': 'synthetic_only'}

    correction = wrap(correct, recorder, 'correction.step', detached=True)

    def loop():
        for number in (1, 2):
            correction(None, number, {'model_sha256': f'synthetic-current-model-{number}',
                                     'reference_sha256': 'synthetic-reference'})
        return dict(status='synthetic_only', closure=False, updates=0, repair_claim=False)

    with recorder.span('research.invocation', detached=True):
        wrap(loop, recorder, 'training.loop', detached=True)()
    return callbacks
