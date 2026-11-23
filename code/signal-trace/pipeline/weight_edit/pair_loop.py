"""Generic observe/correct controller, without a native backend or command.

The backend must supply a separately justified observable and complete scope.
This controller does not establish either; the Taylor-remainder backend is retired.
"""


def iterate(backend):
    updates = 0
    while True:
        observation = backend.observe(updates + 1)
        if observation['complete'] is not True or observation['ended'] is not True:
            result = dict(status='incomplete_acquisition', closure=False, updates=updates)
            break
        if observation['outside'] == 0:
            if updates == 0:
                result = dict(status='initially_inside_tolerance', closure=False,
                              updates=0, repair_claim=False)
            else:
                result = dict(status='closed', closure=True, updates=updates,
                              post_closure=backend.post_closure(), repair_claim=False)
            break
        correction = backend.correct(updates + 1, observation)
        if correction['status'] != 'accepted':
            result = dict(status=correction['status'], closure=False, updates=updates,
                          correction=correction)
            break
        updates += 1
    result.setdefault('repair_claim', False)
    backend.finish(result)
    return result
