"""Bind a prospective post-closure reading to its recorded model and population.

This is identity/admission arithmetic only. No acquisition or objective is executed,
and record consistency does not establish that a computational observable is valid.
The Taylor-remainder reader has been retired.
"""
from .functional_reference import checked


def admitted(root, declaration):
    closure = checked(root, declaration['closure_source'])
    measured = checked(root, closure['measurement'])
    if (closure.get('closure') is not True or closure.get('updates', 0) < 1
            or measured.get('complete') is not True or measured.get('ended') is not True
            or measured.get('outside') != 0):
        raise ValueError('other-answer evaluation requires complete fresh-trace training closure')
    spec = declaration['functional_loop']
    if (closure['model_file'] != spec['first_state']['file']
            or closure['model_sha256'] != spec['first_state']['model_sha256']
            or closure['reference_sha256'] != declaration['reference']['sha256']
            or measured.get('model_sha256') != closure['model_sha256']
            or measured.get('reference_sha256') != closure['reference_sha256']):
        raise ValueError('the evaluation must use the closed model and unchanged original reference')
    proposed = closure['protocol']
    if (spec['cases'] != proposed['cases'] or spec['reader_protocol'] != proposed['reader_protocol']
            or spec['training_members'] != proposed['members']):
        raise ValueError('the other-answer population and protocol must be fixed before training')
    if spec.get('acquisition') != 'native-events-only':
        raise ValueError('evaluation cannot use token-based or turn-based registration')
    return closure
