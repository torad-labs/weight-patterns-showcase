"""Authenticate a measured group checkpoint and its chain back to the stock cohort."""
from pathlib import Path

from pipeline.weight_edit import final_experiment as baseline, group_development_origin


def bound_json(binding, expected):
    if binding != baseline.binding(expected):
        raise ValueError(f'Changed joint stock lineage at {expected.name}')
    return baseline.read(expected)


def from_parent(parent, lineage):
    """Keep only a stock-rooted group lineage. A fresh origin stays a separate admission."""
    seen=set()

    def visit(folder):
        folder=baseline.run_path(folder)
        if folder in seen:
            raise ValueError('Cycle in stock group continuation')
        seen.add(folder)
        declaration_path=folder/'declaration.json'
        prior=baseline.read(declaration_path)
        group=prior.get('group_contrast')
        if not group:
            return None
        direct=prior.get('group_development_origin')
        inherited=prior.get('joint_stock_lineage')
        state_path=folder/'final-state.json'
        state=baseline.read(state_path)
        stock=lineage['original_model']
        base=dict(file=stock['record'], model_sha256=stock['sha256'])
        if not inherited and not (direct and direct.get('pristine_exit')):
            if state['base_identity']==base:
                raise ValueError('Stock group checkpoint lacks its measured pristine-origin lineage')
            return None
        if inherited and direct:
            raise ValueError('Ambiguous parent of stock group continuation')
        seed=prior['resume_checkpoint']
        seed_folder=baseline.run_path((baseline.ROOT/seed['record']).parent)
        if direct:
            checked=group_development_origin.checked_origin(direct['origin'],lineage,predecessor=seed_folder)
            if direct!=checked or checked['checkpoint']!=seed or checked['measurement']!=prior['start_fresh']:
                raise ValueError('Stock group run does not begin at its authenticated pristine origin')
            root=checked['origin']
            history=0
        else:
            previous=visit(seed_folder)
            if inherited!=previous:
                raise ValueError('Changed predecessor in the stock group chain')
            check_successor(previous,prior)
            root=previous['origin']
            history=previous['historical_updates']
        checkpoint=baseline.binding(state_path)
        measurement=state['last_measurement']
        step=state['iteration']
        observed=bound_json(measurement,folder/'loop'/str(step)/'measurement.json')
        manifest=bound_json(state['checkpoint'],folder/'final-checkpoint/state.json')
        branch_path=folder/'group-boundary.json'
        branch=baseline.read(branch_path)
        panel=bound_json(branch['group_measurement'],folder/'loop'/str(step)/'group-measurement.json')
        exit_record=bound_json(branch['launcher_exit'],folder/'launcher-exit.json')
        identity=dict(kind='native-working-revision/1',run=str(folder.relative_to(baseline.ROOT)),
                      generation=step,original_sha256=stock['sha256'])
        primary=[v['measurement'] for v in panel['members'] if v['id']==group['primary']]
        common=('format','status','iteration','model_identity','base_identity','inventory_identity','transition')
        if (prior['operation']!='final-experiment-signed-group-contrast-training'
                or prior['original_model']!=stock or prior['eta']!=lineage['eta']
                or prior['counter_semantics']['historical_predecessor']!=history
                or state['base_identity']!=base or state['model_identity']!=identity
                or state['eta']!=prior['eta'] or state['status']!='complete'
                or any(manifest[k]!=state[k] for k in common) or manifest['complete'] is not True
                or state['loss_identity']!=prior['loss_identity'] or prior['loss_identity']!=lineage['loss_identity']
                or observed['model_identity']!=identity or observed['generation']!=step or observed['step']!=step
                or not observed['complete'] or not observed['ended']
                or panel['generation']!=step or panel['model_identity']!=identity
                or panel['policy']!=group['policy'] or primary!=[measurement]
                or branch['installed_updates']!=step or branch['state']!=checkpoint
                or exit_record['exit_code']!=0 or exit_record['installed_updates']!=step
                or exit_record['last_measurement']!=measurement):
            raise ValueError('Stock ancestry, group objective, endpoint or optimizer step differs')
        bound_json(group['policy'],baseline.ROOT/group['policy']['record'])
        return dict(format='joint-stock-continuation/1', origin=root, policy=group['policy'],
                    parent=baseline.binding(declaration_path), group_boundary=baseline.binding(branch_path),
                    group_observation=branch['group_measurement'], checkpoint=checkpoint,
                    manifest=state['checkpoint'], measurement=measurement, exit=branch['launcher_exit'],
                    historical_updates=history+step, local_updates=step, eta=state['eta'])

    return visit(parent)


def check_successor(line, declaration):
    if (not line or declaration.get('group_contrast',{}).get('policy')!=line['policy']
            or declaration['resume_checkpoint']!=line['checkpoint']
            or declaration['start_fresh']!=line['measurement']
            or declaration['predecessor_exit']['record']!=line['exit']
            or declaration['resume_predecessor_generation']!=line['local_updates']
            or declaration['counter_semantics']['historical_predecessor']!=line['historical_updates']
            or declaration['eta']!=line['eta']):
        raise ValueError('Stock continuation requires the same group objective, native checkpoint, remainder and eta')
