"""External checkpoint identities, including states whose full copy was released.

MLflow stores small manifests, not weights or a serving-model package. The
controller binds weight hashes; this observer checks small metadata and file
presence/size without rereading or rebuilding multi-gigabyte checkpoints.
"""
import hashlib
import json
from pathlib import Path
import tempfile

def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()


def record_identity(record):
    """An update revision labels a state; it is never a weight-content digest."""
    value = record.get('model_identity')
    if value is not None:
        if (record.get('model_sha256') is not None or not isinstance(value, dict)
                or set(value) != {'kind', 'run', 'generation', 'original_sha256'}
                or value.get('kind') != 'native-working-revision/1'
                or not isinstance(value.get('run'), str) or not value['run']
                or Path(value['run']).is_absolute() or '..' in Path(value['run']).parts
                or type(value.get('generation')) is not int or value['generation'] < 0
                or not isinstance(value.get('original_sha256'), str)
                or len(value['original_sha256']) != 64
                or any(c not in '0123456789abcdef' for c in value['original_sha256'])):
            raise ValueError('a working revision is an explicit run/update label, not a model SHA256')
        return dict(value)
    value = record.get('model_sha256')
    if not isinstance(value, str) or not value:
        raise ValueError('a model needs its recorded working revision or legacy file SHA256')
    return {'kind': 'file-sha256', 'sha256': value}


def record_generation(record):
    identity = record_identity(record)
    generation = identity['generation']
    if 'generation' in record and record['generation'] != generation:
        raise ValueError('the model generation differs from its recorded revision')
    return generation


def identity_tags(record):
    identity = record_identity(record)
    tags = {'model.identity_kind': identity['kind'],
            'model.sha256': identity['sha256'] if identity['kind'] == 'file-sha256' else ''}
    if identity['kind'] != 'file-sha256':
        tags.update({'model.revision_run': identity['run'], 'model.generation': record_generation(record),
                     'model.original_sha256': identity['original_sha256']})
    return tags


def ram_revision(root, declaration, record, step, *, initial=False):
    """A RAM revision has provenance, but no downloadable per-update weights."""
    revision = record_identity(record)
    generation = record_generation(record)
    if (revision['original_sha256'] != declaration['original_model']['sha256']
            or generation != step or not (root / revision['run']).is_relative_to(
                root / 'studies/04-round-2-overlap/runs')):
        raise ValueError('RAM revision must identify its run, update and original model')
    artifacts = {'initial-measurement.json' if initial else 'reached.json': encoded(record)}
    if initial:
        if step != 0 or not record.get('complete') or not record.get('ended'):
            raise ValueError('RAM baseline must identify a complete initial acquisition')
        resumed = declaration.get('resume_checkpoint')
        if resumed is not None:
            capture = record.get('capture', {})
            expected = f"{revision['run']}/loop/0/capture/capture.json"
            if (record.get('resumed_checkpoint') != resumed or record.get('recovered_from') is not None
                    or capture.get('record') != expected or not capture.get('sha256')):
                raise ValueError('fresh RAM resume must bind its checkpoint and new baseline capture')
            provenance = {'resume_checkpoint': resumed, 'fresh_acquisition': capture,
                          'predecessor_counter': record.get('predecessor_counter')}
        else:
            if record.get('recovered_from') != declaration.get('start_fresh'):
                raise ValueError('RAM baseline must identify its complete declared retained acquisition')
            provenance = {key: declaration[key] for key in ('start_experiment', 'start_fresh')}
    else:
        if (record.get('storage') != 'same anonymous RAM GGUF and resident leaves'
                or record.get('matrices') != declaration['functional_loop']['matrices']):
            raise ValueError('RAM installation must name the complete declared matrix inventory')
        provenance = {'installed_at': record.get('installed_at')}
    original = root / declaration['original_model']['record']
    identity = {
        'format': 'external-checkpoint-reference/2', 'sha256': None, 'model_identity': revision,
        'generation': generation, 'weights_uri': None,
        'storage': 'anonymous RAM model; no per-update weight file or checkpoint retained',
        'identity_source': 'controller run/update label, not a content digest',
        'verification': 'Observer reads revision/provenance metadata only; it does not inspect RAM weights.',
        'role': 'retained initial model' if initial else 'installed update', 'training_step': step,
        'original': {'uri': original.as_uri(), 'sha256': revision['original_sha256'], **present(original)},
        'full_copy': {'present': False, 'bytes': None}, 'availability': 'metadata_only_not_retained',
        'checkpoint_available': False, 'provenance': provenance,
    }
    artifacts['checkpoint.json'] = encoded(identity)
    return identity, artifacts


def working_checkpoint(root, declaration, reached, step):
    """Show current-file availability from the controller's existing progress marker."""
    revision = record_identity(reached)
    generation = record_generation(reached)
    if 'file' not in reached:
        return ram_revision(root, declaration, reached, step)
    original = root / declaration['original_model']['record']
    file = root / revision['run'] / 'working/model.gguf'
    if (root / reached['file'] != file or revision['original_sha256'] != declaration['original_model']['sha256']
            or reached.get('step') != step):
        raise ValueError('the reached revision must belong to its run and original model')
    identity = {
        'format': 'external-checkpoint-reference/2', 'sha256': None, 'model_identity': revision,
        'generation': generation, 'weights_uri': None,
        'storage': 'one mutable working GGUF; past generations retain metadata only',
        'identity_source': 'controller run/update label, not a content digest',
        'verification': 'Observer reads the controller status/revision and file presence only; no weight hashing.',
        'role': 'installed update', 'training_step': step,
        'original': {'uri': original.as_uri(), 'sha256': revision['original_sha256'], **present(original)},
        'full_copy': {'present': False, 'bytes': None}, 'availability': 'metadata_only_not_retained',
    }
    try:
        current = json.loads((file.parent / 'state.json').read_text())
        current_revision = record_identity(current)
        record_generation(current)
        if (current_revision['run'] != revision['run']
                or current_revision['original_sha256'] != revision['original_sha256']
                or current['status'] not in ('ready', 'writing')):
            raise ValueError('current progress must identify the same run and original')
        identity['working_status'] = current['status']
        if current_revision == revision:
            identity['availability'] = 'unavailable'
            if current['status'] == 'ready':
                observed = present(file)
                if observed['present']:
                    identity.update(availability='current_working_file_present', weights_uri=file.as_uri(), full_copy=observed)
        elif current_revision['generation'] <= generation:
            raise ValueError('current progress does not identify this run or a newer revision')
    except (OSError, ValueError, KeyError, TypeError) as error:
        identity.update(availability='unavailable', availability_error=f'{type(error).__name__}: {error}')
    return identity, {'reached.json': encoded(reached), 'checkpoint.json': encoded(identity)}


def present(path):
    try:
        return {'present': path.is_file(), 'bytes': path.stat().st_size}
    except FileNotFoundError:
        return {'present': False, 'bytes': None}


def checkpoint_identity(root, declaration, *, reached=None, step=0, starting_measurement=None):
    root = Path(root).resolve()
    if starting_measurement is not None:
        return ram_revision(root, declaration, starting_measurement, step, initial=True)
    if reached and reached.get('model_identity') is not None:
        return working_checkpoint(root, declaration, reached, step)
    baseline = declaration['original_model']
    initial = declaration.get('working_model', {}).get('start') if not reached else None
    original = root / baseline['record']
    file = root / (reached['file'] if reached else initial['file'] if initial else baseline['record'])
    sha = reached['model_sha256'] if reached else initial['model_sha256'] if initial else baseline['sha256']
    identity = {
        'format': 'external-checkpoint-reference/1', 'sha256': sha,
        'weights_uri': file.as_uri(), 'full_copy': present(file),
        'storage': 'external GGUF or compact reconstruction; MLflow holds metadata only',
        'identity_source': ('controller reached-state record' if reached else
                            'bound retained starting model' if initial else 'bound original-model declaration'),
        'verification': 'Weight identity is controller-bound. The observer checks metadata hashes and file sizes, not weight bytes.',
        'role': 'installed update' if reached else 'retained initial model' if initial else 'original input model',
        'training_step': step,
        'original': {'uri': original.as_uri(), 'sha256': baseline['sha256'], **present(original)},
    }
    artifacts = {}
    if reached:
        artifacts['reached.json'] = encoded(reached)
        pointer = file.parent / 'compact.json'
        if pointer.is_file():
            try:
                pointer_bytes = pointer.read_bytes()
                binding = json.loads(pointer_bytes)
                record_path = root / binding['record']
                if record_path.resolve() != (file.parent / 'state-tensors/record.json').resolve():
                    raise ValueError('compact binding belongs to another checkpoint directory')
                record_bytes = record_path.read_bytes()
                if hashlib.sha256(record_bytes).hexdigest() != binding['sha256']:
                    raise ValueError('compact record does not match its bound hash')
                record = json.loads(record_bytes)
                if (record.get('format') != 'state-tensors/1' or record.get('model_sha256') != sha
                        or record.get('original_sha256') != baseline['sha256']):
                    raise ValueError('compact record identifies a different model or original')
                tensors = record_path.parent / 'tensors.bin'
                observed = present(tensors)
                size_matches = observed['present'] and observed['bytes'] == record['bytes']
                identity['compact'] = {
                    'pointer_uri': pointer.as_uri(), 'record_uri': record_path.as_uri(),
                    'record_sha256': binding['sha256'], 'record_hash_verified': True,
                    'tensors_uri': tensors.as_uri(), 'tensors_sha256': record['tensors_sha256'],
                    'tensors_present': observed['present'], 'tensors_bytes': observed['bytes'],
                    'tensors_size_matches_record': size_matches,
                    'components_present': bool(size_matches and identity['original']['present']),
                    'restore_with': 'pipeline.weight_edit.state_tensors.restore; requires the original and bound tensor layout',
                }
                artifacts['compact-pointer.json'] = pointer_bytes
                artifacts['compact-record.json'] = record_bytes
            except (OSError, ValueError, KeyError, TypeError) as error:
                identity['compact_error'] = f'{type(error).__name__}: {error}'
    compact_present = identity.get('compact', {}).get('components_present', False)
    identity['availability'] = ('full_and_compact_present' if identity['full_copy']['present'] and compact_present
                                else 'full_file_present' if identity['full_copy']['present']
                                else 'compact_components_present' if compact_present else 'unavailable')
    artifacts['checkpoint.json'] = encoded(identity)
    return identity, artifacts


def _version(client, name, key, tag='checkpoint_sha256'):
    token = None
    while True:
        page = client.search_model_versions(f"name = '{name}'", max_results=1000, page_token=token)
        for version in page:
            if version.tags.get(tag) == key:
                return version
        token = getattr(page, 'token', None)
        if not token:
            return None


def link_model(client, root, declaration, parent, experiment, state, checkpoint,
               *, reached=None, step=0, starting_measurement=None):
    from mlflow.entities import LoggedModelInput, LoggedModelOutput
    from mlflow.exceptions import MlflowException
    identity, artifacts = checkpoint_identity(root, declaration, reached=reached, step=step,
                                            starting_measurement=starting_measurement)
    sha = identity['sha256']
    typed = identity.get('model_identity')
    key = f'{typed["kind"]}:{typed["run"]}:{identity["generation"]}' if typed else sha
    tag = 'checkpoint_revision_key' if typed else 'checkpoint_sha256'
    model_tags = {tag: key, 'storage': 'external_reference', 'role': identity['role']}
    if typed:
        model_tags.update(checkpoint_identity_kind=typed['kind'], checkpoint_revision_run=typed['run'],
                          checkpoint_generation=str(identity['generation']))
    name = 'weight-patterns-' + Path(declaration['original_model']['record']).stem.replace('_', '-')
    cache = state.setdefault('models', {})
    held = cache.get(key)
    if held is None:
        models = client.search_logged_models([experiment],
            filter_string=f"tags.{tag} = '{key}'", max_results=1)
        model = models[0] if models else client.create_logged_model(
            experiment, name=(f'working-state-{identity["generation"]}' if typed else Path(identity['weights_uri']).stem),
            source_run_id=parent if reached else None,
            model_type='external-working-state-reference' if typed else 'external-gguf-reference',
            tags=model_tags,
            params=({'revision_kind': typed['kind'], 'revision_run': typed['run'],
                     'generation': str(identity['generation']), 'original_sha256': typed['original_sha256']} if typed else
                    {'weights_uri': identity['weights_uri'], 'sha256': sha}))
        held = cache[key] = dict(model_id=model.model_id, artifact_uri=model.artifact_location,
                                registry_name=name, sha256=sha, weights_uri=identity['weights_uri'],
                                role=identity['role'])
        checkpoint()
    held.update(weights_uri=identity['weights_uri'], identity_kind=typed['kind'] if typed else 'file-sha256')
    if typed:
        held['generation'] = identity['generation']
    if 'artifact_uri' not in held:
        held['artifact_uri'] = client.get_logged_model(held['model_id']).artifact_location

    # The registry points to metadata, never a disposable full copy. Availability
    # is refreshed independently of the immutable reached-record metric receipt.
    digest = hashlib.sha256(b''.join(key.encode() + artifacts[key] for key in sorted(artifacts))).hexdigest()
    changed = held.get('manifest_sha256') != digest
    if changed:
        with tempfile.TemporaryDirectory(prefix='wp-model-reference-') as directory:
            for filename, payload in artifacts.items():
                (Path(directory) / filename).write_bytes(payload)
            client.log_model_artifacts(held['model_id'], directory)
        client.finalize_logged_model(held['model_id'], 'READY')

    if 'version' not in held:
        try:
            client.get_registered_model(name)
        except MlflowException as error:
            if error.error_code != 'RESOURCE_DOES_NOT_EXIST':
                raise
            client.create_registered_model(name, description=
                'External GGUF checkpoint identities. Version artifacts are downloadable metadata, '
                'not packaged weights or an MLflow serving model. Check checkpoint.json for full-copy '
                'and compact-reconstruction availability. Typed working states retain only the current weights; '
                'their run/update label is not a content digest.')
        version = _version(client, name, key, tag)
        if version is None:
            version = client.create_model_version(name, source=held['artifact_uri'],
                run_id=parent if reached else None, model_id=held['model_id'],
                tags=model_tags,
                description=encoded(identity).decode(), await_creation_for=0)
        held['version'] = version.version
        checkpoint()
    if changed:
        client.set_logged_model_tags(held['model_id'], {'availability': identity['availability']})
        client.set_model_version_tag(name, held['version'], 'availability', identity['availability'])
        client.update_model_version(name, held['version'], description=encoded(identity).decode())
        held.update(manifest_sha256=digest, availability=identity['availability'])
        checkpoint()

    relation = f'{parent}:{step}:{identity["role"]}'
    if relation not in held.setdefault('linked', []):
        if reached:
            client.log_outputs(parent, [LoggedModelOutput(held['model_id'], step)])
        else:
            client.log_inputs(parent, models=[LoggedModelInput(held['model_id'])])
        held['linked'].append(relation)
        checkpoint()
    latest = state.get('model_latest_step', -1)
    if step >= latest:
        for field in ('model_id', 'registry_name', 'version', 'sha256', 'weights_uri', 'role',
                      'availability', 'artifact_uri', 'identity_kind'):
            client.set_tag(parent, 'model.' + field, '' if held[field] is None else str(held[field]))
        if typed:
            client.set_tag(parent, 'model.generation', str(held['generation']))
        alias = (f'run-{parent}-latest' if reached else
                 f'run-{parent}-initial' if identity['role'] == 'retained initial model' else 'original')
        if state.get('model_alias_key', state.get('model_alias_sha256')) != key:
            client.set_registered_model_alias(name, alias, held['version'])
            state['model_alias_key'] = key
        state['model_latest_step'] = step
        checkpoint()
    return held
