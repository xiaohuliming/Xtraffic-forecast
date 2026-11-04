"""CPU-only Mac consumer wiring, with explicit hand-built foreign-Windows facts.

These fixtures are NOT source/Torch/native checkpoint/Windows Handle evidence.
They intentionally have production-shaped metadata so the unmodified Mac
consumers can execute in pytest's temporary tree. Their non-pickle checkpoint
bytes are never loaded. No fixture output is a scientific seed1/2 result.
"""
from __future__ import annotations
import ast
import builtins
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PureWindowsPath
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pytest
from scripts.remote.windows import graph_h240_confirmation as q
from scripts import retrieve_graph_h240_confirmation as c

B = Path(os.environ.get('XTRAFFIC_GRAPH_AUTHORITY_ROOT', '/Users/xhlm/Desktop/Study/科研实习/outputs/remote_runs/ndr_generalization_20261013'))
N8 = Path(os.environ.get('XTRAFFIC_GRAPH_N8_WORKTREE', '/Users/xhlm/.codex/worktrees/shared-sta-causal-norm-sparse/科研实习'))
SCOPE = 'hand_built_CPU_mechanical_Mac_consumer_fixture_not_source_native_OS_or_scientific_evidence'
SEALED = B / 'formal7_complete30_chunk_v2_recovery_terminal_9fb856790b334089aef9df40e4ce6bc4/complete_preservation/source_host'
OPAQUE = B / 'control/opaque_chunk_transfer_transport_v2_20261016.py'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes): path.write_bytes(value)
    else: path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
    return path


def record(path): return {'path': str(path), 'sha256': q.sha(path)}


def original_helpers():
    path = Path(__file__).with_name('test_graph_h240_confirmation.py')
    spec = importlib.util.spec_from_file_location('_graph_original_metadata_fixture', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def available_paths():
    paths = [OPAQUE, SEALED / 'control/source_preparation.json',
        B / 'control/next_event_cost_97440a5_complete_source.zip',
        B / 'formal7_complete30_discovery_root_2d98b34da51f4bd7bb966a202de0ae78/complete30_verify_only_discovery_analysis.json',
        N8 / 'tests/ndr_research/test_shared_sta_norm_serial_queue.py', q._ENGINE_LIBRARY_PATH, c._LIBRARY_PATH]
    if any(not path.is_file() for path in paths): pytest.skip('private frozen opaque/source43/seed42 CPU fixtures are unavailable')


def declared_bytes():
    """Read only actual frozen code/authority bytes, never checkpoint tensors."""
    manifest = q.read(Path(__file__).resolve().parents[2] / 'docs/audits/graph_h240_confirmation_runtime_closure_2026-10-16.json')
    indexed = {}
    def visit(value):
        if isinstance(value, dict):
            if 'path' in value and 'sha256' in value:
                label=value['path']
                # Foreign source labels are provenance, never local I/O targets.
                if isinstance(label,str) and not PureWindowsPath(label).is_absolute():
                    path=Path(label)
                else:
                    path=None
                if path is not None and path.is_absolute() and path.is_file():
                    raw = path.read_bytes()
                    if hashlib.sha256(raw).hexdigest() == value['sha256']: indexed[value['sha256']] = raw
            for child in value.values(): visit(child)
        elif isinstance(value, list):
            for child in value: visit(child)
    visit(manifest)
    authority_paths = [
        SEALED / 'control/original_plan.json', SEALED / 'control/source_preparation.json', SEALED / 'control/discovery_policy.json',
        SEALED.parent / 'receipts/2d98b34da51f4bd7bb966a202de0ae78_completed.json',
        B / 'formal7_complete30_discovery_root_2d98b34da51f4bd7bb966a202de0ae78/complete30_verify_only_discovery_analysis.json']
    for path in authority_paths:
        raw = path.read_bytes(); indexed[hashlib.sha256(raw).hexdigest()] = raw
    with zipfile.ZipFile(B / 'control/next_event_cost_97440a5_complete_source.zip') as archive:
        for name in manifest['source43_original_files_sha256']:
            raw = archive.read(name)
            assert hashlib.sha256(raw).hexdigest() == manifest['source43_original_files_sha256'][name]
            indexed[hashlib.sha256(raw).hexdigest()] = raw
    for path in (Path(q.__file__), Path(c.__file__)):
        raw=path.read_bytes(); indexed[hashlib.sha256(raw).hexdigest()]=raw
    return indexed


def original_observations(plan, state):
    # This is a mechanical data factory, not a validator or an OS probe.
    tree = ast.parse((N8 / 'tests/ndr_research/test_shared_sta_norm_serial_queue.py').read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'observation_fixture')
    namespace = {'q': q.libraries()}
    exec(compile(ast.Module(body=[function], type_ignores=[]), 'original_CPU_observation_factory', 'exec'), namespace)
    value = namespace['observation_fixture'](plan, state)
    value.pop('files')
    return value


def prepare_mechanical_wire(tmp_path, *, failed_region=False, synthetic_only=False):
    """Create only temporary hand-built native/OS declarations and plain bytes.

    synthetic_only=False is a production SHAPE, explicitly not a production
    admission. No source admission, constructor, Torch load or OS process runs.
    """
    available_paths(); h = original_helpers(); indexed = declared_bytes()
    plan = h.plan.__wrapped__(tmp_path)
    plan.update(synthetic_only=synthetic_only, fixture_scope=SCOPE,
        run_root='C:/CPU_fixture/results', supervision_root='C:/CPU_fixture/supervision')
    for key in list(plan):
        if isinstance(plan[key], dict) and set(plan[key]) == {'path', 'sha256'}:
            plan[key]['path'] = 'C:/CPU_fixture/controls/' + key + ('.json' if key in {'formal7_plan','source_preparation','discovery_analysis','seed42_preservation','discovery_policy','gui_policy'} else '.py')
    plan['controller']['sha256'] = q.sha(Path(q.__file__))
    plan['collector']['sha256'] = q.sha(Path(c.__file__))
    plan['control_files'] = {row['path']: row['sha256'] for row in plan.values() if isinstance(row,dict) and set(row) == {'path','sha256'}}
    for role, rows in plan['predecessors'].items():
        raw = (json.dumps({'fixture_scope':SCOPE,'role':role}, sort_keys=True)+'\n').encode()
        digest=hashlib.sha256(raw).hexdigest(); indexed[digest]=raw
        rows['raw']={'path':'C:/CPU_fixture/predecessors/'+role+'.json','sha256':digest}
    plan['outputs']={key:plan['run_root']+'/'+name for key,name in q.OUTPUTS.items()}
    h.seal(plan); q.validate_plan(plan)
    state,journal,_post,native=h.complete(plan)
    state['fixture_scope']=native['fixture_scope']=SCOPE
    mirror=tmp_path/'foreign_source_mirror'
    def source(label):
        assert label.startswith('C:/')
        return mirror / label[3:]
    results=source(plan['run_root'])
    for index,(name,kind,_region,hidden,seed) in enumerate(q.CONFIGS):
        base=q.BASE_IDS[0 if kind=='pooled_graph' else 1]
        directory=results/'tasks'/name
        config=q.read(SEALED/'results/tasks'/base/'config.json');config['seed']=seed
        digest=q.canonical(config)
        proof=q.read(SEALED/'results/tasks'/base/'independent_verify/verification.json')
        proof['config_digest']=digest
        for region,leaf in proof['source_validation']['regions'].items():
            value=10.0 if kind=='pooled_graph' else 11.0
            if failed_region and seed==2 and kind=='pooled_graph' and region=='ContraCosta': value=12.0
            leaf['global_masked_mae']=value;leaf['absolute_error_sum']=value*leaf['valid_count']
        best=write(directory/'best.pt',('NOT_PICKLE_NOT_TRAINED_CPU_MECHANICAL '+name+' best\n').encode())
        last=write(directory/'last.pt',('NOT_PICKLE_NOT_TRAINED_CPU_MECHANICAL '+name+' last\n').encode())
        proof['checkpoint_sha256']=q.sha(best)
        write(directory/'config.json',config);write(directory/'independent_verify/verification.json',proof)
        write(directory/'summary.json',{'fixture_scope':SCOPE,'config_digest':digest,'completed_epochs':30})
        write(directory/'data_identity.json',config['data_identity']);write(directory/'source_snapshot.json',config['source_snapshot'])
        write(directory/'training_diagnostics.jsonl',{'fixture_scope':SCOPE,'not_training_history':True})
        for path in (best,last):write(path.with_suffix('.pt.sha256.json'),{'fixture_scope':SCOPE,'sha256':q.sha(path),'config_digest':digest,'size_bytes':path.stat().st_size})
        unit=native['native_units'][name]
        unit.update(fixture_scope=SCOPE,constructor=config['model_config'],config_digest=digest,
            checkpoint_sha256={'best.pt':q.sha(best),'last.pt':q.sha(last)})
        for row in unit['artifacts'].values():row['sha256']=q.sha(results/row['path'])
        for row in state['jobs'][index*2:index*2+2]:
            row['native_admission']['artifacts']={path:q.sha(results/path) for path in row['native_admission']['artifacts']}
    # Rebind durable accepted rows to their actual temporary opaque artifact bytes.
    for index,event in enumerate(journal):
        if event['event']=='owned_wait_return':
            job=next(row for row in state['jobs'] if row['job_id']==event['job_id'])
            journal[index]={**job,'event':'owned_wait_return'}
    native['owned_wait_rows_sha256']=q.canonical(state['jobs'])
    native_path=write(results/'source_native_admission.json',native)
    state['source_native_receipt_sha256']=q.sha(native_path)
    write(results/'status.json',state)
    (results/'process_journal.jsonl').write_text('\n'.join(json.dumps(row,sort_keys=True) for row in journal)+'\n')
    u=c.library(plan)
    post={'schema_version':q.POST_SCHEMA,'stage':q.STAGE,'plan_digest':plan['plan_digest'],'status':'completed',
        'accepted_job_ids':state['completed'],'owned_wait_rows_sha256':q.canonical(state['jobs']),
        'source_native_receipt_sha256':q.sha(native_path),'files':list(u.inventory_tree(results).values())}
    write(results/'postflight.json',post)
    terminal=q.validate_terminal(plan,state,journal,post,native)
    observations=original_observations(plan,state)
    supervisor=observations['infrastructure_raw']['supervisor_terminal']
    infra=observations['infrastructure_raw']['infrastructure_waits']
    parent=write(source(plan['supervision_root'])/'launcher_parent_wait.json',infra['queue_parent_wait'])
    infra['queue_parent_wait_binding']['sha256']=q.sha(parent)
    for role,value in [('supervisor_terminal',supervisor),('infrastructure_waits',infra)]:
        path=write(source(plan['supervision_root'])/(role+'.json'),value)
        observations['infrastructure_raw_bindings'][role]['sha256']=q.sha(path)
    plan_path=write(source('C:/CPU_fixture/control/plan.json'),plan)
    all_rows={};locations={}
    def add(name,label,raw=None):
        path=source(label)
        if raw is not None:write(path,raw)
        all_rows[name]={'path':name,'bytes':path.stat().st_size,'sha256':q.sha(path)};locations[name]=label
    for name in u.inventory_tree(results):add('results/'+name,plan['run_root']+'/'+name)
    add('control/plan.json','C:/CPU_fixture/control/plan.json')
    bindings={}
    for name,row in u._expected_controls(plan).items():
        if row['sha256'] not in indexed:pytest.skip('actual frozen byte archive missing '+row['sha256'])
        add(name,row['source_path'],indexed[row['sha256']])
        bindings[name]={'expected_sha256':row['sha256'],'observed_sha256':row['sha256']}
    supervision=u.inventory_tree(source(plan['supervision_root']))
    for name in supervision:add('control/supervision/'+name,plan['supervision_root']+'/'+name)
    files=[all_rows[name] for name in sorted(all_rows)]
    frame={'schema_version':q.INVENTORY_SCHEMA,'observed_at_utc':'2026-10-16T00:00:00+00:00',
        'stage':q.STAGE,'plan_digest':plan['plan_digest'],'plan_sha256':q.sha(plan_path),
        'roots':{'results':plan['run_root'],'control':plan['supervision_root']},'bound_locations':locations,
        'files':files,'files_digest':q.canonical(files),'supervision_files':list(supervision.values()),
        'status':state,'terminal':terminal,'preservation_mode':'strict_success',
        'source_git_observation':{'expected':plan['source_identity'],'observed':copy.deepcopy(plan['source_identity'])},
        'control_binding_observations':bindings,'postflight_sha256':q.sha(results/'postflight.json'),
        'source_native_receipt_sha256':q.sha(native_path),'native_admission':native,**observations,
        'fixture_scope':SCOPE}
    after=copy.deepcopy(frame);after['observed_at_utc']='2026-10-16T00:00:01+00:00'
    c.validate_inventory(frame,plan)
    q.validate_source_terminal_observations(plan,state,frame)
    scope=write(tmp_path/'fixture_scope.json',{'scope':SCOPE,'hand_written_native_receipt':True,'hand_written_OS_records':True,
        'actual_source_native_or_Torch_or_Handle_evidence':False,'checkpoint_format':'explicit_non_pickle_bytes',
        'scientific_output_authorized':False,'production_execution_plan':False})
    return SimpleNamespace(plan=plan,before=frame,after=after,mirror=mirror,scope=scope,tmp=tmp_path,results=results)


def collect(wire,monkeypatch, *, after=None):
    def forbidden(*_args,**_kwargs):raise AssertionError('SSH, worker, source inference or tensor deserialization is forbidden')
    monkeypatch.setattr(subprocess,'Popen',forbidden)
    original=builtins.__import__
    def opaque_only(name,*args,**kwargs):
        if name=='torch' or name.startswith(('torch.','ndr_research')) or name=='h5py':forbidden()
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',opaque_only)
    class NoRemoteConnection:
        @property
        def python(self):forbidden()
    destination=wire.tmp/'preserved'
    result=c.collect_stage(wire.plan,source_before=lambda:copy.deepcopy(wire.before),
        source_after=lambda:copy.deepcopy(wire.after if after is None else after),native_output=destination,
        opaque_path=OPAQUE,connection=NoRemoteConnection(),reuse_entries=[{'remote_root':'C:/','local_root':str(wire.mirror)}],token='cpuMechanicalOnly')
    return result,destination/'preservation.json'


def test_full_unstubbed_opaque_collect_admit_and_family6_Mac_consumer(tmp_path,monkeypatch):
    wire=prepare_mechanical_wire(tmp_path)
    preservation,path=collect(wire,monkeypatch)
    assert preservation['status']=='complete_success_preserved' and preservation['production_launchable'] is False
    copied=Path(preservation['source_host_tree']['path'])
    assert c.admit_copied(copied,wire.before,wire.plan)==wire.before['terminal']
    transport=q.read(q.pin(preservation['transport_manifest']))
    assert len(transport['requests'])==len(wire.before['files'])
    for row in transport['requests']:
        assert row['remote_path']==wire.before['bound_locations'][row['path']]
        assert q.sha(Path(row['mac_promoted_target']))==row['sha256']
    completions=list((wire.tmp/'preserved/transport_audit/cpuMechanicalOnly').glob('*/completion.json'))
    assert len(completions)==len(transport['requests'])
    assert all(q.read(path)['method']=='explicit_mapped_whole_file_sha256_reuse' for path in completions)
    result=c.family6(path,seed42_root=SEALED,output=tmp_path/'CPU_only_not_scientific_family.json')
    assert result['models']==6 and result['new_original_owned_wait0']==8
    assert result['status']=='confirmed_each_seed_both_sources' and result['production_launchable'] is False
    assert q.read(wire.scope)['actual_source_native_or_Torch_or_Handle_evidence'] is False
    assert (copied/'results/tasks/pooled_graph_h240_seed1/best.pt').read_bytes().startswith(b'NOT_PICKLE_NOT_TRAINED')


def test_full_family6_failure_cannot_be_offset_by_other_region_or_seed(tmp_path,monkeypatch):
    wire=prepare_mechanical_wire(tmp_path,failed_region=True)
    _preservation,path=collect(wire,monkeypatch)
    result=c.family6(path,seed42_root=SEALED,output=tmp_path/'CPU_only_negative_not_scientific_family.json')
    assert result['status']=='confirmation_failed_preserve_negative_result'
    assert result['per_seed_both_region_strict']['2']=={'Alameda':True,'ContraCosta':False}
    assert result['per_seed_both_region_strict']['1']=={'Alameda':True,'ContraCosta':True}


def test_full_family6_retains_real_synthetic_only_rejection(tmp_path,monkeypatch):
    wire=prepare_mechanical_wire(tmp_path,synthetic_only=True)
    _preservation,path=collect(wire,monkeypatch)
    output=tmp_path/'must_not_create_family.json'
    with pytest.raises(ValueError,match='synthetic CPU protocol evidence'):c.family6(path,seed42_root=SEALED,output=output)
    assert not output.exists()


def test_second_frame_change_cannot_promote_or_write_family(tmp_path,monkeypatch):
    wire=prepare_mechanical_wire(tmp_path)
    after=copy.deepcopy(wire.after);after['status']['counts']['owned_wait0']=7
    with pytest.raises(ValueError,match='source changed between complete collection frames'):
        collect(wire,monkeypatch,after=after)
    assert not (tmp_path/'preserved/source_host').exists()
    assert not (tmp_path/'preserved/preservation.json').exists()


def test_corrupt_opaque_source_hits_CPU_network_trap_and_cannot_preserve(tmp_path,monkeypatch):
    wire=prepare_mechanical_wire(tmp_path)
    path=wire.results/'tasks/pooled_graph_h240_seed1/best.pt'
    path.write_bytes(b'CORRUPTED_NOT_PICKLE_CPU_FIXTURE')
    with pytest.raises(AssertionError,match='SSH, worker, source inference or tensor deserialization is forbidden'):
        collect(wire,monkeypatch)
    assert not (tmp_path/'preserved/source_host').exists()
    assert not (tmp_path/'preserved/preservation.json').exists()


def test_family6_rejects_mutated_copied_checkpoint_before_metric_output(tmp_path,monkeypatch):
    wire=prepare_mechanical_wire(tmp_path)
    preserved,path=collect(wire,monkeypatch)
    copied=Path(preserved['source_host_tree']['path'])
    checkpoint=copied/'results/tasks/pooled_graph_h240_seed1/best.pt'
    checkpoint.write_bytes(b'CHANGED_NOT_PICKLE_CPU_FIXTURE')
    output=tmp_path/'must_not_write_scientific_family.json'
    with pytest.raises(ValueError,match='tree'):
        c.family6(path,seed42_root=SEALED,output=output)
    assert not output.exists()


def test_admit_copied_checks_original_wait_even_with_rebound_exact_opaque_files(tmp_path,monkeypatch):
    wire=prepare_mechanical_wire(tmp_path)
    preserved,_path=collect(wire,monkeypatch)
    root=Path(preserved['source_host_tree']['path']);u=c.library(wire.plan)
    status=q.read(root/'results/status.json');native=q.read(root/'results/source_native_admission.json')
    journal=[json.loads(line) for line in (root/'results/process_journal.jsonl').read_text().splitlines()]
    status['jobs'][0]['terminal_owned_wait_proof']['wait_returncode']=None
    for index,row in enumerate(journal):
        if row['event']=='owned_wait_return' and row['job_id']==status['jobs'][0]['job_id']:
            journal[index]={**status['jobs'][0],'event':'owned_wait_return'}
    native['owned_wait_rows_sha256']=q.canonical(status['jobs'])
    write(root/'results/source_native_admission.json',native)
    status['source_native_receipt_sha256']=q.sha(root/'results/source_native_admission.json')
    write(root/'results/status.json',status)
    (root/'results/process_journal.jsonl').write_text('\n'.join(json.dumps(row,sort_keys=True) for row in journal)+'\n')
    post=q.read(root/'results/postflight.json')
    post.update(owned_wait_rows_sha256=q.canonical(status['jobs']),source_native_receipt_sha256=status['source_native_receipt_sha256'],
        files=[row for name,row in u.inventory_tree(root/'results').items() if name!='postflight.json'])
    write(root/'results/postflight.json',post)
    frame=copy.deepcopy(wire.after);frame.update(status=status,native_admission=native,
        postflight_sha256=q.sha(root/'results/postflight.json'),source_native_receipt_sha256=status['source_native_receipt_sha256'])
    frame['files']=list(u.inventory_tree(root).values());frame['files_digest']=q.canonical(frame['files'])
    # Every copied byte, raw source frame and native receipt reference is now
    # coherent; only the original held Handle wait fact is deliberately null.
    with pytest.raises(ValueError,match='genuine original held Handle wait0'):
        c.admit_copied(root,frame,wire.plan)


def test_code_archive_index_never_opens_foreign_Windows_labels(monkeypatch):
    available_paths()
    actual_Path=Path
    def local_only(label,*args):
        if isinstance(label,str) and PureWindowsPath(label).is_absolute():
            raise AssertionError('foreign Windows label reached local Path I/O')
        return actual_Path(label,*args)
    monkeypatch.setattr(sys.modules[__name__],'Path',local_only)
    indexed=declared_bytes()
    assert q.ANALYSIS_SHA in indexed and q.FORMAL_PLAN_SHA in indexed
