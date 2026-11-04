"""CPU protocol and native-schema tests. No production plan or GPU is created."""
from __future__ import annotations
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import subprocess
from types import SimpleNamespace

import pytest
from scripts.remote.windows import graph_h240_confirmation as q
from scripts import retrieve_graph_h240_confirmation as c

B = Path(os.environ.get('XTRAFFIC_GRAPH_AUTHORITY_ROOT', '/Users/xhlm/Desktop/Study/科研实习/outputs/remote_runs/ndr_generalization_20261013'))
COST = Path(os.environ.get('XTRAFFIC_GRAPH_COST_WORKTREE', '/Users/xhlm/.codex/worktrees/ndr-cost-profiling/科研实习'))
N8 = Path(os.environ.get('XTRAFFIC_GRAPH_N8_WORKTREE', '/Users/xhlm/.codex/worktrees/shared-sta-causal-norm-sparse/科研实习'))


def require_paths(*paths):
    if any(not Path(path).is_file() for path in paths):
        pytest.skip('private frozen authority/dependency archive is not available')


def seal(value):
    value['plan_digest'] = q.canonical({k: v for k, v in value.items() if k != 'plan_digest'})
    return value


def file_record(path):
    return {'path': str(path), 'sha256': q.sha(path)}


@pytest.fixture
def plan(tmp_path):
    old_path = B / 'control/formal7_frozen_execution_plan_20261015T050849Z_6b9501f945.json'
    if not old_path.exists(): pytest.skip('actual b4f command authority is not present')
    require_paths(B / 'control/source_preparation_host_receipt_d3cbbfb05aa24038b78c4a85ae0782b5.json', q._ENGINE_LIBRARY_PATH)
    old = q.read(q.pin(file_record(old_path), q.FORMAL_PLAN_SHA))
    prep = q.read(B / 'control/source_preparation_host_receipt_d3cbbfb05aa24038b78c4a85ae0782b5.json')
    result = {'schema_version': q.PLAN_SCHEMA, 'stage': q.STAGE, 'synthetic_only': True, 'budget_frozen': True,
        'total_budget_seconds': 123, 'absolute_deadline_utc': '2099-01-01T00:00:00+00:00', 'current_boot_utc': '2026-01-01T00:00:00+00:00',
        'source_identity': {'git_sha': q.SOURCE_GIT, 'git_tree': 'b'*40, 'git_dirty': False},
        'original_job_templates': {row['job_id']: row['command'] for row in old['jobs'] if row['job_id'] in
            ('pooled_graph_h240', 'pooled_graph_h240_verify', 'pooled_shared_sta', 'pooled_shared_sta_verify')},
        'shared_producer_files': prep['source_files'], 'control_files': {}, 'minimum_free_gpu_mib': 14000,
        'run_root': str(tmp_path / 'results'), 'supervision_root': str(tmp_path / 'supervision'),
        'predecessors': {name: {'raw': {'path': 'C:/fixture/' + name + '.json', 'sha256': 'c'*64}} for name in
            ('whole7_verify_only','dualcast_numerical','dualcast_complete_cost','strong4_fixed30')},
        **{key: old[key] for key in ('python','repo','data_dir','graph_dir')},
        **{key: False for key in ('automatic_retry','automatic_resume','automatic_successor_dispatch','automatic_shutdown','automatic_transfer','test_evaluated','Orange_evaluated')}}
    constants = {'serial_engine':q.ENGINE_SHA, 'collector_library':q.COLLECTOR_LIBRARY_SHA, 'predecessor_helper':q.PREDECESSOR_SHA,
        'successor_adapter':q.ADAPTER_SHA, 'native_core':q.NATIVE_CORE_SHA, 'formal7_plan':q.FORMAL_PLAN_SHA,
        'source_preparation':q.PREPARATION_SHA, 'discovery_analysis':q.ANALYSIS_SHA, 'seed42_preservation':q.SEED42_SHA, 'discovery_policy':q.POLICY_SHA,
        'process_helper':'808abd383d6d40fb5dd5f8b6aa02af9d39f49f3ac1e8b41967d4baab78dd1d60',
        'windows_supervisor_primitives':'5d78666a4053b03231c6a96475021209f55de574fe52704075d808a604d7c7c0',
        'gpu_provider':'23c999a6e4321da1d3298d6eb9e9682798e112648b5e2f6751a3850a9e052015',
        'gui_implementation':'d59ec43e58fe0b9575ebe8ee9fa8d454aa100cc2931ebe8bd1375b8279dc24ba',
        'gui_policy':'d9dcd0c20aadb6cdf39b90fcb9b8963f7251a4f9c7fc100364ceea27648463c9', 'controller':'a'*64, 'collector':'b'*64}
    for key,digest in constants.items():
        result[key] = {'path':'C:/fixture/'+key+'.py','sha256':digest}
        result['control_files'][result[key]['path']] = digest
    result['jobs'] = [{'job_id':name,'timeout_seconds':17} for name,_ in q.STAGES[q.STAGE]]
    result['outputs'] = {key:result['run_root']+'/'+name for key,name in q.OUTPUTS.items()}
    return seal(result)


def complete(plan):
    e = q.libraries(); state = e._bootstrap(plan)
    state.update(status='completed', queue_instance={'pid':90,'creation_filetime_100ns':90}, pid=90,
        source_native_receipt_sha256='b'*64, predecessor_admissions={
            'whole7_verify_only': {'formal_process_counts':dict(e.OLD_COUNTS),'original_reboot_exit_code':None,'completed30_producer_owned_wait_returncode':None},
            'dualcast_numerical':{},'dualcast_complete_cost':{'status':'admitted_preserved_dualcast_memory_cost_terminal'},'strong4_fixed30':{'status':'admitted_complete_source_native_strong4_fixed30','pairs':4,'native_owned_Popen_rows':8,'original_held_wait_proof_synthesized':False}})
    journal=[]
    for index,(job_id,role) in enumerate(q.STAGES[q.STAGE]):
        attempt={'job_id':job_id,'role':role,'pid':100+index,'creation_filetime_100ns':100+index,'executable_path':plan['python'],
            'command_sha256':q.canonical(q.expected_jobs(plan)[index]),'boot_utc':plan['current_boot_utc'],'Popen_creationflags':8,
            'captured_owned_handle_value':300+index,'external_deadline_arm':{'job_id':job_id,'armed_monotonic':index*2,'deadline_monotonic':index*2+17,
                'armed_filetime_100ns':1,'timeout_seconds':17,'plan_digest':plan['plan_digest'],'session_id':'synthetic'}}
        state['attempts'].append(copy.deepcopy(attempt));state['counts'][role+'_attempts']+=1
        for event in ('before_popen','owned_handle_created_before_identity_query','after_popen'): journal.append(attempt|{'event':event})
        raw=attempt|{'event':'owned_wait_result_observed','returncode':0,'child_returncode':0};journal.append(raw)
        proof={'schema_version':'held_handle_terminal_wait_proof_v1',**{k:attempt[k] for k in ('pid','creation_filetime_100ns','captured_owned_handle_value','executable_path')},
            'executable_identity_basis':'strict_prewait_owned_capture','same_captured_owned_handle':True,'wait_returncode':0,'GetExitCodeProcess':0}
        task_name=job_id.removesuffix('_verify')
        suffixes=['independent_verify/verification.json'] if job_id.endswith('_verify') else ['best.pt','last.pt','best.pt.sha256.json','last.pt.sha256.json','summary.json']
        captured={'tasks/'+task_name+'/'+suffix:'d'*64 for suffix in suffixes}
        row=attempt|{'returncode':0,'process_wall_seconds':1,'raw_wait_observation_sha256':q.canonical(raw),'terminal_owned_wait_proof':proof,
                     'native_admission':{'scope':'pre_native_artifact_identity_only','artifacts':captured}}
        state['jobs'].append(row);state['completed'].append(job_id);state['counts']['owned_wait0']+=1
        journal.append(row|{'event':'owned_wait_return'})
    units={}
    roles={'best':'best.pt','last':'last.pt','verification':'independent_verify/verification.json','summary':'summary.json','best_marker':'best.pt.sha256.json','last_marker':'last.pt.sha256.json',
           'config':'config.json','history':'training_diagnostics.jsonl','data_identity':'data_identity.json','snapshot':'source_snapshot.json'}
    for index,(name,kind,_,hidden,seed) in enumerate(q.CONFIGS):
        units[name]={'seed':seed,'kind':kind,'completed_epochs':30,'native_checkpoint_schemas_loaded_on_cpu':True,'inference_repeated_by_controller':False,'test_evaluated':False,
            'producer_emits_formal_pid_fields':False,'constructor':{'synthetic':True},'config_digest':'a'*64,'checkpoint_sha256':{'best.pt':'d'*64,'last.pt':'d'*64},
            'training_pid':100+2*index,'verification_pid':101+2*index,'independence_evidence':'distinct_owned_serial_popen_handles',
            'artifacts':{key:{'path':'tasks/'+name+'/'+suffix,'sha256':'d'*64} for key,suffix in roles.items()}}
    native={'schema_version':q.NATIVE_SCHEMA,'stage':q.STAGE,'plan_digest':plan['plan_digest'],'source_identity':plan['source_identity'],'shared_producer_files':plan['shared_producer_files'],
        'native_units':units,'metric_values_extracted_for_selection':False,'accepted_job_ids':state['completed'],'owned_wait_rows_sha256':q.canonical(state['jobs']),
        'predecessor_admissions':state['predecessor_admissions']}
    post={'schema_version':q.POST_SCHEMA,'stage':q.STAGE,'plan_digest':plan['plan_digest'],'status':'completed','accepted_job_ids':state['completed'],
        'owned_wait_rows_sha256':q.canonical(state['jobs']),'source_native_receipt_sha256':'b'*64,'files':[{'path':'source_native_admission.json','bytes':1,'sha256':'b'*64}]}
    return state,journal,post,native


def test_actual_b4f_clones_only_seed_output_and_keeps_both_microbatches(plan):
    assert q.validate_plan(plan) is plan
    jobs=q.expected_jobs(plan)
    for index,(name,kind,_,hidden,seed) in enumerate(q.CONFIGS):
        base=q.BASE_IDS[0 if kind=='pooled_graph' else 1]
        old=list(plan['original_job_templates'][base]);new=jobs[index*2]
        expected=list(old);expected[expected.index('--seed')+1]=str(seed)
        original=q.ORIGINAL_ROOT+'/tasks/'+base;target=plan['run_root']+'/tasks/'+name
        expected=[target+x[len(original):] if x==original or x.startswith(original+'/') else x for x in expected]
        assert expected==new
        assert new[new.index('--microbatch-size')+1]==('16' if kind=='pooled_graph' else '2')
        assert new[new.index('--epochs')+1]=='30' and new[new.index('--batch-size')+1]=='16'
        assert jobs[index*2+1][-1]==target+'/independent_verify'
    assert len(jobs)==8


@pytest.mark.parametrize('change',[lambda p:p['jobs'].reverse(),lambda p:p['jobs'].pop(),lambda p:p.update(total_budget_seconds=None),
    lambda p:p.update(budget_frozen=False),lambda p:p.update(test_evaluated=True),lambda p:p['original_job_templates']['pooled_shared_sta'].__setitem__(p['original_job_templates']['pooled_shared_sta'].index('--microbatch-size')+1,'16'),
    lambda p:p['predecessors'].pop('strong4_fixed30'),lambda p:p['process_helper'].update(sha256='0'*64),lambda p:p['shared_producer_files'].pop(next(iter(p['shared_producer_files'])))])
def test_no_recipe_scope_count_or_budget_weakening(plan,change):
    change(plan)
    with pytest.raises((ValueError,KeyError,TypeError)):q.validate_plan(seal(plan))


def test_complete_requires_eight_raw_waits_and_four_source_native_pairs(plan):
    state,journal,post,native=complete(plan)
    assert q.validate_terminal(plan,state,journal,post,native)['success'] is True
    with pytest.raises(ValueError):q.validate_terminal(plan,state,journal,post,None)


@pytest.mark.parametrize('change',[
    lambda s,j,p,n:s['counts'].update(owned_wait0=4),lambda s,j,p,n:s['jobs'][0]['terminal_owned_wait_proof'].update(executable_identity_basis='postexit_QueryFullProcessImageName'),
    lambda s,j,p,n:s['jobs'][0]['terminal_owned_wait_proof'].update(wait_returncode=None),lambda s,j,p,n:j.pop(0),
    lambda s,j,p,n:n['native_units'].pop('pooled_graph_h240_seed2'),lambda s,j,p,n:n['native_units']['pooled_shared_sta_seed1'].update(verification_pid=999),
    lambda s,j,p,n:n['native_units']['pooled_graph_h240_seed1'].update(seed=42),lambda s,j,p,n:s['predecessor_admissions']['whole7_verify_only'].update(original_reboot_exit_code=0),
    lambda s,j,p,n:n['native_units']['pooled_graph_h240_seed1']['checkpoint_sha256'].update({'best.pt':'e'*64}),lambda s,j,p,n:s.update(status='invented'),
    lambda s,j,p,n:s['jobs'][0]['native_admission']['artifacts'].update({'tasks/pooled_graph_h240_seed1/best.pt':'e'*64})])
def test_real_raw_validator_rejects_fabricated_counts_images_waits_or_native_units(plan,change):
    state,journal,post,native=complete(plan);change(state,journal,post,native)
    with pytest.raises((ValueError,KeyError,TypeError)):q.validate_terminal(plan,state,journal,post,native)


def test_os_loop_and_handle_code_are_original_with_only_explicit_namespace_adaptation(plan):
    e=q.libraries(); original=q.load({'path':str(q._ENGINE_LIBRARY_PATH),'sha256':q.ENGINE_SHA})
    for name in ('source_terminal_observations','validate_source_terminal_observations','_infrastructure_capture','_infrastructure_wait','fixed_launcher','dispatch','run_queue','watchdog_function'):
        assert getattr(e,name).__code__.co_code==getattr(original,name).__code__.co_code
        assert getattr(e,name).__code__.co_consts==getattr(original,name).__code__.co_consts
    assert e.DeadlineState.dispatch.__code__.co_code==original.DeadlineState.dispatch.__code__.co_code
    child=SimpleNamespace(wait=lambda **kw:(_ for _ in ()).throw(TimeoutError('blocked')),_handle=10)
    row=e._infrastructure_wait(child,SimpleNamespace(terminal=lambda handle:0),timeout=1)
    assert row['actual_wait_returncode'] is None and row['held_handle_signaled_exit_code'] is None


def test_deadline_state_uses_eight_graph_jobs_and_stops_only_owned_job(plan):
    require_paths(N8/'tests/ndr_research/test_shared_sta_norm_serial_queue.py')
    e=q.libraries()
    source=ast.parse((N8/'tests/ndr_research/test_shared_sta_norm_serial_queue.py').read_text())
    nodes=[node for node in source.body if isinstance(node,ast.ClassDef) and node.name in ('Clock','Backend')]
    namespace={};exec(compile(ast.Module(body=nodes,type_ignores=[]),'original_synthetic_backend','exec'),namespace)
    clock,backend=namespace['Clock'](),namespace['Backend']()
    config={'scope':q.SCOPE,'stage':q.STAGE,'jobs':plan['jobs'],'overall_seconds':123,'absolute_deadline_epoch':2000,'plan_digest':plan['plan_digest'],
        'policy_sha256':q.canonical(e.policy(plan)),'controller_record':plan['controller'],'primitive_record':plan['windows_supervisor_primitives'],
        'session_id':'fixture','watchdog_identity':{'pid':4,'creation_filetime_100ns':4},'dispatcher_identity':{'pid':5,'creation_filetime_100ns':5}}
    state=e.DeadlineState(config=config,backend=backend,monotonic=clock.mono,now=clock.now)
    def request(op,pid=2,**fields):return state.dispatch({'op':op,'caller_pid':pid,'caller_creation_filetime_100ns':backend.identities[pid]['creation_filetime_100ns'],'plan_digest':plan['plan_digest'],**fields})
    request('arm_overall',pid=1);request('attach_parent')
    with pytest.raises(ValueError):request('arm_job',job_id='static_train')
    request('arm_job',job_id=plan['jobs'][0]['job_id']);clock.value=18;state.check_deadline()
    assert state.terminal and backend.calls[-1]==('terminate',)
    assert state.events[-1]['event']=='TerminateJobObject_on_exclusively_owned_job'


def test_completed_predecessor_helper_has_only_successor_pin_substitution_and_no_fresh_admit():
    path=COST/'scripts/remote/windows/mage_train_only_cost_admission.py'
    require_paths(path)
    helper=q.load(file_record(path),q.PREDECESSOR_SHA)
    code=helper.strong4_completed_admission.__code__
    assert 'admit' not in code.co_names and 'admit_stage' not in code.co_names and 'load_frozen' in code.co_names
    assert helper.SUCCESSOR_SHA=='eec22a844575b95bb3d8874e95061fa207b531f560845c4ee3372001400a196c'
    original_globals=helper.predecessor_admission.__globals__.copy()
    predecessor,completed=q.completed_predecessor_functions(helper)
    assert predecessor.__code__ is helper.predecessor_admission.__code__ and completed.__code__ is code
    assert predecessor.__globals__ is completed.__globals__ and predecessor.__globals__ is not helper.__dict__
    assert predecessor.__globals__['SUCCESSOR_SHA']==q.ADAPTER_SHA
    assert helper.__dict__==original_globals and helper.SUCCESSOR_SHA=='eec22a844575b95bb3d8874e95061fa207b531f560845c4ee3372001400a196c'
    assert helper.strong4_completed_admission.__code__ is code
    assert helper.WORKER_SHA=='6f3c7ee89b88b97bdef5a81e7724e05dc5a2775088e164cfe18e6dae546b6991'


def test_failed_or_partial_preservation_never_opens_family_metric(tmp_path,monkeypatch):
    require_paths(c._LIBRARY_PATH)
    path=tmp_path/'preservation.json';path.write_text(json.dumps({'schema_version':c.PRESERVED_SCHEMA,'status':'complete_failure_preserved'}))
    with pytest.raises(ValueError,match='complete success'):c.family6(path,seed42_root=tmp_path,output=tmp_path/'family.json')
    assert not (tmp_path/'family.json').exists()


def test_mac_import_and_readiness_do_not_import_torch_or_open_cache():
    assert q.readiness()['production_launchable'] is False
    assert q.readiness()['budget_frozen_here'] is False
    code = "import runpy,sys;runpy.run_path(" + repr(str(Path(q.__file__))) + ");assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, '-B', '-c', code], check=True)


@pytest.fixture
def native_pair(tmp_path):
    native_path=COST/'scripts/remote/windows/ndr_formal_development_queue.py'
    require_paths(native_path, COST/'tests/ndr_research/test_formal_development_queue.py')
    native=q.load(file_record(native_path),q.NATIVE_CORE_SHA)
    spec=importlib.util.spec_from_file_location('_original_formal_test_metadata',COST/'tests/ndr_research/test_formal_development_queue.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    return native,helper,tmp_path


@pytest.mark.parametrize('index',[0,1,2,3])
def test_explicit_reseed_native_schema_keeps_original_full_history_loader_and_checkpoint_checks(native_pair,index):
    native,helper,root=native_pair
    name,kind,_,hidden,seed=q.CONFIGS[index];base=q.BASE_IDS[0 if kind=='pooled_graph' else 1]
    entry='scripts/train_ndr_transfer.py' if kind=='pooled_graph' else 'scripts/train_shared_staeformer_transfer.py'
    context={'root':root,'source_files':{entry:'a'*64},'native_core':native,'plan':{}}
    checkpoints=helper.synthetic_pair(context,{'job_id':base,'kind':kind,'entrypoint':entry})
    old=root/'tasks'/base;directory=root/'tasks'/name;old.rename(directory)
    config=checkpoints['best.pt']['config'];config['seed']=seed
    digest=q.canonical(config)
    for filename in ('config.json','summary.json','run_state.json','independent_verify/verification.json','best.pt.sha256.json','last.pt.sha256.json'):
        path=directory/filename;value=q.read(path)
        if filename=='config.json':value=config
        else:value['config_digest']=digest
        path.write_text(json.dumps(value))
    for checkpoint in checkpoints.values():checkpoint['config_digest']=digest
    seen=[]
    def loader(_context,_entry,path):seen.append(path.name);return checkpoints[path.name]
    result=q.validate_native_pair(context,{'job_id':name,'kind':kind,'entrypoint':entry},1000,1001,checkpoint_loader=loader)
    assert result['completed_epochs']==30 and seen==['best.pt','last.pt']
    checkpoints['last.pt']['loaders']['val']['Alameda']['cursor']-=1
    with pytest.raises(ValueError,match='omitted split samples'):
        q.validate_native_pair(context,{'job_id':name,'kind':kind,'entrypoint':entry},1000,1001,checkpoint_loader=loader)


@pytest.mark.parametrize('change',['missing','tie','one_region_fails','one_seed_fails','nan'])
def test_family6_all_seeds_and_both_regions_have_no_mean_offset(change):
    values={base+'_seed'+str(seed):{'Alameda':1.0 if base=='pooled_graph_h240' else 2.0,'ContraCosta':1.0 if base=='pooled_graph_h240' else 2.0}
            for base in q.BASE_IDS for seed in (42,1,2)}
    if change=='missing':
        values.pop('pooled_shared_sta_seed2')
        with pytest.raises(ValueError):c.compare_family6(values)
    elif change=='nan':
        values['pooled_graph_h240_seed1']['Alameda']=float('nan')
        with pytest.raises(ValueError):c.compare_family6(values)
    else:
        values['pooled_graph_h240_seed1']['ContraCosta']=2.0 if change=='tie' else 3.0
        result=c.compare_family6(values)
        assert result['1']['Alameda'] is True and result['1']['ContraCosta'] is False
        assert result['2']=={'Alameda':True,'ContraCosta':True}


def test_private_completed_strong4_uses_past_deadline_existing_tree_and_never_fresh_admit(tmp_path,monkeypatch):
    # Reuse the audited toy checkpoint-pair metadata seam, while the actual
    # inventory, exact rows/tree, preservation receipt and Git predicates run.
    require_paths(COST/'scripts/remote/windows/ndr_formal_development_queue.py', COST/'tests/ndr_research/test_mage_train_only_cost_admission.py', B/'control/verify_only_cost_strong4_adapter_v5_20261016.py')
    native=q.load(file_record(COST/'scripts/remote/windows/ndr_formal_development_queue.py'),q.NATIVE_CORE_SHA)
    path=COST/'tests/ndr_research/test_mage_train_only_cost_admission.py'
    spec=importlib.util.spec_from_file_location('_completed_strong4_original_test_fixture',path)
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    fixture=original.completed_strong4.__wrapped__(tmp_path,monkeypatch)
    gate=original.gate
    assert q.sha(Path(gate.__file__))==q.PREDECESSOR_SHA
    adapter=q.load(file_record(B/'control/verify_only_cost_strong4_adapter_v5_20261016.py'),q.ADAPTER_SHA)
    def trap(*_args,**_kwargs):raise AssertionError('fresh admission invoked for completed strong4')
    monkeypatch.setattr(adapter,'admit_stage',trap);monkeypatch.setattr(adapter,'admit_strong4',trap)
    predecessor,completed=q.completed_predecessor_functions(gate)
    result=completed(fixture.row,fixture.cost_row,fixture.binding,fixture.modules,{},gate.Budget(30))
    assert fixture.root.exists() and result['admission']['pairs']==4 and result['admission']['native_owned_Popen_rows']==8
    assert ('load_frozen',COST) in fixture.calls
    assert all(row['original_held_wait_proof'] is None for row in result['processes'])
