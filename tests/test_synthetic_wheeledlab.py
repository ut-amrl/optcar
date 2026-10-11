"""Independent WheeledLab source parity; only tiny temporary collections are saved."""
import ast
from dataclasses import asdict, replace
import json
import sys
from types import ModuleType, SimpleNamespace
from typing import Dict, Tuple

import numpy as np
import pytest
import torch

from test_wheeledlab_parity import ROOT, load_functions, reference
from optcar.dataset import DataConfig, FKDDataset
from optcar.datagen.build_dataset import build_dataset, inspect_sources
from optcar.datagen.dbm_to_dataset import (CommandConfig, DynamicBicycleEnv, generate,
    make_controller, randomized_params, simulation_params)
from optcar.datagen.raw_bins import describe_bin
from optcar.datagen.wheeledlab.base_controller import BaseController
from optcar.models.dynamic_bicycle import DBMParams
from optcar.utils.math import quat_from_yaw_torch, yaw_from_quat_torch

pytestmark = pytest.mark.skipif(not ROOT.is_dir(), reason='Local WheeledLab source is unavailable')


def source_module(relative, name, namespace=None, skip_imports=False):
    path = ROOT / relative
    tree = ast.parse(path.read_text())
    # The controller's only package dependency is its abstract base interface.
    if skip_imports:
        tree.body = [node for node in tree.body if not isinstance(node, ast.Try)]
    module = ModuleType(name)
    module.__dict__.update(namespace or {})
    sys.modules[name] = module
    exec(compile(tree, str(path), 'exec'), module.__dict__)
    return module


@pytest.fixture
def source_controller():
    return source_module('wheeledlab_controllers/wheeledlab_controllers/parametric_open_loop.py',
                         'wheeledlab_controller_reference', {'BaseController': BaseController}, True).ParametricOpenLoopController


@pytest.mark.parametrize('method', ['bezier', 'random_walk', 'fourier', 'polynomial'])
@pytest.mark.parametrize('batch', [1, 3, 32])
def test_all_batched_command_generators_match(source_controller, method, batch):
    commands = CommandConfig(duration=.2, curve_methods=[[method], [method]],
                             noise_std=[.2, .03], offset_fraction=.1)
    actual = make_controller(commands, .02, 10, 2, 'cpu')
    expected = source_controller(action_dim=2, device='cpu', horizon_steps=10, default_dt=.02,
        action_bounds=commands.action_bounds, curve_methods=commands.curve_methods,
        noise_std=commands.noise_std, clip_actions=True, offset_fraction=.1, seed=44)
    state = SimpleNamespace(n_envs=batch)
    for _ in range(10):
        torch.testing.assert_close(actual.compute_control(state, dt=.02), expected.compute_control(state, dt=.02), atol=0, rtol=0)


def test_run_parameters_use_reference_randomizer_and_resampling_flag():
    module = source_module('wheeledlab_op/wheeledlab_op/env_setup/param_randomizer.py', 'wheeledlab_randomizer_reference')
    commands = CommandConfig(parameter_ranges={'len_f': [.23, .26], 'mass': [12, 14]})
    base = simulation_params(DBMParams(), commands, .02)
    expected = module.ParamRandomizer(seed=42).randomize_params(replace(base, **commands.parameter_ranges))
    assert asdict(randomized_params(base, commands, 0)) == asdict(expected)
    assert asdict(randomized_params(base, commands, 1)) == asdict(expected)
    commands.resample_params_each_run = True
    expected = module.ParamRandomizer(seed=43).randomize_params(replace(base, **commands.parameter_ranges))
    assert asdict(randomized_params(base, commands, 1)) == asdict(expected)
    assert expected.len_f != randomized_params(base, commands, 0).len_f


def reference_env(reference, params, batch, yaw):
    ns = {**reference, 'Dict': Dict, 'Tuple': Tuple, 'quat_from_yaw_torch': quat_from_yaw_torch}
    load_functions(ROOT/'wheeledlab_envs/wheeledlab_envs/analytical/base_bicycle_env.py', ['reset_state', 'step', '_reset_envs'], ns)
    env = SimpleNamespace(num_envs=batch, device='cpu', dt=.02, max_velocity=params.max_velocity,
        max_steering=params.max_steering, env_spacing=2.5, start_pos_x=0, start_pos_y=0, start_pos_z=0,
        start_rot_yaw=0, start_pos_x_range=None, start_pos_y_range=None, initial_yaw_range=[yaw,yaw],
        visualizer=None, state=torch.zeros(batch,13), vel_body=torch.zeros(batch,3), wheel_speed=torch.zeros(batch),
        env_terminated=torch.zeros(batch,dtype=torch.bool), auto_reset_episodes=True, sync_reset=True,
        bounded=False, min_wheel_velocity=params.min_wheel_velocity, max_wheel_velocity=params.max_wheel_velocity,
        episode_length=0, last_reset_ids=None)
    from optcar.datagen.dbm_to_dataset import model_params
    tensors = {key:torch.tensor(value) for key,value in asdict(model_params(params)).items()}
    def rollout(current_pos, current_quat, actions, dt, current_vel_body, current_wheel_speed):
        states, wheel = reference['_batch_rollout_dynamic_bicycle_jit_ref'](current_pos,current_quat,current_vel_body,
            current_wheel_speed,actions,dt,**tensors,max_lateral_acceleration_mps2=params.max_lateral_acceleration_mps2 or 0.)
        env.model.wheel = wheel
        return states
    class ReferenceModel:
        __call__ = staticmethod(rollout)
        def get_last_wheel_speed(self):
            return self.wheel
    env.model = ReferenceModel()
    env._check_obstacle_collisions = lambda:torch.zeros(batch,dtype=torch.bool)
    env._get_obs = env.get_obs = lambda:env.state.clone()
    env._reset_envs = lambda ids:ns['_reset_envs'](env,ids)
    ns['reset_state'](env)
    env.step = lambda action:ns['step'](env, action)
    return env


@pytest.mark.parametrize('limit', [10., None])
def test_full_collection_matches_source_steps_and_collector(tmp_path, reference, source_controller, limit):
    commands = CommandConfig(duration=.12, initial_yaw=[1.3,1.3], max_lateral_acceleration_mps2=limit)
    cfg = DataConfig(history_steps=2, future_steps=1, stride=1)
    out = tmp_path/'actual'
    result = generate(out,'road',data=cfg,commands=commands,runs=2,batch_size=3)
    assert (result['completed'],result['trajectories']) == (2,6)
    collector_module = source_module('wheeledlab_data/wheeledlab_data/datacollection/session.py','wheeledlab_collector_reference')
    context = collector_module.DataCollectionContext('reference','sample','cpu',3,.02,None,None,None,None,
        num_runs=2,max_steps_per_run=6,sim_dt=.02,decimation=1)
    collector = collector_module.DataCollector(tmp_path/'expected',context)
    params = simulation_params(DBMParams(),commands,.02)
    source_state = source_module('wheeledlab_op/wheeledlab_op/utils/state_struct.py','wheeledlab_state_reference').State_struct_torch
    for run in range(2):
        env = reference_env(reference,params,3,1.3)
        control = source_controller(action_dim=2,device='cpu',horizon_steps=6,default_dt=.02,
            action_bounds=commands.action_bounds,curve_methods=commands.curve_methods,noise_std=commands.noise_std,
            clip_actions=True,offset_fraction=0,seed=42+run)
        collector.start_run(run)
        for _ in range(6):
            state = source_state(pos=env.state[:,:3],quat=env.state[:,3:7],vel=env.state[:,7:10],
                                 ang_vel=env.state[:,10:13],n_envs=3)
            actions = control.compute_control(state,dt=.02)
            normalized = actions.clone()
            normalized[:,0] = torch.clamp(actions[:,0]/params.max_velocity,-1,1)
            normalized[:,1] = torch.clamp(actions[:,1]/params.max_steering,-1,1)
            env.step(normalized)
            state = source_state(pos=env.state[:,:3],quat=env.state[:,3:7],vel=env.state[:,7:10],
                                 ang_vel=env.state[:,10:13],n_envs=3)
            collector.record_step(state.get_vector_state(['pos','quat','vel','ang_vel']),actions)
        expected_file = collector.finish_run()
        actual_file = out/f'data/cpu_{run:010d}.bin'
        actual = np.fromfile(actual_file,np.float32).reshape(3,6,15)
        expected = np.fromfile(expected_file,np.float32).reshape(3,6,15)
        np.testing.assert_allclose(actual,expected,atol=1e-6,rtol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(actual[...,3:7],axis=-1),1,atol=1e-6)
        assert actual_file.stat().st_size == 3*6*15*4
        assert describe_bin(actual_file)['num_envs'] == 3
    metadata=json.loads((out/'metadata/metadata.json').read_text())
    assert metadata['num_envs']==3 and metadata['num_runs']==2 and metadata['max_steps_per_run']==6
    # Builder consumes every trajectory from each packed file, preserving commands.
    spec={'path':str(out),'source':'sim','terrain':'road','split':'auto'}
    plan=inspect_sources([spec],cfg)
    assert plan['files']==2 and not plan['errors']
    assert sum(plan['windows'].values())==24
    built=build_dataset(tmp_path/'built',[spec],cfg,device='cpu')
    assert built['new_windows']==24
    built_manifest=json.loads((tmp_path/'built/processed/sim/road/manifest.json').read_text())
    first=next(iter(built_manifest['episodes'].values()))
    shard=next(s for s in built_manifest['processed_shards'] if s['split']==first['split'])
    saved=np.fromfile(tmp_path/'built/processed/sim/road'/shard['path'],np.float32).reshape(-1,3,42)
    rows=np.fromfile(out/'data/cpu_0000000000.bin',np.float32).reshape(3,6,15)
    processor=SimpleNamespace(history_horizon=2,total_horizon=3,stride=1)
    chunks=reference['generate_chunks_torch'](processor,torch.tensor(rows))
    expected=reference['_process_chunks_to_frames_torch'](processor,chunks)
    np.testing.assert_array_equal(saved[0,:,:41],expected[0].numpy())
    assert sum(len(FKDDataset(tmp_path/'built',cfg,split)) for split,count in plan['windows'].items() if count)==24
    before={p:p.read_bytes() for p in (out/'data').glob('*.bin')}
    assert generate(out,'road',data=cfg,commands=commands,runs=2,batch_size=3)['completed']==2
    assert {p:p.read_bytes() for p in (out/'data').glob('*.bin')}==before
    with pytest.raises(ValueError):
        generate(out,'road',data=cfg,commands=commands,runs=2,batch_size=4)


def test_cancelled_collection_saves_only_complete_runs(tmp_path):
    calls=[]
    result=generate(tmp_path/'cancelled','road',data=DataConfig(),commands=CommandConfig(duration=.08),
        runs=3,batch_size=2,progress=lambda info:calls.append(info),cancelled=lambda:len(calls)>=5)
    assert result['completed']==1 and len(list((tmp_path/'cancelled/data').glob('*.bin')))==1
    assert len(calls)==5


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA is unavailable')
def test_indexed_gpu_step_matches_cpu():
    cpu=DynamicBicycleEnv(32,device='cpu',initial_yaw=[.7,.7])
    gpu=DynamicBicycleEnv(32,device='cuda:0',initial_yaw=[.7,.7])
    action=torch.tensor([[3.,.45]]).expand(32,-1)
    for _ in range(4):
        torch.testing.assert_close(gpu.step(action).cpu(),cpu.step(action),atol=2e-5,rtol=2e-5)
    assert str(gpu.state.device)=='cuda:0'


def test_random_initial_yaw_and_fixed_control_seed_match_source(reference, source_controller):
    commands=CommandConfig(initial_yaw=[-1.5,1.5],fix_control_seed=True)
    params=simulation_params(DBMParams(),commands,.02)
    actual=DynamicBicycleEnv(32,params=params,initial_yaw=commands.initial_yaw,seed=47)
    expected=reference_env(reference,params,32,0)
    # WheeledLab's env.reset(seed=...) seeds Torch before sampling initial yaw.
    ns={**reference,'quat_from_yaw_torch':quat_from_yaw_torch}
    load_functions(ROOT/'wheeledlab_envs/wheeledlab_envs/analytical/base_bicycle_env.py',['reset_state'],ns)
    expected.initial_yaw_range=commands.initial_yaw
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(47)
        ns['reset_state'](expected)
    torch.testing.assert_close(actual.state,expected.state,atol=0,rtol=0)
    state=SimpleNamespace(n_envs=32)
    a=make_controller(commands,.02,6,0,'cpu').compute_control(state,dt=.02)
    b=make_controller(commands,.02,6,3,'cpu').compute_control(state,dt=.02)
    torch.testing.assert_close(a,b,atol=0,rtol=0)


def test_native_bezier_uses_reference_control_points_without_stretch(source_controller, monkeypatch):
    module=sys.modules[source_controller.__module__]
    original=module.ParametricSignalGenerator._scale_to_bounds_batch
    def native(self, signals):
        return signals if self.methods==['bezier'] else original(self,signals)
    monkeypatch.setattr(module.ParametricSignalGenerator,'_scale_to_bounds_batch',native)
    commands=CommandConfig(duration=10,noise_std=[0,0],bezier_full_range=False)
    actual=make_controller(commands,.02,500,0,'cpu')
    expected=source_controller(action_dim=2,device='cpu',horizon_steps=500,default_dt=.02,
        action_bounds=commands.action_bounds,curve_methods=commands.curve_methods,noise_std=[0,0],offset_fraction=0,seed=42)
    state=SimpleNamespace(n_envs=32)
    actual.compute_control(state,dt=.02);expected.compute_control(state,dt=.02)
    torch.testing.assert_close(actual._mean_sequences,expected._mean_sequences,atol=0,rtol=0)
    velocity=actual._mean_sequences[:,:,0]
    assert torch.all(velocity.min(1).values>-2) and torch.all(velocity.max(1).values<5)
    full=make_controller(replace(commands,bezier_full_range=True),.02,500,0,'cpu')
    full.compute_control(state,dt=.02)
    assert torch.all(full._mean_sequences[:,:,0].min(1).values==-2)
    assert torch.all(full._mean_sequences[:,:,0].max(1).values==5)
    torch.testing.assert_close(full._mean_sequences[:,:,1],actual._mean_sequences[:,:,1],atol=0,rtol=0)


def test_domain_ranges_and_action_shaping_match_source(reference):
    from optcar.models.dynamic_bicycle import DynamicBicycleModel
    source=source_module('wheeledlab_models/wheeledlab_models/analytical/bicycle/base_bicycle_model.py','wheeledlab_base_model_reference')
    class SourceModel(source.BaseBicycleModel):
        def rollout(self,*args,**kwargs):
            pass
    shaping={'throttle_deadband':[-.1,.1],'steer_deadband':[-.02,.02],
        'throttle_multiplier':.85,'throttle_bias':.2,'steer_pos_multiplier':1.1,
        'steer_pos_bias':.01,'steer_neg_multiplier':.9,'steer_neg_bias':-.01}
    model=DynamicBicycleModel(action_shaping=shaping)
    expected_model=SourceModel(device='cpu',**shaping)
    q=torch.tensor([[1.,0,0,0]]).expand(3,-1)
    states=torch.zeros(3,13);states[:,3:7]=q
    commands=torch.tensor([[[.05,.01],[3,.4]],[[1.,-.3],[-2.,-.2]],[[2.,.2],[3.,-.4]]])
    adjusted=expected_model._apply_action_adjustments(expected_model._apply_action_deadbands(commands))
    expected,wheel=reference['_batch_rollout_dynamic_bicycle_jit_ref'](states[:,:3],q,torch.zeros(3,3),torch.zeros(3),adjusted,.02,**model.tensors)
    actual,actual_wheel=model.rollout(states,commands,.02,torch.zeros(3))
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(actual_wheel,wheel,atol=1e-6,rtol=1e-6)
    commands_cfg=CommandConfig(parameter_ranges={'mass':[11,15],'max_steering_angle':[.4,.6],
        'throttle_multiplier':[.75,1.25],'steer_neg_bias':[-.03,.03], 'C_drag':[3,.2,'normal']})
    base=simulation_params(DBMParams(),commands_cfg,.02)
    source_randomizer=source_module('wheeledlab_op/wheeledlab_op/env_setup/param_randomizer.py','wheeledlab_domain_randomizer_reference')
    expected=source_randomizer.ParamRandomizer(seed=42).randomize_params(replace(base,**commands_cfg.parameter_ranges))
    assert asdict(randomized_params(base,commands_cfg,0))==asdict(expected)
    env=DynamicBicycleEnv(2,params=expected)
    assert env.model.throttle_multiplier==expected.throttle_multiplier
    assert env.model.steer_neg_bias==expected.steer_neg_bias


def test_per_trajectory_command_bias_matches_sim_runner():
    from optcar.datagen.wheeledlab.action_bias import ActionBiasSampler
    ranges={'velocity_bias':[-.3,.4],'steering_bias':[-.02,.03]}
    actual=ActionBiasSampler(ranges,32,'cpu',44)
    ns={'np':np,'torch':torch}
    load_functions(ROOT/'wheeledlab_op/wheeledlab_op/runners/sim_runner.py',['_apply_action_bias','_resample_action_bias'],ns)
    expected=SimpleNamespace(cfg=SimpleNamespace(env_setup=SimpleNamespace(num_envs=32,params=ranges)),tensor_device='cpu')
    # Compare to the source global RNG without changing other tests' random state.
    previous=np.random.get_state()
    try:
        np.random.seed(44);ns['_resample_action_bias'](expected)
    finally:
        np.random.set_state(previous)
    actions=torch.tensor([[2.,.2]]).expand(32,-1)
    torch.testing.assert_close(actual._apply_action_bias(actions),ns['_apply_action_bias'](expected,actions),atol=0,rtol=0)
    assert actual._velocity_bias.std()>0


@pytest.mark.parametrize('ranges',[
    {'mass':[15,11]}, {'mass':[13,-1,'normal']}, {'mass':[13,1,'unknown']},
    {'mass':[1,float('nan')]}, {'dt':[.01,.03]}, {'velocity_bias':[0,1,'normal']},
    {'max_steering':[.4,.6],'max_steering_angle':[.4,.6]},
])
def test_invalid_ranges_fail_before_output(tmp_path,ranges):
    with pytest.raises(ValueError):
        generate(tmp_path/'invalid','road',commands=CommandConfig(parameter_ranges=ranges))
    assert not (tmp_path/'invalid').exists()


def test_gui_generation_parameter_ranges(tmp_path,monkeypatch):
    from optcar.gui import server
    from optcar.utils.file_io import write_yaml,terrain_path
    client=server.create_app(tmp_path/'datasets').test_client()
    ranges={'mass':[11.,15.,'uniform'], 'len_f':[.2,.28,'uniform'],
            'throttle_multiplier':[.75,1.25,'uniform'], 'steering_bias':[-.025,.025,'uniform']}
    defaults=client.get('/api/defaults').json
    assert 'steer_pos_multiplier' in {item['key'] for item in defaults['generation_parameters']}
    write_yaml(terrain_path(tmp_path/'datasets','road')/'dbm_params.yaml',{'params':asdict(DBMParams())})
    called=[]
    def fake_start(self, terrain, run, **kwargs):
        called.append(CommandConfig(**kwargs['commands']))
        return {'job_id':'a'*32,'tmux_session':'optcar-test','attach_command':'tmux attach -t optcar-test'}
    monkeypatch.setattr(server.GenerationJobs,'start',fake_start)
    response=client.post('/api/generate',json={'terrain':'road','commands':{'parameter_ranges':ranges,
        'resample_params_each_run':True,'bezier_full_range':False}})
    assert response.status_code==200 and response.json['tmux_session']=='optcar-test'
    assert called[0].parameter_ranges==ranges and called[0].bezier_full_range is False


def test_saved_collection_records_resolved_parameters_and_command_bias(tmp_path):
    commands=CommandConfig(duration=.08,parameter_ranges={'mass':[11,15],
        'throttle_multiplier':[.8,1.2],'velocity_bias':[-.2,.2],'steering_bias':[-.02,.02]},
        resample_params_each_run=True,bezier_full_range=False)
    result=generate(tmp_path/'collection','road',commands=commands,runs=2,batch_size=3)
    assert result['completed']==2
    manifest=json.loads((tmp_path/'collection/manifest.json').read_text())
    entries=list(manifest['episodes'].values())
    assert entries[0]['metadata']['params']['mass']!=entries[1]['metadata']['params']['mass']
    for index,entry in enumerate(entries):
        saved=np.fromfile(tmp_path/'collection'/entry['raw'],np.float32).reshape(3,4,15)
        control=make_controller(commands,.02,4,index,'cpu')
        controls=torch.stack([control.compute_control(SimpleNamespace(n_envs=3),dt=.02) for _ in range(4)],1).numpy()
        bias=np.array(entry['metadata']['action_bias'],np.float32)
        np.testing.assert_array_equal(saved[...,13:15],controls+bias[:,None,:])
        assert np.isfinite(saved).all()


def test_domain_starting_pose_matches_analytical_reference(reference):
    commands=CommandConfig(parameter_ranges={'start_pos_x':[1,2],'start_pos_y':[-3,-2],
        'start_pos_z':[.5,1.5],'start_rot_yaw':[.7,.9]})
    params=randomized_params(simulation_params(DBMParams(),commands,.02),commands,0)
    actual=DynamicBicycleEnv(3,params=params,initial_yaw=commands.initial_yaw)
    expected=reference_env(reference,params,3,params.start_rot_yaw)
    expected.start_pos_x=params.start_pos_x;expected.start_pos_y=params.start_pos_y;expected.start_pos_z=params.start_pos_z
    ns={**reference,'quat_from_yaw_torch':quat_from_yaw_torch}
    load_functions(ROOT/'wheeledlab_envs/wheeledlab_envs/analytical/base_bicycle_env.py',['reset_state'],ns)
    ns['reset_state'](expected)
    torch.testing.assert_close(actual.state,expected.state,atol=0,rtol=0)
    actions=torch.tensor([[3.,.2]]).expand(3,-1)
    normalized=actions/torch.tensor([params.max_velocity,params.max_steering])
    expected.step(normalized)
    torch.testing.assert_close(actual.step(actions),expected.state,atol=1e-6,rtol=1e-6)
