"""Batched collection regression and temporary end-to-end dataset fixtures."""
from dataclasses import asdict, replace
import hashlib
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from optcar.dataset import DataConfig, FKDDataset
from optcar.datagen.build_dataset import build_dataset, inspect_sources
from optcar.datagen.dbm_to_dataset import (CommandConfig, DynamicBicycleEnv, generate,
    make_controller, randomized_params, simulation_params)
from optcar.datagen.raw_bins import describe_bin
from optcar.datagen.collection.randomization import ActionBiasSampler
from optcar.models.dynamic_bicycle import DBMParams, DynamicBicycleModel


@pytest.mark.parametrize('method', ['bezier', 'random_walk', 'fourier', 'polynomial'])
@pytest.mark.parametrize('batch', [1,3,32])
def test_batched_command_generators(oracle, method, batch):
    commands = CommandConfig(duration=.2, curve_methods=[[method],[method]], noise_std=[.2,.03], offset_fraction=.1)
    controller = make_controller(commands, .02, 10, 2, 'cpu')
    values = torch.stack([controller.compute_control(SimpleNamespace(n_envs=batch),dt=.02) for _ in range(10)])
    torch.testing.assert_close(values, torch.from_numpy(oracle[f'controller.{method}.batch{batch}']), atol=0, rtol=0)


def test_parameter_randomization_and_resampling(oracle):
    commands = CommandConfig(parameter_ranges={'len_f':[.23,.26], 'mass':[12,14]})
    base = simulation_params(DBMParams(), commands, .02)
    fields = ('len_f','mass','max_steering','max_velocity','C_drag','throttle_multiplier')
    for run in (0,1):
        values = randomized_params(base, commands, run)
        np.testing.assert_array_equal([getattr(values,key) for key in fields], oracle['randomizer.run0'])
    commands.resample_params_each_run = True
    values = randomized_params(base, commands, 1)
    np.testing.assert_array_equal([getattr(values,key) for key in fields], oracle['randomizer.run1'])


def test_seeded_yaw_and_fixed_control_seed(oracle):
    commands = CommandConfig(initial_yaw=[-1.5,1.5], fix_control_seed=True)
    params = simulation_params(DBMParams(), commands, .02)
    env = DynamicBicycleEnv(32, params=params, initial_yaw=commands.initial_yaw, seed=47)
    torch.testing.assert_close(env.state, torch.from_numpy(oracle['yaw_seed47.state']), atol=0, rtol=0)
    state = SimpleNamespace(n_envs=32)
    a = make_controller(commands,.02,6,0,'cpu').compute_control(state,dt=.02)
    b = make_controller(commands,.02,6,3,'cpu').compute_control(state,dt=.02)
    torch.testing.assert_close(a,b,atol=0,rtol=0)


def test_native_bezier_and_optional_stretch(oracle):
    commands = CommandConfig(duration=10, noise_std=[0,0], bezier_full_range=False)
    native = make_controller(commands,.02,500,0,'cpu')
    full = make_controller(replace(commands,bezier_full_range=True),.02,500,0,'cpu')
    state = SimpleNamespace(n_envs=32)
    native.compute_control(state,dt=.02); full.compute_control(state,dt=.02)
    torch.testing.assert_close(native._mean_sequences, torch.from_numpy(oracle["controller.bezier.native_full_range_false"]), atol=0, rtol=0)
    velocity = native._mean_sequences[:,:,0]
    assert torch.all(velocity.min(1).values > -2) and torch.all(velocity.max(1).values < 5)
    assert torch.all(full._mean_sequences[:,:,0].min(1).values == -2)
    assert torch.all(full._mean_sequences[:,:,0].max(1).values == 5)
    torch.testing.assert_close(full._mean_sequences[:,:,1], native._mean_sequences[:,:,1], atol=0, rtol=0)


def test_action_shaping_and_rollout(oracle):
    shaping = {'throttle_deadband':[-.1,.1],'steer_deadband':[-.02,.02],
        'throttle_multiplier':.85,'throttle_bias':.2,'steer_pos_multiplier':1.1,
        'steer_pos_bias':.01,'steer_neg_multiplier':.9,'steer_neg_bias':-.01}
    model = DynamicBicycleModel(action_shaping=shaping)
    commands = torch.from_numpy(oracle['action_shaping.input'])
    deadband = model._apply_action_deadbands(commands)
    torch.testing.assert_close(deadband, torch.from_numpy(oracle['action_shaping.deadband']), atol=0, rtol=0)
    torch.testing.assert_close(model._apply_action_adjustments(deadband), torch.from_numpy(oracle['action_shaping.adjusted']), atol=0, rtol=0)
    states = torch.zeros(3,13); states[:,3] = 1
    actual, wheel = model.rollout(states,commands,.02,torch.zeros(3))
    torch.testing.assert_close(actual, torch.from_numpy(oracle['action_shaping.rollout_state']), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(wheel, torch.from_numpy(oracle['action_shaping.rollout_wheel']), atol=1e-6, rtol=1e-6)


def test_per_trajectory_command_bias(oracle):
    sampler = ActionBiasSampler({'velocity_bias':[-.3,.4], 'steering_bias':[-.02,.03]},32,'cpu',44)
    actual = sampler._apply_action_bias(torch.tensor([[2.,.2]]).expand(32,-1))
    torch.testing.assert_close(actual, torch.from_numpy(oracle['action_bias.source_actions']), atol=0, rtol=0)
    assert sampler._velocity_bias.std() > 0


def test_randomized_start_pose_and_first_step(oracle):
    commands = CommandConfig(parameter_ranges={'start_pos_x':[1,2],'start_pos_y':[-3,-2],
        'start_pos_z':[.5,1.5],'start_rot_yaw':[.7,.9]})
    params = randomized_params(simulation_params(DBMParams(),commands,.02),commands,0)
    fields = ('start_pos_x','start_pos_y','start_pos_z','start_rot_yaw')
    np.testing.assert_array_equal([getattr(params,key) for key in fields], oracle['domain_pose.params'])
    env = DynamicBicycleEnv(3,params=params,initial_yaw=commands.initial_yaw)
    torch.testing.assert_close(env.state, torch.from_numpy(oracle['domain_pose.initial_state']), atol=0, rtol=0)
    torch.testing.assert_close(env.step(torch.tensor([[3.,.2]]).expand(3,-1)), torch.from_numpy(oracle['domain_pose.step_state']), atol=1e-6, rtol=1e-6)

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

@pytest.mark.parametrize('limit', [10., None])
def test_collection_to_processed_windows(tmp_path, oracle, limit):
    commands = CommandConfig(duration=.12, initial_yaw=[1.3,1.3], max_lateral_acceleration_mps2=limit)
    cfg = DataConfig(history_steps=2, future_steps=1, stride=1)
    out = tmp_path/'actual'
    result = generate(out,'road',data=cfg,commands=commands,runs=2,batch_size=3)
    assert (result['completed'],result['trajectories']) == (2,6)
    key = 'none' if limit is None else '10'
    for run in range(2):
        actual_file = out/f'data/cpu_{run:010d}.bin'
        actual = np.fromfile(actual_file,np.float32).reshape(3,6,15)
        expected = oracle[f'collection.{key}.run{run}']
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
    np.testing.assert_array_equal(saved[0,:,:41], oracle[f'collection.{key}.processed.run0'][0])
    assert sum(len(FKDDataset(tmp_path/'built',cfg,split)) for split,count in plan['windows'].items() if count)==24
    before={p:p.read_bytes() for p in (out/'data').glob('*.bin')}
    assert generate(out,'road',data=cfg,commands=commands,runs=2,batch_size=3)['completed']==2
    assert {p:p.read_bytes() for p in (out/'data').glob('*.bin')}==before
    with pytest.raises(ValueError):
        generate(out,'road',data=cfg,commands=commands,runs=2,batch_size=4)


def test_collection_resume_keeps_recorded_identity(tmp_path):
    root = tmp_path / 'collection'
    commands = CommandConfig(duration=.08)
    generate(root,'road',commands=commands,runs=1,batch_size=2)
    path = root / 'manifest.json'
    doc = json.loads(path.read_text())
    doc['provenance']['pipeline'] = 'recorded-collector-v1'
    identity = hashlib.sha256(json.dumps(doc['provenance'],sort_keys=True).encode()).hexdigest()[:12]
    entry = next(iter(doc['episodes'].values()))
    doc['episodes'] = {f'batch_{identity}_000000000': entry}
    path.write_text(json.dumps(doc))
    before = (root / entry['raw']).read_bytes()
    result = generate(root,'road',commands=commands,runs=2,batch_size=2)
    assert result['completed'] == 2
    assert (root / entry['raw']).read_bytes() == before
    saved = json.loads(path.read_text())
    assert saved['provenance'] == doc['provenance']
    assert all(key.startswith(f'batch_{identity}_') for key in saved['episodes'])
    assert json.loads((root/'metadata/metadata.json').read_text())['num_envs'] == 2
    with pytest.raises(ValueError,match='configuration changed'):
        generate(root,'road',commands=replace(commands,seed=55),runs=2,batch_size=2)
