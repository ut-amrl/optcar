"""Fixture-only checks for assembling the existing raw and FKD formats."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from optcar.dataset import DataConfig, DatasetWriter, FKDDataset, integrate_deltas, model_sample, pack_windows
from optcar.datagen.build_dataset import build_dataset, inspect_sources
from optcar.datagen.raw_bins import describe_bin
from optcar.gui.server import create_app


def raw_fixture(folder, width=15, dt=.02, environments=1, name='run', source='/bags/run'):
    folder.mkdir(parents=True, exist_ok=True)
    rows=np.zeros((environments, 12, width), dtype=np.float32)
    rows[...,0]=np.arange(12)*.1
    rows[...,3]=1
    rows[...,7]=5
    rows[...,13]=np.arange(12)*.2
    rows[...,14]=.3
    if width==17:
        rows[...,15:]=99  # Diagnostic commands must not become training actions.
    for env in range(environments):
        rows[env,:,1]=env*10
    file=folder/f'{name}.bin';rows.tofile(file)
    meta=folder/'metadata';meta.mkdir(exist_ok=True)
    (meta/'schema.json').write_text(json.dumps({'features':width}))
    (meta/'dataset_config.py').write_text(f'class DatasetConfig:\n    control_dt = {dt}\n    num_envs = {environments}\n')
    (meta/'conversion.json').write_text(json.dumps({'bins':{file.name:{'source':source}}}))
    return file,rows


def spec(path, **kwargs):
    return {'path':str(path),'terrain':'road','source':'real','split':'train',**kwargs}


@pytest.mark.parametrize('width', [15,17])
def test_build_preserves_actions_all_envs_and_restart(tmp_path,width):
    file,rows=raw_fixture(tmp_path/'inputs',width,environments=2)
    original=file.read_bytes();cfg=DataConfig(history_steps=4,future_steps=3,stride=2)
    source=spec(file.parent)
    plan=inspect_sources([source],cfg)
    assert plan['files']==1 and plan['windows']['train']==6
    result=build_dataset(tmp_path/'assembled',[source],cfg)
    assert result['new_windows']==6
    assert file.read_bytes()==original
    manifest=json.loads((tmp_path/'assembled/road/real/manifest.json').read_text())
    episodes=list(manifest['episodes'].values())
    assert len(episodes)==1
    assert episodes[0]['raw']==str(file)
    assert not (tmp_path/'assembled/road/real/raw').exists()
    assert episodes[0]['shards']==[] and episodes[0]['stored_windows']==6
    first=np.fromfile(tmp_path/'assembled/road/real'/manifest['processed_shards'][0]['path'],np.float32).reshape(-1,7,42)
    np.testing.assert_allclose(first[0,:,39:41],rows[0,:7,13:15])
    np.testing.assert_allclose(first[0,3,13:20],[0,0,0,1,0,0,0],atol=1e-6)
    assert len(FKDDataset(tmp_path/'assembled',cfg))==6
    assert build_dataset(tmp_path/'assembled',[source],cfg)['new_windows']==0
    with pytest.raises(ValueError,match='changed'):
        build_dataset(tmp_path/'assembled',[source],DataConfig(history_steps=3,future_steps=3,stride=2))


def test_matching_timestep_selection_duplicates_and_leakage(tmp_path):
    folder=tmp_path/'inputs';file,_=raw_fixture(folder,dt=.04)
    cfg=DataConfig(history_steps=3,future_steps=2,dt=.02)
    assert 'Timestep' in inspect_sources([spec(folder)],cfg)['errors'][0]
    cfg.dt=.04
    assert inspect_sources([spec(folder,selected_paths=[])],cfg)['files']==0
    assert 'already selected' in inspect_sources([spec(folder),spec(file)],cfg)['errors'][0]
    other=tmp_path/'other';raw_fixture(other,dt=.04) # Same source bag identity.
    assert 'different splits' in inspect_sources([spec(folder),spec(other,split='test')],cfg)['errors'][0]
    with pytest.raises(ValueError,match='summing'):
        inspect_sources([spec(folder)],cfg,fractions=[np.nan,0,0])


def test_parent_of_three_raw_terrains_creates_processed_folder(tmp_path):
    parent = tmp_path / 'corl26' / 'sim'
    sources, originals = [], {}
    for terrain in ('grass', 'road', 'mulch'):
        file, _ = raw_fixture(parent / terrain, environments=2, source=f'/bags/{terrain}')
        sources.append(spec(file.parent, terrain=terrain, source='sim'))
        originals[file] = file.read_bytes()
    cfg = DataConfig(history_steps=3, future_steps=2, stride=1)
    result = build_dataset(parent, sources, cfg)
    assert result['dataset'] == str(parent / 'processed')
    assert result['completed'] == 3 and result['new_windows'] == 48
    for terrain in ('grass', 'road', 'mulch'):
        assert (parent / 'processed' / terrain / 'sim' / 'manifest.json').is_file()
    assert all(file.read_bytes() == value for file, value in originals.items())
    assert build_dataset(parent, sources, cfg)['new_windows'] == 0
    assert len(FKDDataset(parent / 'processed', cfg)) == 48


@pytest.mark.parametrize('location', ['same', 'nested', 'existing'])
def test_shared_and_nonempty_folders_keep_inputs_and_resume(tmp_path, location):
    file, _ = raw_fixture(tmp_path / 'inputs')
    cfg = DataConfig(history_steps=3, future_steps=2, stride=1)
    output = file.parent if location == 'same' else file.parent / 'built' if location == 'nested' else tmp_path / 'existing'
    output.mkdir(parents=True, exist_ok=True)
    note = output / 'notes.txt'
    note.write_text('keep this file')
    original = file.read_bytes()
    result = build_dataset(output, [spec(file.parent)], cfg)
    assert result['new_windows'] == 8
    assert file.read_bytes() == original and note.read_text() == 'keep this file'
    # Scanning the original folder again must exclude this build's raw copies
    # and processed shards, including when they live within the input tree.
    plan = inspect_sources([spec(file.parent)], cfg)
    assert plan['files'] == 1 and not plan['errors']
    assert build_dataset(output, [spec(file.parent)], cfg)['new_windows'] == 0


def test_existing_output_file_is_preserved(tmp_path):
    file, _ = raw_fixture(tmp_path / 'inputs')
    output = tmp_path / 'output'
    conflict = output / 'road' / 'real' / 'metadata' / 'schema.json'
    conflict.parent.mkdir(parents=True)
    conflict.write_text('existing user metadata')
    with pytest.raises(ValueError, match='existing file'):
        build_dataset(output, [spec(file)], DataConfig(history_steps=3, future_steps=2))
    assert conflict.read_text() == 'existing user metadata'
    assert not (output / 'dataset_build.json').exists()


def test_raw_only_metadata_saved_split_and_wheeledlab_nan_policy(tmp_path):
    cfg=DataConfig(history_steps=3,future_steps=2,dt=.04)
    _,rows=raw_fixture(tmp_path/'fixtures',dt=.04)
    raw=tmp_path/'synthetic'
    with DatasetWriter(raw,cfg,'sim','road',process_windows=False) as writer:
        assert writer.write_episode('episode',rows[0,:,:13],rows[0,:,13:15],'val')==0
    assert not (raw/'processed').exists()
    assert describe_bin(raw/'raw/episode.bin')['dt']==.04
    plan=inspect_sources([spec(raw,source='sim',split='auto')],cfg)
    assert plan['files']==1 and plan['windows']['val']==1
    assert inspect_sources([spec(raw,source='sim',split='train')],cfg)['errors']
    assert inspect_sources([spec(raw,source='sim',split='auto',dt=.02)],cfg)['errors']
    result=build_dataset(tmp_path/'assembled',[spec(raw,source='sim',split='auto')],cfg)
    assert result['new_windows']==1 and len(FKDDataset(tmp_path/'assembled',cfg,'val'))==1
    file,rows=raw_fixture(tmp_path/'bad');rows[0,4,7]=np.nan;rows.tofile(file)
    result=build_dataset(tmp_path/'skipped',[spec(file)],DataConfig(history_steps=3,future_steps=2))
    assert result['new_windows']==0 and len(result['skipped'])==1


def test_builder_inspection_api_and_generation_mode(tmp_path,monkeypatch):
    from optcar.gui import server
    from optcar.utils.file_io import terrain_path,write_yaml
    from dataclasses import asdict
    from optcar.models.dynamic_bicycle import DBMParams
    file,_=raw_fixture(tmp_path/'inputs')
    client=create_app(tmp_path/'datasets').test_client()
    response=client.post('/api/dataset-builder/inspect',json={'sources':[spec(file)],'data':{'history_steps':3,'future_steps':2}})
    assert response.status_code==200 and response.json['files']==1
    assert client.post('/api/dataset-builder/inspect',json={'sources':[]}).status_code==400
    called=[]
    def fake_start(self,terrain,run,**kwargs):
        called.append((terrain,run))
        return {'job_id':'a'*32,'tmux_session':'optcar-test'}
    monkeypatch.setattr(server.GenerationJobs,'start',fake_start)
    write_yaml(terrain_path(tmp_path/'datasets','road')/'dbm_params.yaml',{'params':asdict(DBMParams())})
    response=client.post('/api/generate',json={'terrain':'road','raw_only':True})
    assert response.status_code==200 and response.json['tmux_session']=='optcar-test'
    assert called==[('road','default')]
    assert client.post('/api/generate',json={'terrain':'road','raw_only':False}).status_code==400
    assert len(called)==1


def test_full_rotation_and_delta_reconstruction():
    from scipy.spatial.transform import Rotation
    rotations=Rotation.from_euler('xyz',[[.3,-.4,1.2],[.4,-.1,1.4],[.5,.2,1.8],[.2,.3,2.1],[.3,.4,2.4]])
    states=np.zeros((1,5,13),np.float32)
    states[0,:,:3]=[[2,3,4],[2.1,3.2,4.1],[2.3,3.4,4.2],[2.4,3.3,4.4],[2.5,3.2,4.3]]
    states[0,:,3:7]=rotations.as_quat()[:,[3,0,1,2]]
    states[0,:,7:10]=[[1,2,3]]*5
    states[0,:,10:13]=[[.2,.3,.4]]*5
    states[0,1::2,3:7]*=-1 # Same orientations, alternating recorded signs.
    packed=pack_windows(states,np.zeros((1,5,2),np.float32),3)
    expected=rotations[2].inv().apply(states[0,:,:3]-states[0,2,:3])
    np.testing.assert_allclose(packed[0,:,13:16],expected,atol=5e-7)
    np.testing.assert_allclose(packed[0,:,20:23],rotations[2].inv().apply(states[0,:,7:10]),atol=5e-7)
    np.testing.assert_allclose(packed[0,:,33:36],rotations.inv().apply(states[0,:,7:10]),atol=8e-7)
    rebuilt=integrate_deltas(packed[:,1:,26:33],packed[:,0,:7])
    np.testing.assert_allclose(rebuilt[0,:,:3],states[0,1:,:3],atol=1e-6)
    assert torch.all(torch.sum(rebuilt[0,:,3:7]*packed[0,1:,3:7],-1).abs()>.99999)
    sample=model_sample(packed[0],3)
    assert sample['commands'].shape==(2,2) and sample['target_delta'].shape==(2,7)


@pytest.mark.parametrize('shard_windows', [2, 20])
def test_cpu_gpu_build_parity_progress_and_resume(tmp_path, shard_windows):
    if not torch.cuda.is_available():
        pytest.skip('CUDA is unavailable')
    file, rows = raw_fixture(tmp_path/'inputs', environments=4)
    # Full rotations and sign changes exercise more than planar motion.
    from optcar.utils.math import euler_to_quat_torch
    rng=np.random.default_rng(6)
    q=euler_to_quat_torch(torch.tensor(rng.normal(size=(4*12,3)),dtype=torch.float32)).reshape(4,12,4).numpy()
    q[:,1::2]*=-1; rows[...,3:7]=q;rows.tofile(file)
    cfg=DataConfig(history_steps=4,future_steps=3,stride=2,shard_windows=shard_windows)
    progress=[]
    cpu=build_dataset(tmp_path/'cpu',[spec(file)],cfg,device='cpu')
    gpu=build_dataset(tmp_path/'gpu',[spec(file)],cfg,device='cuda:0',progress=progress.append)
    assert cpu['new_windows']==gpu['new_windows']==12
    assert progress[0]['current']=='Inspecting selected bins'
    assert progress[-1]['completed']==progress[-1]['total']==12
    if shard_windows < 12:
        assert any(0<p['completed']<12 for p in progress)
    assert any('Converting' in p['current'] for p in progress)
    assert any('Writing' in p['current'] for p in progress)
    assert all(p['device']=='cuda:0' for p in progress)
    cpu_shards=sorted((tmp_path/'cpu').rglob('processed/data/**/*.bin'))
    gpu_shards=sorted((tmp_path/'gpu').rglob('processed/data/**/*.bin'))
    assert len(cpu_shards)==len(gpu_shards)
    for left,right in zip(cpu_shards,gpu_shards):
        np.testing.assert_allclose(np.fromfile(left,np.float32),np.fromfile(right,np.float32),atol=2e-6,rtol=2e-6)
    assert build_dataset(tmp_path/'gpu',[spec(file)],cfg,device='cpu')['new_windows']==0


def test_invalid_build_device_before_output(tmp_path):
    file,_=raw_fixture(tmp_path/'inputs')
    with pytest.raises(ValueError,match='unavailable'):
        build_dataset(tmp_path/'out',[spec(file)],device='cuda:9999')
    assert not (tmp_path/'out').exists()


def test_loading_workers_read_concurrently_keep_order_and_copy(monkeypatch):
    from threading import Barrier, Lock, current_thread
    from optcar.datagen import build_dataset as module
    barrier=Barrier(2);lock=Lock();threads=set()
    original=np.zeros((1,7,15),np.float32);original[...,3]=1
    def load(path,*args):
        with lock:threads.add(current_thread().name)
        barrier.wait(timeout=5)
        return original,15
    monkeypatch.setattr(module,'load_bin',load)
    entries=[{'path':str(index),'columns':15,'num_envs':1} for index in range(4)]
    loaded=list(module.load_files_parallel(entries,max_workers=2))
    assert [entry for entry,_ in loaded]==entries
    assert len(threads)==2
    assert all(not np.shares_memory(rows,original) for _,rows in loaded)
    loaded[0][1][...,0]=8
    assert not original[...,0].any()


def test_parallel_build_matches_single_worker_and_worker_count_can_change(tmp_path):
    for index in range(4):
        raw_fixture(tmp_path/'inputs',environments=2,name=f'run_{index}',source=f'/bags/run_{index}')
    cfg=DataConfig(history_steps=3,future_steps=2,stride=1,shard_windows=3)
    sources=[spec(tmp_path/'inputs')]
    one=build_dataset(tmp_path/'one',sources,cfg,device='cpu',workers=1)
    parallel=build_dataset(tmp_path/'parallel',sources,cfg,device='cpu',workers=4)
    assert one['new_windows']==parallel['new_windows']==64
    left=sorted((tmp_path/'one').rglob('*.bin'));right=sorted((tmp_path/'parallel').rglob('*.bin'))
    assert len(left)==len(right)
    for a,b in zip(left,right):
        assert a.relative_to(tmp_path/'one')==b.relative_to(tmp_path/'parallel')
        assert a.read_bytes()==b.read_bytes()
    assert build_dataset(tmp_path/'parallel',sources,cfg,device='cpu',workers=2)['new_windows']==0


@pytest.mark.parametrize('workers',[0,-1,2.5,True])
def test_invalid_workers_before_output(tmp_path,workers):
    with pytest.raises(ValueError,match='workers must be a positive integer'):
        build_dataset(tmp_path/'invalid',[],workers=workers)
    assert not (tmp_path/'invalid').exists()


def test_cancelled_loader_does_not_read(monkeypatch):
    from optcar.datagen import build_dataset as module
    monkeypatch.setattr(module,'load_bin',lambda *args:pytest.fail('Cancelled loader read a file'))
    assert list(module.load_files_parallel([{'path':'unused'}],max_workers=16,cancelled=lambda:True))==[]


def test_separate_synthetic_collections_can_repeat_saved_episode_ids(tmp_path):
    parent=tmp_path/'grass'
    cfg=DataConfig(history_steps=3,future_steps=2,stride=1)
    original={}
    # Reproduce grass_0 / grass_1, whose config-derived saved batch IDs repeat.
    # Contents can differ: an ID alone is not a content-duplicate check.
    for index in range(2):
        _,rows=raw_fixture(tmp_path/f'fixture_{index}')
        rows[...,0]+=index
        with DatasetWriter(parent/f'grass_{index}',cfg,'sim','grass',process_windows=False) as writer:
            writer.write_episode('batch_settings_000000000',rows[0,:,:13],rows[0,:,13:15],
                                 'val',session='batch_settings_000000000')
        path=parent/f'grass_{index}/raw/batch_settings_000000000.bin'
        original[path]=path.read_bytes()
    source=spec(parent,terrain='grass',source='sim',split='auto')
    plan=inspect_sources([source],cfg)
    assert plan['files']==2 and plan['windows']['val']==16 and not plan['errors']
    result=build_dataset(tmp_path/'built',[source],cfg,device='cpu',workers=2)
    assert result['completed']==2 and result['new_windows']==16
    manifest=json.loads((tmp_path/'built/grass/sim/manifest.json').read_text())
    assert len(manifest['episodes'])==2
    assert all(e['split']=='val' for e in manifest['episodes'].values())
    assert all(path.read_bytes()==contents for path,contents in original.items())
    assert build_dataset(tmp_path/'built',[source],cfg,device='cpu')['new_windows']==0
    # Selecting the same file through another source still fails.
    assert 'already selected' in inspect_sources([source,spec(next(iter(original)),terrain='grass',source='sim',split='auto')],cfg)['errors'][0]


def test_resume_legacy_per_environment_build_without_rewriting(tmp_path):
    from optcar.datagen.build_dataset import _episode_name
    file, rows = raw_fixture(tmp_path/'inputs', environments=4)
    cfg = DataConfig(history_steps=4, future_steps=3, stride=2, shard_windows=4)
    sources = [spec(file)]
    entries = inspect_sources(sources, cfg)['bins']
    output = tmp_path/'built'
    target = output/'road/real'
    # A stopped older builder has committed its first two environments.
    with DatasetWriter(target, cfg, 'real', 'road', {'inputs': entries, 'calibrations': []}) as writer:
        for env in range(2):
            writer.write_episode(_episode_name(entries[0], env), rows[env, :, :13], rows[env, :, 13:15],
                                 'train', session=entries[0]['session'])
    original = {p: p.read_bytes() for p in target.rglob('*.bin')}
    result = build_dataset(output, sources, cfg, device='cpu')
    assert result['new_windows'] == 6
    assert all(p.read_bytes() == contents for p, contents in original.items())
    assert len(FKDDataset(output, cfg)) == 12
    manifest = json.loads((target/'manifest.json').read_text())
    batched = manifest['episodes'][_episode_name(entries[0], 'batch')]
    assert batched['metadata']['environments'] == [2, 3]
    assert batched['shards']==[] and batched['stored_windows']==6
    assert [s['shape'][0] for s in manifest['processed_shards']] == [4, 2]
    assert build_dataset(output, sources, cfg, device='cpu')['new_windows'] == 0


def test_cancel_during_batch_write_removes_uncommitted_windows_and_resumes(tmp_path):
    file, _ = raw_fixture(tmp_path/'inputs', environments=4)
    cfg = DataConfig(history_steps=4, future_steps=3, stride=2, shard_windows=4)
    events = []
    result = build_dataset(tmp_path/'built', [spec(file)], cfg, device='cpu', progress=events.append,
                           cancelled=lambda: any(p['current'].startswith('Saved windows') for p in events))
    assert result['completed'] == result['new_windows'] == 0
    assert not list((tmp_path/'built').rglob('*.bin'))
    assert build_dataset(tmp_path/'built', [spec(file)], cfg, device='cpu')['new_windows'] == 12


def test_carryover_combines_input_files_and_loader_preserves_file_ids(tmp_path):
    sources, originals = [], {}
    for index in range(3):
        file, rows = raw_fixture(tmp_path/f'input_{index}', environments=2, source=f'/bags/{index}')
        rows[...,0] += 100*index
        rows.tofile(file)
        sources.append(spec(file))
        originals[file] = file.read_bytes()
    cfg=DataConfig(history_steps=4,future_steps=3,stride=2,shard_windows=10)
    result=build_dataset(tmp_path/'built',sources,cfg,device='cpu')
    target=tmp_path/'built/road/real'
    manifest=json.loads((target/'manifest.json').read_text())
    assert result['new_windows']==18
    assert [s['shape'] for s in manifest['processed_shards']]==[[10,7,42],[8,7,42]]
    assert [s['path'] for s in manifest['processed_shards']]==[
        'processed/data/train/processed_0000000000.bin',
        'processed/data/train/processed_0000000001.bin']
    assert len(manifest['episodes'])==3
    assert all(e['shards']==[] and e['windows']==e['stored_windows']==6 for e in manifest['episodes'].values())
    packed=np.concatenate([np.fromfile(target/s['path'],np.float32).reshape(s['shape']) for s in manifest['processed_shards']])
    np.testing.assert_array_equal(packed[:,0,41],np.repeat([0,1,2],6))
    for index in range(3):
        np.testing.assert_allclose(packed[index*6:(index+1)*6,0,0],100*index+np.tile([0,.2,.4],2),atol=1e-5)
        np.testing.assert_allclose(packed[index*6:(index+1)*6,0,39],np.tile([0,.4,.8],2))
    dataset=FKDDataset(tmp_path/'built',cfg)
    assert len(dataset)==18 and len(dataset.shards)==2
    for index in range(18):
        sample=dataset[index]
        torch.testing.assert_close(sample['commands'],torch.from_numpy(packed[index,3:-1,39:41]))
    assert all(file.read_bytes()==original for file,original in originals.items())
    assert build_dataset(tmp_path/'built',sources,cfg,device='cpu')['new_windows']==0


def test_interleaved_splits_have_independent_carryover(tmp_path):
    sources=[]
    for index, split in enumerate(['train','val','train','val']):
        file,rows=raw_fixture(tmp_path/f'input_{index}',environments=2,source=f'/bags/{index}')
        rows[...,0]+=index*100;rows.tofile(file)
        sources.append(spec(file,split=split))
    cfg=DataConfig(history_steps=4,future_steps=3,stride=2,shard_windows=10)
    build_dataset(tmp_path/'built',sources,cfg,device='cpu')
    target=tmp_path/'built/road/real'
    manifest=json.loads((target/'manifest.json').read_text())
    for split,ids in [('train',[0,2]),('val',[1,3])]:
        shards=[s for s in manifest['processed_shards'] if s['split']==split]
        assert [s['shape'][0] for s in shards]==[10,2]
        packed=np.concatenate([np.fromfile(target/s['path'],np.float32).reshape(s['shape']) for s in shards])
        np.testing.assert_array_equal(packed[:,0,41],np.repeat(ids,6))
        dataset=FKDDataset(tmp_path/'built',cfg,split)
        assert len(dataset)==12 and len(dataset.shards)==2


def test_failed_next_input_flushes_previous_carried_windows(tmp_path):
    first,_=raw_fixture(tmp_path/'first',environments=2,source='/bags/first')
    second,rows=raw_fixture(tmp_path/'second',environments=2,source='/bags/second')
    rows[...,3:7]=0;rows.tofile(second)
    cfg=DataConfig(history_steps=4,future_steps=3,stride=2,shard_windows=10)
    with pytest.raises(ValueError,match='Zero quaternion'):
        build_dataset(tmp_path/'built',[spec(first),spec(second)],cfg,device='cpu')
    target=tmp_path/'built/road/real'
    manifest=json.loads((target/'manifest.json').read_text())
    assert [s['shape'][0] for s in manifest['processed_shards']]==[6]
    assert len(manifest['episodes'])==1
    assert all(e['windows']==e['stored_windows']==6 for e in manifest['episodes'].values())
    assert len(FKDDataset(tmp_path/'built',cfg))==6
