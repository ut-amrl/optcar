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
    manifest=json.loads((tmp_path/'assembled/processed/real/road/manifest.json').read_text())
    episodes=list(manifest['episodes'].values())
    assert len(episodes)==1
    assert episodes[0]['raw']==str(file)
    assert not (tmp_path/'assembled/processed/real/road/raw').exists()
    assert episodes[0]['shards']==[] and episodes[0]['stored_windows']==6
    first=np.fromfile(tmp_path/'assembled/processed/real/road'/manifest['processed_shards'][0]['path'],np.float32).reshape(-1,7,42)
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
        assert (parent / 'processed' / 'sim' / terrain / 'manifest.json').is_file()
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
    assert result['dataset'] == str(output / 'processed')
    assert file.read_bytes() == original and note.read_text() == 'keep this file'
    # Scanning the original folder again must exclude this build's raw copies
    # and processed shards, including when they live within the input tree.
    plan = inspect_sources([spec(file.parent)], cfg)
    assert plan['files'] == 1 and not plan['errors']
    assert build_dataset(output, [spec(file.parent)], cfg)['new_windows'] == 0


@pytest.mark.parametrize('split,relative', [
    ('train', 'metadata/schema.json'), ('train', 'metadata/dataset_config.py'),
    ('train', 'train/processed_0000000000.bin'), ('val', 'valid/processed_0000000000.bin'),
    ('test', 'test/processed_0000000000.bin'),
])
def test_existing_output_file_is_preserved(tmp_path, split, relative):
    file, _ = raw_fixture(tmp_path / 'inputs')
    output = tmp_path / 'output'
    conflict = output / 'processed' / 'real' / 'road' / relative
    conflict.parent.mkdir(parents=True)
    conflict.write_text('existing user metadata')
    with pytest.raises(ValueError, match='existing file'):
        build_dataset(output, [spec(file, split=split)], DataConfig(history_steps=3, future_steps=2))
    assert conflict.read_text() == 'existing user metadata'
    assert not (output / 'processed' / 'dataset_build.json').exists()


def test_source_terrain_split_layout_keeps_processed_metadata_and_raw_inputs(tmp_path):
    sources, originals = [], {}
    groups = [('real', 'road'), ('sim', 'grass')]
    for split in ('train', 'val', 'test'):
        for kind, terrain in groups:
            file, rows = raw_fixture(tmp_path / 'inputs' / kind / terrain / split,
                                     environments=2, source=f'/bags/{kind}/{terrain}/{split}')
            rows[..., 0] += len(sources) * 100
            rows.tofile(file)
            sources.append(spec(file, source=kind, terrain=terrain, split=split))
            originals[file] = file.read_bytes()
    cfg = DataConfig(history_steps=4, future_steps=3, stride=2, shard_windows=4)
    output = tmp_path / 'chosen_output'
    result = build_dataset(output, sources, cfg, device='cpu', workers=2)
    assert result['dataset'] == str(output / 'processed')
    assert result['completed'] == 6 and result['new_windows'] == 36
    assert {folder.name for folder in (output / 'processed').iterdir() if folder.is_dir()} == {'real', 'sim'}
    for kind, terrain in groups:
        group = output / 'processed' / kind / terrain
        assert {folder.name for folder in group.iterdir() if folder.is_dir()} == {'train', 'test', 'valid', 'metadata'}
        assert not (group / 'raw').exists() and not (group / 'processed').exists()
        schema = json.loads((group / 'metadata/schema.json').read_text())
        assert schema['features'] == 42 and schema['feature_dim'] == 42
        assert schema['dtype'] == 'float32' and schema['quaternion_order'] == 'wxyz'
        assert schema['history_horizon'] == 4 and schema['future_horizon'] == 4
        assert schema['total_horizon'] == 7 and schema['control_dt'] == .02
        config_text = (group / 'metadata/dataset_config.py').read_text()
        assert 'class ProcessedDatasetConfig:' in config_text
        assert 'class DatasetConfig:' not in config_text
        manifest = json.loads((group / 'manifest.json').read_text())
        assert manifest['flat_processed'] is True
        for split, folder in [('train', 'train'), ('val', 'valid'), ('test', 'test')]:
            shards = [entry for entry in manifest['processed_shards'] if entry['split'] == split]
            assert [entry['shape'] for entry in shards] == [[4, 7, 42], [2, 7, 42]]
            assert [entry['path'] for entry in shards] == [
                f'{folder}/processed_0000000000.bin', f'{folder}/processed_0000000001.bin']
            dataset = FKDDataset(group / folder, cfg, split)
            assert len(dataset) == 6
            assert all(entry[3] == terrain and entry[2] == kind for entry in dataset.shards)
            recorded = np.concatenate([np.fromfile(group / entry['path'], np.float32).reshape(entry['shape'])
                                       for entry in shards])
            source_index = next(index for index, entry in enumerate(sources)
                                if entry['source'] == kind and entry['split'] == split)
            np.testing.assert_array_equal(recorded[:, 0, 41], np.full(6, source_index))
            np.testing.assert_allclose(recorded[:, 0, 0], source_index * 100 + np.tile([0, .2, .4], 2), atol=1e-5)
            np.testing.assert_allclose(recorded[:, 0, 39], np.tile([0, .4, .8], 2))
    assert all(file.read_bytes() == content for file, content in originals.items())
    assert len(FKDDataset(output / 'processed', cfg, 'all')) == 36
    assert build_dataset(output, sources, cfg, device='cpu')['new_windows'] == 0


def test_raw_only_metadata_saved_split_and_standard_nan_policy(tmp_path):
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
    response=client.post('/api/generate',json={'terrain':'road'})
    assert response.status_code==200 and response.json['tmux_session']=='optcar-test'
    assert called==[('road','default')]
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
    cpu_shards=sorted((tmp_path/'cpu/processed/real/road/train').glob('*.bin'))
    gpu_shards=sorted((tmp_path/'gpu/processed/real/road/train').glob('*.bin'))
    assert len(cpu_shards)==len(gpu_shards)==(12+shard_windows-1)//shard_windows
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
    manifest=json.loads((tmp_path/'built/processed/sim/grass/manifest.json').read_text())
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
    target = output/'processed/real/road'
    # A stopped older builder has committed its first two environments.
    with DatasetWriter(target, cfg, 'real', 'road', {'inputs': entries, 'calibrations': []},
                       flat_processed=True) as writer:
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


def test_old_output_layout_is_rejected_before_creating_a_second_corpus(tmp_path):
    file, _ = raw_fixture(tmp_path / 'inputs', environments=2)
    cfg = DataConfig(history_steps=4, future_steps=3, stride=2, shard_windows=4)
    sources = [spec(file)]
    plan = inspect_sources(sources, cfg)
    output = tmp_path / 'chosen_output'
    root = output / 'processed'
    legacy_group = root / 'road' / 'real' / 'processed' / 'data'
    legacy_group.mkdir(parents=True)
    legacy_bin = legacy_group / 'existing.bin'
    np.arange(7 * 42, dtype=np.float32).reshape(1, 7, 42).tofile(legacy_bin)
    index = root / 'dataset_build.json'
    index.write_text(json.dumps({'config': plan['config'], 'bins': plan['bins'],
                                 'seed': plan['seed'], 'fractions': plan['fractions']}))
    original_bin, original_index = legacy_bin.read_bytes(), index.read_bytes()
    original_raw = file.read_bytes()

    with pytest.raises(ValueError, match='older nested output layout'):
        build_dataset(output, sources, cfg, device='cpu')

    assert not (root / 'real' / 'road').exists()
    assert list(root.rglob('*.bin')) == [legacy_bin]
    assert legacy_bin.read_bytes() == original_bin
    assert index.read_bytes() == original_index and file.read_bytes() == original_raw


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
    target=tmp_path/'built/processed/real/road'
    manifest=json.loads((target/'manifest.json').read_text())
    assert result['new_windows']==18
    assert [s['shape'] for s in manifest['processed_shards']]==[[10,7,42],[8,7,42]]
    assert [s['path'] for s in manifest['processed_shards']]==[
        'train/processed_0000000000.bin',
        'train/processed_0000000001.bin']
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
    target=tmp_path/'built/processed/real/road'
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
    target=tmp_path/'built/processed/real/road'
    manifest=json.loads((target/'manifest.json').read_text())
    assert [s['shape'][0] for s in manifest['processed_shards']]==[6]
    assert len(manifest['episodes'])==1
    assert all(e['windows']==e['stored_windows']==6 for e in manifest['episodes'].values())
    assert len(FKDDataset(tmp_path/'built',cfg))==6


def test_disjoint_real_source_appends_without_changing_completed_sim_data(tmp_path):
    sim_file, _ = raw_fixture(tmp_path / 'sim_inputs', environments=2, source='/bags/sim')
    real_file, _ = raw_fixture(tmp_path / 'real_inputs', environments=2, source='/bags/real')
    output = tmp_path / 'assembled'
    sim_config = DataConfig(history_steps=4, future_steps=3, stride=2, shard_windows=2)
    real_config = DataConfig(history_steps=4, future_steps=3, stride=1, shard_windows=2)
    sim_source = spec(sim_file.parent, source='sim')
    real_source = spec(real_file.parent, source='real')
    sim_result = build_dataset(output, [sim_source], sim_config, device='cpu')
    assert sim_result['new_windows'] == 6
    root = output / 'processed'

    def snapshot(folder):
        return {path: (path.read_bytes(), path.stat().st_ino, path.stat().st_size, path.stat().st_mtime_ns)
                for path in folder.rglob('*') if path.is_file()}

    sim_original = snapshot(root / 'sim')
    index = root / 'dataset_build.json'
    index_original = (index.read_bytes(), index.stat().st_ino, index.stat().st_size, index.stat().st_mtime_ns)
    real_result = build_dataset(output, [real_source], real_config, device='cpu')
    assert real_result['dataset'] == sim_result['dataset'] == str(root)
    assert real_result['new_windows'] == 12
    assert snapshot(root / 'sim') == sim_original
    assert (index.read_bytes(), index.stat().st_ino, index.stat().st_size, index.stat().st_mtime_ns) == index_original
    sim_manifest = json.loads((root / 'sim' / 'road' / 'manifest.json').read_text())
    real_manifest = json.loads((root / 'real' / 'road' / 'manifest.json').read_text())
    assert sim_manifest['config']['stride'] == 2 and real_manifest['config']['stride'] == 1
    assert len(list((root / 'real' / 'road' / 'train').glob('*.bin'))) == 6
    assert sum(shard['shape'][0] for shard in real_manifest['processed_shards']) == 12
    assert all(shard['shape'][1:] == [7, 42] for shard in real_manifest['processed_shards'])
    assert len(FKDDataset(root, sim_config)) == 18
    assert len(FKDDataset(root, real_config)) == 18

    completed = snapshot(root)
    for source, config in ((sim_source, sim_config), (real_source, real_config)):
        repeated = build_dataset(output, [source], config, device='cpu')
        assert repeated['new_windows'] == 0 and repeated['completed'] == 1
        assert snapshot(root) == completed
    changed = DataConfig(history_steps=4, future_steps=3, stride=1, shard_windows=2)
    with pytest.raises(ValueError, match='Inputs or settings changed for'):
        build_dataset(output, [sim_source], changed, device='cpu')
    assert snapshot(root) == completed
    sim_file.with_name('extra.bin').write_bytes(sim_file.read_bytes())
    with pytest.raises(ValueError, match='Inputs or settings changed for'):
        build_dataset(output, [sim_source], sim_config, device='cpu')
    assert snapshot(root) == completed


@pytest.mark.parametrize('history,future,dt', [(3, 4, .02), (4, 3, .04)])
def test_disjoint_source_rejects_incompatible_training_dimensions_without_writes(tmp_path, history, future, dt):
    sim_file, _ = raw_fixture(tmp_path / 'sim_inputs', source='/bags/sim')
    real_file, _ = raw_fixture(tmp_path / 'real_inputs', source='/bags/real', dt=dt)
    output = tmp_path / 'assembled'
    sim_config = DataConfig(history_steps=4, future_steps=3, dt=.02, stride=2, shard_windows=2)
    build_dataset(output, [spec(sim_file, source='sim')], sim_config, device='cpu')
    root = output / 'processed'
    original = {path: (path.read_bytes(), path.stat().st_ino, path.stat().st_size, path.stat().st_mtime_ns)
                for path in root.rglob('*') if path.is_file()}
    real_config = DataConfig(history_steps=history, future_steps=future, dt=dt, stride=1, shard_windows=3)
    assert real_config.window_steps == sim_config.window_steps == 7
    with pytest.raises(ValueError, match='(?i)history|future|timestep|training'):
        build_dataset(output, [spec(real_file, source='real')], real_config, device='cpu')
    assert not (root / 'real').exists()
    assert {path: (path.read_bytes(), path.stat().st_ino, path.stat().st_size, path.stat().st_mtime_ns)
            for path in root.rglob('*') if path.is_file()} == original
