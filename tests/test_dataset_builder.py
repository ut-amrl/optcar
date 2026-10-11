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
    assert len(episodes)==2
    first=np.fromfile(tmp_path/'assembled/road/real'/episodes[0]['shards'][0]['path'],np.float32).reshape(-1,7,42)
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
    with pytest.raises(ValueError,match='separate'):
        build_dataset(folder/'output',[spec(folder)],cfg)
    with pytest.raises(ValueError,match='summing'):
        inspect_sources([spec(folder)],cfg,fractions=[np.nan,0,0])


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
    def fake_generate(*args,**kwargs):
        called.append(kwargs['process_windows'])
        return {'dataset':str(args[0]),'completed':0,'total':0,'new_windows':0}
    monkeypatch.setattr(server,'generate',fake_generate)
    write_yaml(terrain_path(tmp_path/'datasets','road')/'dbm_params.yaml',{'params':asdict(DBMParams())})
    response=client.post('/api/generate',json={'terrain':'road','raw_only':True})
    assert response.status_code==200
    import time
    for _ in range(100):
        if client.get('/api/jobs/'+response.json['job_id']).json['status']=='complete':
            break
        time.sleep(.01)
    assert called==[False]


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
