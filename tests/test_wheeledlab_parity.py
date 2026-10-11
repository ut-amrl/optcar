"""Optional numerical parity checks against the local WheeledLab source, read only.

No ROS/Isaac/CuPy import or simulation dataset is required. The reference functions
are loaded from their source independently of OptCar's implementations.
"""
import ast
import copy
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple,Union

import numpy as np
import pytest
import torch

from optcar.dataset import pack_windows
from optcar.models.dynamic_bicycle import DynamicBicycleModel,_rk4_step,_rollout_torch
from optcar.utils.math import euler_to_quat_torch

ROOT=Path(os.environ.get('WHEELEDLAB_SOURCE',Path(__file__).resolve().parents[2]/'WheeledLab/source'))
pytestmark=pytest.mark.skipif(not ROOT.is_dir(),reason='Local WheeledLab source is unavailable')


def load_functions(path,names,namespace):
    source=path.read_text();tree=ast.parse(source);functions=[]
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name in names:
            node.decorator_list=[];functions.append(node)
        if isinstance(node,ast.ClassDef):
            for method in node.body:
                if isinstance(method,ast.FunctionDef) and method.name in names:
                    method.decorator_list=[];functions.append(method)
    assert {node.name for node in functions}==set(names)
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),namespace)


@pytest.fixture
def reference():
    ns={'torch':torch,'np':np,'Tuple':Tuple,'Union':Union}
    load_functions(ROOT/'wheeledlab_utils/wheeledlab_utils/math.py',
        ['cross_product_3d','qinverse_torch','qrotate_torch','qmultiply_torch','qrelative_torch','wrap_to_pi','yaw_from_quat_torch'],ns)
    load_functions(ROOT/'wheeledlab_utils/wheeledlab_utils/filter.py',['quaternion_continuity_filter_batch'],ns)
    load_functions(ROOT/'wheeledlab_utils/wheeledlab_utils/transform_utils.py',['horizon_world_to_body_frame','compute_delta_frame_representation'],ns)
    load_functions(ROOT/'wheeledlab_data/wheeledlab_data/data_processing/base.py',['_process_chunks_to_frames_torch','generate_chunks_torch'],ns)
    load_functions(ROOT/'wheeledlab_models/wheeledlab_models/analytical/bicycle/dynamic_bicycle_model.py',
        ['_compute_dx_dynamic_ref','_step_dynamic_bicycle_ref_RK4','_batch_rollout_dynamic_bicycle_jit_ref'],ns)
    return ns


@pytest.mark.parametrize('width',[15,17])
def test_processed_world_body_delta_and_actions_match(reference,width):
    rng=np.random.default_rng(42);batch,steps=3,11
    raw=torch.tensor(rng.normal(size=(batch,steps,width)),dtype=torch.float32)
    raw[...,3:7]=euler_to_quat_torch(torch.tensor(rng.normal(size=(batch*steps,3)),dtype=torch.float32)).reshape(batch,steps,4)
    raw[:,1::2,3:7]*=-1
    raw[:,5:,3:7]*=-1
    processor=SimpleNamespace(history_horizon=6,total_horizon=11,stride=2)
    expected=reference['_process_chunks_to_frames_torch'](processor,raw)
    actual=pack_windows(raw[...,:13],raw[...,13:15],6)
    torch.testing.assert_close(actual[...,:41],expected,atol=0,rtol=0)
    assert torch.equal(raw[...,13:15],actual[...,39:41])
    # Check the window indexing against WheeledLab rather than duplicating it.
    chunks=reference['generate_chunks_torch'](processor,torch.cat((raw,raw),1))
    assert chunks.shape==(18,11,width)
    torch.testing.assert_close(chunks[0],raw[0])


@pytest.mark.parametrize('dt',[.005,.02,.04])
def test_dbm_rk4_and_world_states_match(reference,dt):
    model=DynamicBicycleModel()
    x=torch.tensor([[1,2,3.13,2,.15,.2,2.3],[-1,3,-3.13,-1,.1,-.3,-1.5],[0,0,0,0,0,0,0]],dtype=torch.float32)
    u=torch.tensor([[3,.7],[-8,-.7],[0,0]],dtype=torch.float32)
    expected=reference['_step_dynamic_bicycle_ref_RK4'](x,u,dt,**model.tensors)
    actual=_rk4_step(x,u,dt,**model.tensors)
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)
    q=torch.cat((torch.cos(x[:,2:3]/2),torch.zeros((3,2)),torch.sin(x[:,2:3]/2)),-1)
    body=x[:,3:6];pos=torch.cat((x[:,:2],torch.zeros((3,1))),-1)
    commands=u[:,None,:].expand(-1,12,-1)
    expected,wheel=reference['_batch_rollout_dynamic_bicycle_jit_ref'](pos,q,body,x[:,6],commands,dt,**model.tensors)
    actual,actual_wheel=_rollout_torch(pos,q,body,x[:,6],commands,dt,**model.tensors)
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(actual_wheel,wheel,atol=1e-6,rtol=1e-6)


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_batched_chunk_view_and_gpu_frames_match_wheeledlab(reference, device):
    from optcar.dataset import chunk_view
    if device.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('CUDA is unavailable')
    rng=np.random.default_rng(71)
    raw=torch.tensor(rng.normal(size=(4,29,15)),dtype=torch.float32,device=device)
    raw[...,3:7]=euler_to_quat_torch(torch.tensor(rng.normal(size=(4*29,3)),dtype=torch.float32,device=device)).reshape(4,29,4)
    raw[:,1::2,3:7]*=-1
    untouched=raw.clone()
    processor=SimpleNamespace(history_horizon=6,total_horizon=11,stride=3)
    expected_chunks=reference['generate_chunks_torch'](processor,raw)
    actual_chunks=chunk_view(raw,11,3).contiguous().reshape(-1,11,15)
    torch.testing.assert_close(actual_chunks,expected_chunks,atol=0,rtol=0)
    expected=reference['_process_chunks_to_frames_torch'](processor,expected_chunks)
    # Processing bounded batches must produce the same windows and signs as
    # WheeledLab's whole-file processing, without modifying overlapping input.
    actual=torch.cat([pack_windows(chunk[...,:13],chunk[...,13:15],6,7)
                      for chunk in actual_chunks.split(5)])
    torch.testing.assert_close(actual[...,:41],expected,atol=1e-6 if device.startswith('cuda') else 0,rtol=1e-6 if device.startswith('cuda') else 0)
    torch.testing.assert_close(actual[...,39:41],expected[...,39:41],atol=0,rtol=0)
    assert torch.all(actual[...,41]==7)
    torch.testing.assert_close(raw,untouched,atol=0,rtol=0)


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_builder_files_match_wheeledlab_whole_file_processor_and_writer(tmp_path, reference, device):
    """Exercise the user's 250/50/75 settings through the source's save routine."""
    import time
    from typing import Optional
    from optcar.dataset import DataConfig, FKDDataset
    from optcar.datagen.build_dataset import build_dataset
    if device.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('CUDA is unavailable')
    rng = np.random.default_rng(80)
    raw = torch.tensor(rng.normal(size=(32, 500, 15)), dtype=torch.float32)
    raw[..., 3:7] = euler_to_quat_torch(torch.tensor(rng.normal(size=(32*500, 3)), dtype=torch.float32)).reshape(32, 500, 4)
    raw[:, 1::2, 3:7] *= -1
    raw[5, 20, 7] = float('nan')
    input_dir = tmp_path/'raw'; input_dir.mkdir()
    file = input_dir/'run.bin'; raw.numpy().tofile(file)
    original = file.read_bytes()
    cfg = DataConfig(history_steps=250, future_steps=50, stride=75, shard_windows=16)
    source = {'path': str(file), 'terrain': 'road', 'source': 'sim', 'split': 'train',
              'columns': 15, 'num_envs': 32, 'dt': .02}
    result = build_dataset(tmp_path/'built', [source], cfg, device=device)
    assert result['new_windows'] == 31*3
    processor = SimpleNamespace(history_horizon=250, future_horizon=51, total_horizon=300, stride=75)
    chunks = reference['generate_chunks_torch'](processor, raw[~torch.isnan(raw).any(dim=1).any(dim=1)].to(device))
    expected = reference['_process_chunks_to_frames_torch'](processor, chunks).cpu()
    expected = torch.cat((expected, torch.zeros((*expected.shape[:2], 1))), -1)
    ns = {'torch': torch, 'np': np, 'Union': Union, 'Path': Path, 'time': time, 'Optional': Optional}
    load_functions(ROOT/'wheeledlab_data/wheeledlab_data/data_processing/processing_sim.py', ['save_processed_batch'], ns)
    reference_dir = tmp_path/'reference'; reference_dir.mkdir()
    for index, values in enumerate(expected.split(cfg.shard_windows)):
        ns['save_processed_batch'](SimpleNamespace(timing=False), values, reference_dir, index)
    actual_files = sorted((tmp_path/'built/processed/sim/road/train').glob('*.bin'))
    expected_files = sorted(reference_dir.glob('*.bin'))
    assert len(actual_files) == len(expected_files) == 6
    for actual, expected_file in zip(actual_files, expected_files):
        if device == 'cpu':
            assert actual.read_bytes() == expected_file.read_bytes()
        else:
            # The independently loaded source uses eager math; OptCar retains
            # WheeledLab's scripted GPU math, whose fusion can round differently.
            np.testing.assert_allclose(np.fromfile(actual, np.float32), np.fromfile(expected_file, np.float32),
                                       atol=2e-6, rtol=2e-6)
    assert not (tmp_path/'built/processed/sim/road/raw').exists()
    assert file.read_bytes() == original
    assert len(FKDDataset(tmp_path/'built', cfg)) == 93


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_cross_file_output_batches_match_wheeledlab_carryover(tmp_path, reference, device):
    """Run WheeledLab's own nested carryover function, including its final flush."""
    import time
    from typing import Optional
    from optcar.dataset import DataConfig, FKDDataset
    from optcar.datagen.build_dataset import build_dataset
    if device.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('CUDA is unavailable')
    source_path=ROOT/'wheeledlab_data/wheeledlab_data/data_processing/processing_sim.py'
    tree=ast.parse(source_path.read_text())
    processor_class=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='RawDatasetProcessor')
    method=next(node for node in processor_class.body if isinstance(node,ast.FunctionDef) and node.name=='process_dataset_folder')
    ingest=next(node for node in method.body if isinstance(node,ast.FunctionDef) and node.name=='_ingest_processed')
    final_flush=next(node for node in ast.walk(method) if isinstance(node,ast.If)
                     and ast.unparse(node.test)=='carryover is not None and carryover_chunks > 0'
                     and 'Saving final batch' in ast.unparse(node))
    factory=ast.parse('''
def reference_writer(self, output_data_folder, chunks_per_file):
    carryover=None
    carryover_chunks=0
    batch_save_idx=0
    total_chunks=0
    chunks_per_file_list=[]
''').body[0]
    factory.body.append(copy.deepcopy(ingest))
    flush=ast.parse('''
def flush():
    nonlocal carryover, carryover_chunks, batch_save_idx, total_chunks
''').body[0]
    flush.body.append(copy.deepcopy(final_flush))
    factory.body.extend([flush,ast.parse('return _ingest_processed, flush').body[0]])
    ns={'torch':torch,'np':np,'Union':Union,'Path':Path,'time':time,'Optional':Optional}
    load_functions(source_path,['save_processed_batch'],ns)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[factory],type_ignores=[])),str(source_path),'exec'),ns)
    processor=SimpleNamespace(history_horizon=4,total_horizon=7,stride=2,timing=False)
    processor.save_processed_batch=lambda values,path,index:ns['save_processed_batch'](processor,values,path,index)
    reference_dir=tmp_path/'reference';reference_dir.mkdir()
    ingest_reference,flush_reference=ns['reference_writer'](processor,reference_dir,10)
    rng=np.random.default_rng(82)
    sources=[]
    for index in range(3):
        raw=torch.tensor(rng.normal(size=(2,12,15)),dtype=torch.float32,device=device)
        raw[...,3:7]=euler_to_quat_torch(torch.tensor(rng.normal(size=(24,3)),dtype=torch.float32,device=device)).reshape(2,12,4)
        raw[:,1::2,3:7]*=-1
        file=tmp_path/f'raw_{index}.bin';raw.cpu().numpy().tofile(file)
        sources.append({'path':str(file),'terrain':'road','source':'sim','split':'train',
                        'columns':15,'num_envs':2,'dt':.02})
        chunks=reference['generate_chunks_torch'](processor,raw)
        expected=reference['_process_chunks_to_frames_torch'](processor,chunks).cpu()
        expected=torch.cat((expected,torch.full((*expected.shape[:2],1),float(index))),-1)
        ingest_reference(expected)
    flush_reference()
    cfg=DataConfig(history_steps=4,future_steps=3,stride=2,shard_windows=10)
    result=build_dataset(tmp_path/'built',sources,cfg,device=device)
    actual_files=sorted((tmp_path/'built/processed/sim/road/train').glob('*.bin'))
    expected_files=sorted(reference_dir.glob('*.bin'))
    assert result['new_windows']==18 and len(actual_files)==len(expected_files)==2
    assert [file.stat().st_size//(7*42*4) for file in actual_files]==[10,8]
    for actual,expected in zip(actual_files,expected_files):
        if device=='cpu':
            assert actual.read_bytes()==expected.read_bytes()
        else:
            np.testing.assert_allclose(np.fromfile(actual,np.float32),np.fromfile(expected,np.float32),atol=2e-6,rtol=2e-6)
    assert len(FKDDataset(tmp_path/'built',cfg))==18
