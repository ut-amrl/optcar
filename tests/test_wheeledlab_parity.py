"""Optional numerical parity checks against the local WheeledLab source, read only.

No ROS/Isaac/CuPy import or simulation dataset is required. The reference functions
are loaded from their source independently of OptCar's implementations.
"""
import ast
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
