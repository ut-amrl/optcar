"""Independent frozen expectations for state frames, quaternion signs and DBM dynamics."""
import pytest
import torch
from optcar.dataset import pack_windows, chunk_view
from optcar.models.dynamic_bicycle import DynamicBicycleModel, _rk4_step, _rollout_torch


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("width", [15, 17])
def test_world_body_delta_and_sliding_windows(oracle, device, width):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    raw = torch.from_numpy(oracle["frame.raw_seed42"]).to(device)
    if width == 17:
        raw = torch.cat((raw, raw[..., 13:15]), -1)
    before = raw.clone()
    actual = pack_windows(raw[..., :13], raw[..., 13:15], 6)
    torch.testing.assert_close(actual[..., :41].cpu(), torch.from_numpy(oracle["frame.source_processed"]), atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(actual[..., 39:41], raw[..., 13:15], atol=0, rtol=0)
    chunks = chunk_view(torch.cat((raw, raw), 1), 11, 2).reshape(-1,11,width)
    torch.testing.assert_close(chunks[..., :15].cpu(), torch.from_numpy(oracle["frame.source_chunks"]), atol=0, rtol=0)
    torch.testing.assert_close(raw, before, atol=0, rtol=0)


@pytest.mark.parametrize("dt", [.005, .02, .04])
def test_rk4_rollout_and_wheel_states(oracle, dt):
    model = DynamicBicycleModel()
    x, u = (torch.from_numpy(oracle['dbm.input_' + key]) for key in ('x', 'u'))
    key = str(dt).replace('.', 'p')
    torch.testing.assert_close(_rk4_step(x, u, dt, **model.tensors), torch.from_numpy(oracle['dbm.rk4.' + key]), atol=1e-6, rtol=1e-6)
    q = torch.cat((torch.cos(x[:,2:3]/2), torch.zeros((3,2)), torch.sin(x[:,2:3]/2)), -1)
    pos = torch.cat((x[:,:2], torch.zeros((3,1))), -1)
    states, wheel = _rollout_torch(pos, q, x[:,3:6], x[:,6], u[:,None,:].expand(-1,12,-1), dt, **model.tensors)
    torch.testing.assert_close(states, torch.from_numpy(oracle['dbm.rollout.' + key]), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(wheel, torch.from_numpy(oracle['dbm.wheel.' + key]), atol=1e-6, rtol=1e-6)
