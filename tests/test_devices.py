"""Hardware inventory and indexed routing, without generating user collections."""
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from optcar.utils import devices


@pytest.fixture
def two_gpus(monkeypatch):
    monkeypatch.setattr(devices.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(devices.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(devices.torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(devices.torch.cuda, "get_device_properties", lambda i:
        SimpleNamespace(name=f"Test GPU {i}", total_memory=(16 + 8*i) * 2**30))


def test_all_gpus_enumerated_and_api(tmp_path, two_gpus):
    from optcar.gui.server import create_app
    inventory = devices.available_devices()
    assert inventory["default"] == "cuda:0"
    assert inventory["devices"] == [
        {"id": "cpu", "name": "CPU", "memory_gib": None},
        {"id": "cuda:0", "name": "Test GPU 0", "memory_gib": 16.0},
        {"id": "cuda:1", "name": "Test GPU 1", "memory_gib": 24.0},
    ]
    response = create_app(tmp_path).test_client().get("/api/devices")
    assert response.status_code == 200 and response.json == inventory


def test_cpu_inventory_without_cuda(monkeypatch):
    monkeypatch.setattr(devices.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(devices.torch.cuda, "get_device_properties", lambda _: pytest.fail("No GPU available"))
    inventory = devices.available_devices()
    assert inventory["default"] == "cpu" and len(inventory["devices"]) == 1
    assert devices.resolve_device("cpu") == "cpu"
    with pytest.raises(ValueError, match="CUDA is unavailable"):
        devices.resolve_device("cuda:0")


def test_indexed_selection(two_gpus):
    assert devices.resolve_device("cuda:0") == "cuda:0"
    assert devices.resolve_device("cuda:1") == "cuda:1"
    assert devices.resolve_device("cuda") == "cuda:1"
    with pytest.raises(ValueError, match="2 GPU"):
        devices.resolve_device("cuda:2")


@pytest.mark.parametrize("device", ["cuda:2", "cuda:-1", "nonsense", "meta"])
def test_invalid_device_creates_no_collection(tmp_path, two_gpus, device):
    from optcar.datagen.dbm_to_dataset import generate
    output = tmp_path / "collection"
    with pytest.raises(ValueError):
        generate(output, "road", device=device)
    assert not output.exists()


def test_gui_passes_selected_gpu_and_rejects_missing_gpu(tmp_path, monkeypatch, two_gpus):
    from optcar.gui import server
    from optcar.models.dynamic_bicycle import DBMParams
    from optcar.utils.file_io import terrain_path, write_yaml
    write_yaml(terrain_path(tmp_path, "road") / "dbm_params.yaml", {"params": asdict(DBMParams())})
    from optcar.datagen import generation_jobs
    import json
    called=[]
    def fake_tmux(argv,**kwargs):
        if argv[1]=='new-session':
            called.append(argv)
        return SimpleNamespace(stdout='',stderr='',returncode=0)
    monkeypatch.setattr(generation_jobs.shutil,'which',lambda _: '/test/tmux')
    monkeypatch.setattr(generation_jobs.subprocess,'run',fake_tmux)
    client = server.create_app(tmp_path).test_client()
    response = client.post("/api/generate", json={"terrain": "road", "device": "cuda:1"})
    assert response.status_code == 200 and len(called)==1
    record=json.loads((tmp_path/'.generation_jobs'/response.json['job_id']/'status.json').read_text())
    assert record['device']=='cuda:1'
    response = client.post("/api/generate", json={"terrain": "road", "device": "cuda:2"})
    assert response.status_code == 400 and "unavailable" in response.json["error"]
    assert len(called)==1
