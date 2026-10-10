"""Integration checks with real ROS2 serialization, without ROS installed."""
from dataclasses import replace
import json
import fcntl
from pathlib import Path
import shutil
import time

import numpy as np
import pytest
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore

from optcar.datagen.raw_bags import RawBagConfig, convert_raw_bags, discover_bags, read_raw_bag


def _recording(tmp_path, include_aux=False):
    root = tmp_path / "recordings"
    bag = root / "road" / "v2.0-s0.0"
    bag.parent.mkdir(parents=True)
    store = get_typestore(Stores.ROS2_HUMBLE)
    types = store.types
    V = types["geometry_msgs/msg/Vector3"]
    P = types["geometry_msgs/msg/Point"]
    Q = types["geometry_msgs/msg/Quaternion"]
    Pose = types["geometry_msgs/msg/Pose"]
    Twist = types["geometry_msgs/msg/Twist"]
    Header = types["std_msgs/msg/Header"]
    Stamp = types["builtin_interfaces/msg/Time"]
    origin = 1_790_000_000_000_000_000
    with Writer(bag, version=9) as writer:
        connections = {topic: writer.add_connection(topic, kind, typestore=store) for topic, kind in (
            ("/super_odometry/pose", "geometry_msgs/msg/PoseStamped"),
            ("/super_odometry/twist_wf", "geometry_msgs/msg/TwistStamped"),
            ("/cmd_vel", "geometry_msgs/msg/Twist"))}
        aux_connection = writer.add_connection("/super_odometry/twist", "geometry_msgs/msg/TwistStamped", typestore=store) if include_aux else None
        for i in range(11):
            header = Header(Stamp(1790000000, i * 20_000_000), "world")
            # Offset observations after the grid boundary: a sample-and-hold
            # converter must retain the previous observation, not interpolate.
            stamp = origin + i * 20_000_000 + (5_000_000 if i else 0)
            pose = types["geometry_msgs/msg/PoseStamped"](header, Pose(P(float(i), 0., 0.), Q(0., 0., 0., 2.)))
            twist = types["geometry_msgs/msg/TwistStamped"](header, Twist(V(float(i), 0., 0.), V(0., 0., 0.)))
            command = Twist(V(100. + i, 0., 0.), V(0., 0., .1 * i))
            for topic, msg in (("/super_odometry/pose", pose), ("/super_odometry/twist_wf", twist), ("/cmd_vel", command)):
                connection = connections[topic]
                writer.write(connection, stamp, store.serialize_cdr(msg, connection.msgtype))
            if aux_connection is not None and 2 <= i <= 7:
                body_header = Header(Stamp(1790000000, i * 20_000_000), "base_link")
                body_twist = types["geometry_msgs/msg/TwistStamped"](body_header, Twist(V(700. + i, -2., 1.), V(.1, .2, .3)))
                writer.write(aux_connection, stamp, store.serialize_cdr(body_twist, aux_connection.msgtype))
    return root, bag


@pytest.fixture
def recording(tmp_path):
    return _recording(tmp_path)


def test_wheeledlab_sample_hold_and_raw_layout(recording):
    root, bag = recording
    scan = discover_bags(root)
    assert len(scan["bags"]) == 1
    assert scan["bags"][0]["output_relative"] == "data/recordings/road/v2.0-s0.0.bin"
    assert scan["bags"][0]["messages"] == 33
    rows, info = read_raw_bag(bag, RawBagConfig())
    assert rows.shape == (11, 15)
    np.testing.assert_array_equal(rows[:4, 0], [0, 0, 1, 2])
    np.testing.assert_array_equal(rows[:4, 7], [0, 0, 1, 2])
    np.testing.assert_array_equal(rows[:4, 13], [100, 100, 101, 102])
    np.testing.assert_array_equal(rows[:, 3], 1)
    assert info["samples"] == 11


def test_shift_extra_commands_and_atomic_overwrite(recording, tmp_path):
    root, bag = recording
    cfg = RawBagConfig(action_shift=.02, include_cmd_vel=True)
    rows, _ = read_raw_bag(bag, cfg)
    assert rows.shape[1] == 17
    # Shifted actions use older observations than the additional raw commands.
    assert rows[1, 13] == 100 and rows[1, 15] == 101
    output = tmp_path / "bins"
    result = convert_raw_bags(root, [bag], output, cfg)
    assert result["converted"] == 1 and result["failed"] == 0
    target = Path(result["results"][0]["output"])
    before = target.read_bytes()
    refused = convert_raw_bags(root, [bag], output, cfg)
    assert refused["failed"] == 1 and target.read_bytes() == before
    extra = output / "data/unrelated.bin"
    extra.write_bytes(b"untouched")
    assert convert_raw_bags(root, [bag], output, cfg, overwrite=True)["converted"] == 1
    assert extra.read_bytes() == b"untouched"
    assert json.loads((output / "metadata/schema.json").read_text())["features"] == 17
    with pytest.raises(ValueError, match="different conversion settings"):
        convert_raw_bags(root, [bag], output, replace(cfg, dt=.01))


def test_missing_topics_metadata_selection_and_cancellation(recording, tmp_path):
    root, bag = recording
    broken = root / "broken"
    broken.mkdir()
    (broken / "empty.db3").touch()
    assert discover_bags(root)["bags"][0]["error"]
    failed = convert_raw_bags(root, [bag], tmp_path / "missing", RawBagConfig(command_topic="/missing"))
    assert failed["failed"] == 1 and not list((tmp_path / "missing").rglob("*.bin"))
    stop = False
    def update(p):
        nonlocal stop
        if p["phase"] == "reading":
            stop = True
    cancelled = convert_raw_bags(root, [bag], tmp_path / "cancelled", progress=update, cancelled=lambda: stop)
    assert cancelled["cancelled"] and cancelled["converted"] == 0
    assert not list((tmp_path / "cancelled").rglob("*.bin"))
    with pytest.raises(ValueError, match="outside"):
        convert_raw_bags(bag, [root], tmp_path / "outside")


def test_gui_uses_same_converter(recording, tmp_path):
    from optcar.gui.server import create_app
    root, bag = recording
    app = create_app(tmp_path / "datasets")
    client = app.test_client()
    assert client.get("/").status_code == 200
    assert len(client.post("/api/raw-bags/scan", json={"root": str(root)}).json["bags"]) == 1
    assert client.post("/api/raw-bags/convert", json={"root": str(root), "bags": []}).status_code == 400
    cli_result = convert_raw_bags(root, [bag], tmp_path / "direct")
    submitted = client.post("/api/raw-bags/convert", json={"root": str(root), "bags": [str(bag)], "output": str(tmp_path / "gui")})
    assert submitted.status_code == 200
    identifier = submitted.json["job_id"]
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{identifier}").json
        if job["status"] not in ("queued", "running"):
            break
        time.sleep(.01)
    assert job["status"] == "complete", job
    assert job["progress"]["percent"] == 100
    assert Path(job["result"]["results"][0]["output"]).read_bytes() == Path(cli_result["results"][0]["output"]).read_bytes()


def test_cancel_retains_completed_bags_and_output_lock(recording, tmp_path):
    root, bag = recording
    second = root / "road/second"
    shutil.copytree(bag, second)
    stop = False
    def update(progress):
        nonlocal stop
        if progress["phase"] == "complete":
            stop = True
    output = tmp_path / "retained"
    result = convert_raw_bags(root, [bag, second], output, progress=update, cancelled=lambda: stop)
    assert result["cancelled"] and result["converted"] == 1
    assert len(list(output.rglob("*.bin"))) == 1
    assert len(json.loads((output / "metadata/conversion.json").read_text())["bins"]) == 1
    with (output / ".conversion.lock").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="Another conversion"):
            convert_raw_bags(root, [second], output)


@pytest.mark.parametrize('extra_commands', [False, True])
def test_auxiliary_alignment_preserves_main_bins(tmp_path, extra_commands):
    from optcar.datagen.raw_bins import inspect_bin, bin_frame, scan_bins
    root, bag = _recording(tmp_path, include_aux=True)
    cfg = RawBagConfig(include_cmd_vel=extra_commands, action_shift=.02 if extra_commands else 0.)
    baseline, _ = read_raw_bag(bag, replace(cfg, aux_twist_topic=None))
    result = convert_raw_bags(root, [bag], tmp_path / 'bins', cfg)
    assert result['converted'] == 1
    target = Path(result['results'][0]['output'])
    assert target.read_bytes() == baseline.tobytes()
    manifest = json.loads((tmp_path / 'bins/metadata/conversion.json').read_text())
    record = manifest['bins'][str(target.relative_to(tmp_path / 'bins'))]
    aux_meta = record['aux']
    aux = np.fromfile(tmp_path / 'bins' / aux_meta['path'], dtype='<f4').reshape(-1, 6)
    assert aux.shape == (len(baseline), 6)
    assert aux_meta['frame_ids'] == ['base_link']
    assert aux_meta['dt'] == cfg.dt and aux_meta['start_ns'] == record['start_ns']
    assert np.isnan(aux[0]).all() and np.isnan(aux[-1]).all()
    assert set(aux[np.isfinite(aux[:, 0]), 0]).issubset(set(range(702, 708)))
    view = inspect_bin(str(target))
    assert view['forward_source'] == '/super_odometry/twist · body x'
    assert view['forward_velocity'][0] is None
    middle = next(i for i in range(len(aux)) if np.isfinite(aux[i, 0]))
    assert bin_frame(str(target), middle)['forward_velocity'] == aux[middle, 0]
    assert view['forward_stats']['missing'] == int(np.isnan(aux[:, 0]).sum())
    assert len(scan_bins(str(tmp_path / 'bins'))['bins']) == 1
    with pytest.raises(ValueError, match='Auxiliary twist bins are diagnostics'):
        inspect_bin(str(tmp_path / 'bins' / aux_meta['path']))


def test_auxiliary_upgrade_and_disable(tmp_path):
    root, bag = _recording(tmp_path, include_aux=True)
    output = tmp_path / 'bins'
    cfg = RawBagConfig(aux_twist_topic=None)
    result = convert_raw_bags(root, [bag], output, cfg)
    target = Path(result['results'][0]['output'])
    baseline = target.read_bytes()
    manifest_path = output / 'metadata/conversion.json'
    manifest = json.loads(manifest_path.read_text())
    del manifest['config']['aux_twist_topic']  # Existing pre-auxiliary dataset.
    manifest_path.write_text(json.dumps(manifest))
    assert convert_raw_bags(root, [bag], output, RawBagConfig(), overwrite=True)['converted'] == 1
    assert target.read_bytes() == baseline
    aux = list((output / 'aux').rglob('*.bin'))
    assert len(aux) == 1
    assert convert_raw_bags(root, [bag], output, cfg, overwrite=True)['converted'] == 1
    assert not aux[0].exists()
    assert 'aux' not in next(iter(json.loads(manifest_path.read_text())['bins'].values()))


def test_missing_auxiliary_does_not_block_main(recording, tmp_path):
    root, bag = recording
    result = convert_raw_bags(root, [bag], tmp_path / 'bins')
    assert result['converted'] == 1 and result['failed'] == 0
    assert 'No auxiliary twist messages' in result['results'][0]['warning']
    assert not list((tmp_path / 'bins/aux').rglob('*.bin'))
