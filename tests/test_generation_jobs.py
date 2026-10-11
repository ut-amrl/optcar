"""Real tmux lifecycle checks using tiny disposable CPU collections."""
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import pytest

from optcar.datagen.generation_jobs import GenerationJobs
from optcar.gui.server import create_app
from optcar.models.dynamic_bicycle import DBMParams
from optcar.utils.file_io import terrain_path, write_json, write_yaml


@pytest.fixture
def manager(tmp_path):
    if not shutil.which('tmux'):
        pytest.skip('tmux is not installed')
    manager=GenerationJobs(tmp_path/'datasets')
    write_yaml(terrain_path(manager.root,'road')/'dbm_params.yaml',{'params':asdict(DBMParams())})
    yield manager
    for record in manager.list():
        if record['status'] in ('queued','running','cancelling'):
            manager.cancel(record['job_id'])
    deadline=time.monotonic()+15
    while time.monotonic()<deadline and any(record['status'] in ('queued','running','cancelling') for record in manager.list()):
        time.sleep(.1)
    for record in manager.list():
        subprocess.run(['tmux','kill-session','-t',record['tmux_session']],capture_output=True)


def wait_finished(manager,identifier):
    deadline=time.monotonic()+25
    while time.monotonic()<deadline:
        record=manager.get(identifier)
        if record['status'] not in ('queued','running','cancelling'):
            return record
        time.sleep(.1)
    pytest.fail('tmux worker did not finish')


def test_parallel_workers_persist_and_reconnect(manager):
    # More than the previous two-job limit, all using separate tiny outputs.
    jobs=[manager.start('road',f'job-{i}',commands={'duration':.08},runs=2,batch_size=2) for i in range(3)]
    assert len({job['tmux_session'] for job in jobs})==3
    records=[wait_finished(manager,job['job_id']) for job in jobs]
    assert all(record['status']=='complete' for record in records),records
    for record in records:
        assert record['progress']['completed']==record['progress']['total']==8
        assert record['result']['trajectories']==4
        assert record['session_available']
        files=list((Path(record['dataset'])/'data').glob('*.bin'))
        assert len(files)==2 and files[0].stat().st_size==2*4*15*4
        assert np.isfinite(np.fromfile(files[0],np.float32)).all()
        pane=subprocess.run(['tmux','capture-pane','-p','-t',record['tmux_session']],capture_output=True,text=True,check=True).stdout
        assert 'Complete' in pane and record['tmux_session'] in pane
    # A brand-new GUI must recover the job registry without the original server.
    client=create_app(manager.root).test_client()
    assert len(client.get('/api/generation/runs').json['runs'])==3
    assert client.get('/api/jobs/'+jobs[0]['job_id']).json['status']=='complete'
    assert client.post('/api/jobs/'+jobs[0]['job_id']+'/cancel',json={}).json['status']=='complete'


def test_cancel_and_duplicate_output(manager):
    job=manager.start('road','cancel',commands={'duration':.08},runs=2,batch_size=2)
    with pytest.raises(ValueError,match='active collection'):
        manager.start('road','cancel',commands={'duration':.08},runs=2,batch_size=2)
    client=create_app(manager.root).test_client()
    response=client.post('/api/jobs/'+job['job_id']+'/cancel',json={})
    assert response.status_code==200
    record=wait_finished(manager,job['job_id'])
    assert record['status']=='cancelled'
    assert record['result']['completed']==0
    assert not list(Path(record['dataset']).rglob('*.bin'))


def test_lost_session_and_invalid_input(manager):
    with pytest.raises(ValueError,match='integer multiple'):
        manager.start('road','invalid',commands={'duration':.031},runs=1,batch_size=2)
    assert not manager.list()
    # Simulate a disappeared worker record without launching or killing unrelated sessions.
    identifier='f'*32
    write_json(manager.job_dir(identifier)/'status.json',{'job_id':identifier,'status':'running',
        'tmux_session':'optcar-does-not-exist-test','created_at':time.time()-60,'dataset':'missing'})
    record=manager.get(identifier)
    assert record['status']=='failed' and not record['session_available']
    assert 'closed' in record['error']


def test_missing_tmux_fails_before_writing(tmp_path,monkeypatch):
    monkeypatch.setattr('optcar.datagen.generation_jobs.shutil.which',lambda name:None)
    manager=GenerationJobs(tmp_path)
    with pytest.raises(ValueError,match='tmux is required'):
        manager.start('road')
    assert not manager.directory.exists()


def test_worker_failure_is_reported(manager):
    output=terrain_path(manager.root,'road')/'sim'/'blocked'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text('existing file')
    job=manager.start('road','blocked',commands={'duration':.08},runs=1,batch_size=2)
    record=wait_finished(manager,job['job_id'])
    assert record['status']=='failed' and record['error']
    assert record['session_available'] and output.read_text()=='existing file'


def test_cli_launches_same_manager(tmp_path,monkeypatch,capsys):
    from optcar.cli import main
    root=tmp_path/'datasets'
    write_yaml(terrain_path(root,'road')/'dbm_params.yaml',{'params':asdict(DBMParams())})
    config=tmp_path/'config.yaml'
    write_yaml(config,{'commands':{'duration':.08},'runs':2,'batch_size':2})
    calls=[]
    def fake_start(self,terrain,run,**kwargs):
        calls.append((self.root,terrain,run,kwargs))
        return {'job_id':'a'*32,'tmux_session':'optcar-cli-test','attach_command':'tmux attach -t optcar-cli-test'}
    monkeypatch.setattr(GenerationJobs,'start',fake_start)
    main(['generate','--dataset-root',str(root),'--terrain','road','--run','cli',
          '--config',str(config),'--device','cpu'])
    assert json.loads(capsys.readouterr().out)['tmux_session']=='optcar-cli-test'
    assert calls[0][:3]==(root,'road','cli')
    assert calls[0][3]['runs']==2 and calls[0][3]['commands'].duration==.08


@pytest.mark.parametrize('wrapped', [False, True])
def test_gui_explicit_config_and_output_location(tmp_path, monkeypatch, wrapped):
    # No terrain preset exists, and the config is outside the GUI dataset root.
    root = tmp_path / 'datasets'
    config = tmp_path / 'my configs' / 'mulch.yaml'
    params = {**asdict(DBMParams()), 'C_tire_r': 234.0, 'mass': 13.0}
    write_yaml(config, {'params': params, 'calibration': {'label': 'mulch'}} if wrapped else params)
    launches = []
    monkeypatch.setattr('optcar.datagen.generation_jobs.shutil.which', lambda _: '/test/tmux')
    def fake_tmux(argv, **kwargs):
        if argv[1] == 'new-session':
            launches.append(argv)
        return SimpleNamespace(stdout='', stderr='', returncode=0)
    monkeypatch.setattr('optcar.datagen.generation_jobs.subprocess.run', fake_tmux)
    client = create_app(root).test_client()
    preview = client.get('/api/preset', query_string={'terrain': 'mulch', 'path': str(config)})
    assert preview.status_code == 200 and preview.json['params'] == params
    assert preview.json['path'] == str(config)
    response = client.post('/api/generate', json={'terrain': 'mulch', 'run': 'run_01',
        'preset_path': str(config), 'device': 'cpu'})
    assert response.status_code == 200 and len(launches) == 1
    assert response.json['dataset'] == str(config.parent / 'run_01')
    directory = root / '.generation_jobs' / response.json['job_id']
    snapshot = json.loads((directory / 'request.json').read_text())
    assert snapshot['params'] == params and snapshot['preset_path'] == str(config)
    assert snapshot['calibration'] == ({'label': 'mulch'} if wrapped else None)
    assert not (terrain_path(root, 'mulch') / 'dbm_params.yaml').exists()
    # A bad explicit path must not fall back to a different terrain preset.
    write_yaml(terrain_path(root, 'mulch') / 'dbm_params.yaml', {'params': asdict(DBMParams())})
    missing = str(config.parent / 'missing.yaml')
    assert client.get('/api/preset', query_string={'path': missing}).status_code == 400
    error = client.post('/api/generate', json={'terrain': 'mulch', 'preset_path': missing})
    assert error.status_code == 400 and 'missing.yaml' in error.json['error']
    assert len(launches) == 1


def test_builder_tmux_progress_reconnect_and_separate_dashboards(manager, tmp_path):
    from optcar.dataset import DataConfig
    from test_dataset_builder import raw_fixture, spec
    file,_=raw_fixture(tmp_path/'inputs',environments=3)
    config=DataConfig(history_steps=3,future_steps=2,stride=1,shard_windows=3)
    client=create_app(manager.root).test_client()
    response=client.post('/api/dataset-builder/build',json={'output':str(tmp_path/'output'),
        'sources':[spec(file)],'data':asdict(config),'device':'cpu','workers':3})
    assert response.status_code==200,response.json
    job=response.json
    assert job['tmux_session'].startswith('optcar-build-')
    with pytest.raises(ValueError,match='active job'):
        manager.start_build(tmp_path/'output',[spec(file)],config,device='cpu')
    record=wait_finished(manager,job['job_id'])
    assert record['status']=='complete',record
    assert record['device']=='cpu' and record['workers']==3 and record['result']['new_windows']==24
    snapshot=json.loads((manager.job_dir(job['job_id'])/'request.json').read_text())
    assert snapshot['workers']==3
    assert record['progress']['completed']==record['progress']['total']==24
    assert record['progress']['files_completed']==record['progress']['files_total']==1
    new_client=create_app(manager.root).test_client()
    assert new_client.get('/api/dataset-builder/runs').json['runs'][0]['job_id']==job['job_id']
    assert new_client.get('/api/generation/runs').json['runs']==[]
    pane=subprocess.run(['tmux','capture-pane','-p','-t',record['tmux_session']],capture_output=True,text=True,check=True).stdout
    assert 'Complete' in pane and '24 new windows' in pane and 'File loading workers: 3' in pane


def test_builder_cancellation_and_gpu_validation(manager, tmp_path):
    from test_dataset_builder import raw_fixture, spec
    file,_=raw_fixture(tmp_path/'inputs')
    with pytest.raises(ValueError,match='unavailable'):
        manager.start_build(tmp_path/'invalid',[spec(file)],device='cuda:9999')
    assert not manager.list('dataset_build') and not (tmp_path/'invalid').exists()
    job=manager.start_build(tmp_path/'cancelled',[spec(file)],data={'history_steps':3,'future_steps':2},device='cpu')
    manager.cancel(job['job_id'])
    record=wait_finished(manager,job['job_id'])
    assert record['status']=='cancelled',record
    assert record['result']['completed']==record['result']['new_windows']==0
    assert not list((tmp_path/'cancelled').rglob('*.bin'))


def test_builder_cli_same_tmux_manager(tmp_path, monkeypatch, capsys):
    from optcar.cli import main
    config=tmp_path/'builder.yaml'
    write_yaml(config,{'output':str(tmp_path/'output'),'sources':[{'path':str(tmp_path/'raw'),'terrain':'road','source':'sim'}]})
    called=[]
    def launch(self,output,sources,data,seed,fractions,**kwargs):
        called.append((self.root,output,sources,kwargs))
        return {'tmux_session':'optcar-build-test'}
    monkeypatch.setattr(GenerationJobs,'start_build',launch)
    main(['build-dataset','--config',str(config),'--device','cuda:1','--workers','6','--dataset-root',str(tmp_path/'registry')])
    assert json.loads(capsys.readouterr().out)['tmux_session']=='optcar-build-test'
    assert called[0][0]==tmp_path/'registry' and called[0][3]['device']=='cuda:1' and called[0][3]['workers']==6
