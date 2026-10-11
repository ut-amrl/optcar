"""Real tmux lifecycle checks using tiny disposable CPU collections."""
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import subprocess
import time

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
