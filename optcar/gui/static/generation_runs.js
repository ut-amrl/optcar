'use strict';
const generationRunCards = new Map();
let generationRunsBusy = false;
const generationRunActive = job => ['queued','running','cancelling'].includes(job.status);
function generationTime(seconds){
  if(!Number.isFinite(seconds))return '—';
  seconds=Math.max(0,Math.round(seconds));
  return seconds>=3600?`${Math.floor(seconds/3600)}h ${Math.floor(seconds%3600/60)}m`:seconds>=60?`${Math.floor(seconds/60)}m ${seconds%60}s`:`${seconds}s`;
}
function generationRunCard(job,report=generationStatus,refresh=refreshGenerationRuns){
  const card=document.createElement('article');card.className='generation-run-card';
  const header=document.createElement('div');header.className='section-heading';
  const title=document.createElement('strong'),badge=document.createElement('span');badge.className='generation-run-badge';header.append(title,badge);
  const session=document.createElement('code');session.className='generation-run-session';
  const device=document.createElement('div');device.className='muted';
  const progress=document.createElement('progress');progress.max=1;
  const detail=document.createElement('div');detail.className='generation-run-detail';
  const timing=document.createElement('div');timing.className='muted';
  const output=document.createElement('div');output.className='generation-run-output';
  const command=document.createElement('code');command.className='generation-run-attach';
  const error=document.createElement('div');error.className='error generation-run-error';
  const actions=document.createElement('div');actions.className='button-row';
  const button=(label,action)=>{const el=document.createElement('button');el.className='secondary';el.textContent=label;el.onclick=async()=>{el.disabled=true;try{await action();}catch(e){report(e.message,true);}finally{el.disabled=el===entry.cancel&&entry.job.status==='cancelling';}};actions.append(el);return el;};
  const copy=button('Copy attach command',async()=>{try{await navigator.clipboard.writeText(entry.job.attach_command);report('Attach command copied.');}catch{report(`Run in a terminal: ${entry.job.attach_command}`);}});
  const cancel=button('Cancel',async()=>{await api(`/api/jobs/${job.job_id}/cancel`,{});await refresh();});
  const view=button('Visualize',()=>openBinFolder(entry.job.dataset,'syn-bins'));
  const build=button('Use in builder',()=>{addDatasetSource(entry.job.dataset,entry.job.terrain,'sim');selectTab('builder');});
  card.append(header,session,device,progress,detail,timing,output,command,error,actions);
  const entry={card,title,badge,session,device,progress,detail,timing,output,command,error,copy,cancel,view,build,job};
  return entry;
}
function updateGenerationRun(entry,job){
  entry.job=job;entry.title.textContent=job.kind==='dataset_build'?'Dataset build':`${job.terrain} / ${job.run}`;
  entry.badge.textContent=job.status;entry.badge.dataset.status=job.status;
  entry.session.textContent=job.tmux_session;
  entry.device.textContent=job.kind==='dataset_build'?`${job.device} · ${job.workers??16} file loading workers`:`${job.device} · ${job.runs} bins × ${job.batch_size} trajectories · ${job.duration}s · dt ${job.dt}s`;
  const p=job.progress||{},fraction=p.total?Math.min(1,p.completed/p.total):0;
  entry.progress.value=fraction;
  entry.detail.textContent=job.status==='complete'?(job.kind==='dataset_build'?`${job.result.completed} bins · ${job.result.new_windows.toLocaleString()} new windows saved`:`${job.result.completed} bins · ${job.result.trajectories} trajectories saved`):p.current||'Waiting for the worker…';
  if(job.kind==='dataset_build'&&generationRunActive(job)&&p.total)entry.detail.textContent+=` · ${(p.new_windows||0).toLocaleString()} new windows · ${p.files_completed||0}/${p.files_total||0} bins`;
  const elapsed=job.started_at?(job.finished_at||Date.now()/1000)-job.started_at:0;
  const eta=generationRunActive(job)&&fraction>0&&fraction<1?generationTime(elapsed*(1-fraction)/fraction):'—';
  entry.timing.textContent=`${Math.round(fraction*100)}% · Elapsed ${generationTime(elapsed)}${generationRunActive(job)?` · Remaining ${eta}`:''} · Created ${new Date(job.created_at*1000).toLocaleString()}`;
  entry.output.textContent=job.dataset;
  entry.command.textContent=job.attach_command;
  entry.error.textContent=job.error||'';entry.error.hidden=!job.error;
  entry.copy.hidden=!job.session_available;
  entry.cancel.hidden=!generationRunActive(job);entry.cancel.disabled=job.status==='cancelling';
  const saved=(job.result?.completed||0)>0;
  entry.view.hidden=job.kind==='dataset_build'||!saved;entry.build.hidden=job.kind==='dataset_build'||!saved;
}
async function refreshGenerationRuns(){
  if(generationRunsBusy)return;
  generationRunsBusy=true;
  try{
    const {runs}=await api('/api/generation/runs'),container=$('generation-jobs');
    const active=runs.filter(generationRunActive).length,complete=runs.filter(job=>job.status==='complete').length;
    $('generation-runs-summary').textContent=`${active} active · ${complete} complete · ${runs.length} total`;
    if(!runs.length){container.textContent='No tmux collections yet.';return;}
    if(!generationRunCards.size)container.replaceChildren();
    let changed=false,position=0;
    for(const job of runs){
      let entry=generationRunCards.get(job.job_id);
      if(!entry){entry=generationRunCard(job);generationRunCards.set(job.job_id,entry);container.append(entry.card);}
      else if(generationRunActive(entry.job)&&!generationRunActive(job))changed=true;
      const current=container.children[position++];if(current!==entry.card)container.insertBefore(entry.card,current||null);
      updateGenerationRun(entry,job);
    }
    if(changed)await refreshDatasets();
  }finally{generationRunsBusy=false;}
}
bind('generation-runs-refresh',refreshGenerationRuns);
refreshGenerationRuns().catch(e=>generationStatus(e.message,true));
setInterval(()=>{if(state.tab==='generate')refreshGenerationRuns().catch(e=>generationStatus(e.message,true));},2000);
