'use strict';
const rawState = {scan:null, selected:new Set(), job:null, active:false, defaultOutput:null, completedOutput:null};
const bytes = n => n >= 2**30 ? `${(n/2**30).toFixed(2)} GB` : `${(n/2**20).toFixed(1)} MB`;
const clockText = seconds => seconds == null ? '—' : seconds < 60 ? `${seconds.toFixed(1)}s` : `${Math.floor(seconds/60)}m ${Math.floor(seconds%60)}s`;
function rawConfig(){return {dt:number('raw-dt'),pose_topic:$('raw-pose').value.trim(),twist_topic:$('raw-twist').value.trim(),command_topic:$('raw-command').value.trim(),twist_frame:$('raw-frame').value,command_type:$('raw-command-type').value,action_shift:number('raw-shift'),max_duration:$('raw-duration').value===''?null:number('raw-duration'),include_cmd_vel:$('raw-extra-cmd').checked,aux_twist_topic:$('raw-aux-twist').value.trim()||null};}
function rawIssue(b){if(b.error)return b.error;const cfg=rawConfig();const missing=[cfg.pose_topic,cfg.twist_topic,cfg.command_topic].filter(name=>!b.topics.some(t=>t.topic===name&&t.count>0));return missing.length?`Missing or empty topics: ${missing.join(', ')}`:null;}
function visibleBags(){const filter=$('raw-filter').value.toLowerCase();return (rawState.scan?.bags||[]).filter(b=>b.relative_path.toLowerCase().includes(filter));}
function rawCounts(){const selected=(rawState.scan?.bags||[]).filter(b=>rawState.selected.has(b.path));$('raw-count').textContent=rawState.scan?.bags.length??'—';$('raw-selected').textContent=selected.length;$('raw-time').textContent=selected.reduce((n,b)=>n+(b.duration_s||0),0).toFixed(1);$('raw-size').textContent=bytes(selected.reduce((n,b)=>n+b.bytes,0));$('raw-convert').disabled=rawState.active||!selected.length;}
function rawMetadata(b){$('raw-metadata').hidden=false;$('raw-metadata').open=true;$('raw-metadata-title').textContent=b.relative_path;$('raw-metadata-body').textContent=[`Duration: ${clockText(b.duration_s)}`,b.error||'',...b.topics.map(t=>t.topic)].filter(Boolean).join('\n');}

function renderRawBags(){
  for(const b of rawState.scan?.bags||[])if(rawIssue(b))rawState.selected.delete(b.path);
  const grid=$('raw-bags');grid.replaceChildren();
  for(const b of visibleBags()){
    const issue=rawIssue(b);
    const card=document.createElement('article');card.className='bag-tile';
    card.classList.toggle('bag-invalid',Boolean(issue));
    card.classList.toggle('selected',rawState.selected.has(b.path));
    const heading=document.createElement('label');heading.className='bag-tile-heading';
    const check=document.createElement('input');check.type='checkbox';
    check.checked=rawState.selected.has(b.path);check.disabled=Boolean(issue)||rawState.active;
    check.setAttribute('aria-label',`Use ${b.relative_path}`);
    check.onchange=()=>{check.checked?rawState.selected.add(b.path):rawState.selected.delete(b.path);card.classList.toggle('selected',check.checked);rawCounts();};
    const name=document.createElement('strong');name.textContent=b.path.split('/').pop();
    heading.append(name,check);card.append(heading);
    const folder=document.createElement('p');folder.className='bag-tile-folder';
    folder.textContent=b.relative_path.includes('/')?b.relative_path.slice(0,b.relative_path.lastIndexOf('/')):'Source folder';
    folder.title=b.path;card.append(folder);
    const duration=document.createElement('p');duration.className='bag-tile-duration';duration.textContent=clockText(b.duration_s);card.append(duration);
    const topics=document.createElement('ul');topics.className='bag-tile-topics';
    const cfg=rawConfig();const relevant=[...new Set([cfg.pose_topic,cfg.twist_topic,cfg.command_topic])];
    const names=relevant.filter(topic=>b.topics.some(t=>t.topic===topic));
    for(const topic of names){const item=document.createElement('li');item.textContent=topic;item.title=topic;topics.append(item);}
    card.append(topics);
    const footer=document.createElement('div');footer.className='bag-tile-footer';
    const summary=document.createElement('span');summary.textContent=issue?'Needs attention':'';summary.title=issue||'';
    const button=document.createElement('button');button.className='bag-metadata-button';button.textContent='All topics';
    button.setAttribute('aria-label',`View topics for ${b.relative_path}`);button.onclick=()=>rawMetadata(b);
    footer.append(summary,button);card.append(footer);grid.append(card);
  }
  if(!grid.children.length){const empty=document.createElement('p');empty.className='raw-empty';empty.textContent=rawState.scan?'No recordings match this filter.':'Scan a folder to see recordings.';grid.append(empty);}
  rawCounts();
}

function lockRaw(active){rawState.active=active;for(const element of document.querySelectorAll('#raw-workspace aside input, #raw-workspace aside select, #raw-scan, #raw-all, #raw-none'))element.disabled=active;renderRawBags();}
function rawResults(rows){const container=$('raw-results');container.replaceChildren();for(const r of rows){const row=document.createElement('div');row.className=`raw-result ${r.status}`;const name=document.createElement('strong');name.textContent=`${r.status==='complete'?'✓':'!'} ${r.bag}`;const detail=document.createElement('span');detail.textContent=r.error||`${r.samples.toLocaleString()} samples → ${r.output}${r.warning?`\n${r.warning}`:''}`;row.append(name,detail);container.append(row);}}
function showRawProgress(job){const p=job.progress||{};const terminal=!['queued','running'].includes(job.status);const percent=Math.min(100,Math.max(0,p.percent||0));$('raw-progress').hidden=false;$('raw-job-title').textContent=terminal?({complete:'Conversion complete',partial:'Finished with failed bags',failed:'Conversion failed',cancelled:'Conversion cancelled'}[job.status]||job.status):`Bag ${p.bag_index||0} of ${p.total_bags||0} · ${p.phase||job.status}`;$('raw-progress-bar').style.width=`${percent}%`;$('raw-percent').textContent=`${percent.toFixed(1)}%`;$('raw-current').textContent=p.bag||'Preparing recordings…';$('raw-handled').textContent=`${p.completed_bags||0} / ${p.total_bags||0}`;$('raw-messages').textContent=`${(p.messages||0).toLocaleString()} / ${(p.messages_total||0).toLocaleString()}`;$('raw-elapsed').textContent=clockText(job.result?.elapsed_s??p.elapsed_s);$('raw-eta').textContent=terminal?'—':clockText(p.eta_s);$('raw-bag-progress').style.width=`${100*(p.bag_fraction||0)}%`;$('raw-progress-output').textContent=`Output: ${job.result?.output||p.output||$('raw-output').value}`;$('raw-cancel').hidden=terminal;rawState.completedOutput=terminal&&job.result?.converted>0?job.result.output:null;$('raw-view-bins').hidden=!rawState.completedOutput;rawResults(job.result?.results||p.results||[]);if(terminal){localStorage.setItem('optcar-raw-result',JSON.stringify(job));if(job.error)status(job.error,true);else{const r=job.result;status(`${r.converted} bags saved · ${r.failed} failed${job.status==='cancelled'?' · cancelled; completed bins retained':''}. Output: ${r.output}`,Boolean(r.failed));}}}
async function pollRawJob(){try{const job=await api(`/api/jobs/${rawState.job}`);showRawProgress(job);if(['queued','running'].includes(job.status)){setTimeout(pollRawJob,500);}else{localStorage.removeItem('optcar-raw-job');lockRaw(false);}}catch(error){status(`Unable to read conversion progress: ${error.message}`,true);$('raw-job-title').textContent='Progress connection interrupted';lockRaw(false);}}
bind('raw-scan',async()=>{status('Scanning bag folders and metadata…');const scan=await api('/api/raw-bags/scan',{root:$('raw-root').value.trim()});rawState.scan=scan;rawState.selected.clear();$('raw-root').value=scan.root;if(!$('raw-output').value.trim()||$('raw-output').value===rawState.defaultOutput)$('raw-output').value=scan.default_output;rawState.defaultOutput=scan.default_output;rawOutputPreview();$('raw-metadata').hidden=true;for(const b of scan.bags)if(!rawIssue(b))rawState.selected.add(b.path);renderRawBags();const issues=scan.bags.filter(rawIssue).length;status(`${scan.bags.length} recordings found · ${rawState.selected.size} selected${issues?` · ${issues} need attention`:''}. Review your selection, then convert.`);});
bind('raw-all',()=>{for(const b of visibleBags())if(!rawIssue(b))rawState.selected.add(b.path);renderRawBags();});
bind('raw-none',()=>{for(const b of visibleBags())rawState.selected.delete(b.path);renderRawBags();});
$('raw-filter').addEventListener('input',renderRawBags);
for(const id of ['raw-pose','raw-twist','raw-command'])$(id).addEventListener('change',()=>{try{renderRawBags();}catch(error){status(error.message,true);}});
$('raw-root').addEventListener('input',()=>{rawState.scan=null;rawState.selected.clear();renderRawBags();});
bind('raw-convert',async()=>{if(!rawState.scan||!rawState.selected.size)throw new Error('Scan a folder and select bags first.');const result=await api('/api/raw-bags/convert',{root:rawState.scan.root,bags:[...rawState.selected],output:$('raw-output').value.trim(),config:rawConfig(),overwrite:$('raw-overwrite').checked});rawState.job=result.job_id;localStorage.setItem('optcar-raw-job',result.job_id);lockRaw(true);showRawProgress({status:'queued',progress:{output:$('raw-output').value,total_bags:rawState.selected.size}});$('raw-progress').scrollIntoView({behavior:'smooth',block:'nearest'});status('Conversion started. Completed bins are saved as each bag finishes.');pollRawJob();});
bind('raw-cancel',async()=>{await api(`/api/jobs/${rawState.job}/cancel`,{});status('Cancelling conversion. Completed bins will be retained.');});
const previousRawJob=localStorage.getItem('optcar-raw-job');if(previousRawJob){rawState.job=previousRawJob;lockRaw(true);pollRawJob();}else{try{const previousResult=JSON.parse(localStorage.getItem('optcar-raw-result')||'null');if(previousResult)showRawProgress(previousResult);}catch{localStorage.removeItem('optcar-raw-result');}}

function rawOutputPreview(){if(rawState.scan)$('raw-output-preview').textContent=`Output tree: ${$('raw-output').value.trim()||rawState.scan.default_output}/data/${rawState.scan.root.split('/').pop()}/…`;}
$('raw-output').addEventListener('input',rawOutputPreview);

bind('raw-view-bins',()=>openBinFolder(rawState.completedOutput));
