'use strict';
const $ = id => document.getElementById(id);
const number = id => {const value = Number($(id).value); if (!Number.isFinite(value)) throw new Error(`Invalid ${id}`); return value;};
const state = {tab:'raw', session:null, segment:null, prediction:null, calibration:null, params:{}, defaults:null, playing:false, zoom:1, pan:[0,0]};
const names = {len_f:'Front axle distance (m)',len_r:'Rear axle distance (m)',mass:'Mass (kg)',inertia:'Yaw inertia (kg m²)',max_steering_angle:'Steering limit (rad)',C_tire_f:'Front tire stiffness',C_tire_r:'Rear tire stiffness',C_throttle_a:'Drive coefficient',C_rollf:'Rolling resistance',C_drag:'Drag coefficient',wheel_time_constant:'Wheel response (s)',min_wheel_velocity:'Min wheel speed (m/s)',max_wheel_velocity:'Max wheel speed (m/s)'};
function status(text, error=false){if(state.tab==='dataset-viewer'){datasetViewerStatus(text,error);return;}if(state.tab==='builder'){builderStatus(text,error);return;}if(state.tab==='generate'){generationStatus(text,error);return;}for(const id of ['status','raw-status','bin-status']){$(id).textContent=text;$(id).classList.toggle('error',error);}}
async function api(path, body){const response=await fetch(path,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const result=await response.json();if(!response.ok)throw new Error(result.error||response.statusText);return result;}
function bind(id, action){$(id).addEventListener('click',async()=>{const button=$(id);button.disabled=true;try{await action();}catch(error){status(error.message,true);}finally{button.disabled=id==='raw-convert'?rawState.active||!rawState.selected.size:false;}});}
async function refreshDevices(initial=false){
  try{
    const result=await api('/api/devices');
    for(const id of ['device','predict-device','builder-device']){
      const select=$(id),previous=select.value;select.replaceChildren();
      for(const device of result.devices){
        const option=document.createElement('option');option.value=device.id;
        option.textContent=device.id==='cpu'?'CPU':`${device.id} · ${device.name} · ${device.memory_gib} GiB`;
        select.append(option);
      }
      select.value=initial?result.default:result.devices.some(d=>d.id===previous)?previous:result.default;
    }
    const count=result.devices.length-1;
    for(const [id,info] of [['device','device-info'],['builder-device','builder-device-info']]){
      const showDevice=()=>{
        const selected=result.devices.find(d=>d.id===$(id).value);
        $(info).textContent=count?`${count} GPU${count===1?'':'s'} available · ${selected.id==='cpu'?'CPU selected':selected.name+' · '+selected.memory_gib+' GiB'}`:result.cuda_version?'No GPUs available. CPU is available.':'CPU-only Torch detected. Run setup and restart the GUI to enable GPUs.';
      };
      $(id).onchange=showDevice;showDevice();$(info).classList.toggle('error',!count);
    }
  }catch(error){$('device-info').textContent=`GPU detection failed: ${error.message}`;$('device-info').classList.add('error');throw error;}
}
function payload(){if(!state.session)throw new Error('Load a recording first.');return {session_id:state.session.session_id,segment:Number($('segment').value||0)};}
function dataConfig(dtId='builder-dt'){return {history_steps:number('history-steps'),future_steps:number('future-steps'),dt:number(dtId),stride:number('stride'),shard_windows:number('builder-shard-windows')};}
function params(){return Object.fromEntries(Object.keys(state.params).map(key=>[key,number(`param-${key}`)]));}
function setParams(values){state.params=values;$('params').replaceChildren();for(const [key,value]of Object.entries(values)){const label=document.createElement('label');label.textContent=names[key]||key;const input=document.createElement('input');input.type='number';input.id=`param-${key}`;input.value=value;input.step=key==='inertia'?'0.001':key.includes('len')||key.includes('time')?'0.01':'0.1';label.append(input);$('params').append(label);}}
function selectTab(tab){state.tab=tab;$('dataset-viewer-workspace').hidden=tab!=='dataset-viewer';$('tab-dataset-viewer').classList.toggle('active',tab==='dataset-viewer');if(tab==='dataset-viewer'&&typeof drawDatasetViewer==='function')requestAnimationFrame(drawDatasetViewer);$('builder-workspace').hidden=tab!=='builder';$('tab-builder').classList.toggle('active',tab==='builder');$('raw-workspace').hidden=tab!=='raw';$('dynamics-workspace').hidden=tab!=='model';$('generation-workspace').hidden=tab!=='generate';$('dbm-workspace').hidden=tab!=='dbm';if(tab==='dbm'&&!$('dbm-tool').getAttribute('src'))$('dbm-tool').src='/dbm/?path='+encodeURIComponent(typeof binState!=='undefined'&&binState.data?binState.data.path:$('bin-root').value)+'&folder='+encodeURIComponent($('bin-root').value);const binViewer=['bins','syn-bins'].includes(tab);$('bins-workspace').hidden=!binViewer;$('tab-syn-bins').classList.toggle('active',tab==='syn-bins');if(binViewer&&typeof switchBinViewer==='function')switchBinViewer(tab).catch(e=>status(e.message,true));$('tab-bins').classList.toggle('active',tab==='bins');state.playing=false;$('play').textContent='▶';if(typeof stopBinPlayback==='function')stopBinPlayback();if(binViewer&&typeof drawBins==='function')requestAnimationFrame(drawBins);$('tab-raw').classList.toggle('active',tab==='raw');state.prediction=null;$('dbm-controls').hidden=tab!=='dbm';$('dbm-return').hidden=tab!=='dbm';$('model-controls').hidden=tab!=='model';$('tab-dbm').classList.toggle('active',tab==='dbm');$('tab-model').classList.toggle('active',tab==='model');$('tab-generate').classList.toggle('active',tab==='generate');if(tab==='builder'&&typeof refreshBuilderRuns==='function')refreshBuilderRuns().catch(e=>builderStatus(e.message,true));if(tab==='generate'){updateGenerationSummary();if(typeof refreshGenerationRuns==='function')refreshGenerationRuns().catch(e=>generationStatus(e.message,true));}$('view-title').textContent=tab==='dbm'?'DBM calibration':'Transformer evaluation';clearMetrics();drawAll();}
function clearMetrics(){for(const id of ['metric-position','metric-heading','metric-final'])$(id).textContent='—';}
function showPrediction(result){state.prediction=result;$('metric-position').textContent=result.metrics.position_rmse_m.toFixed(3);$('metric-heading').textContent=(result.metrics.heading_mae_rad*180/Math.PI).toFixed(2);$('metric-final').textContent=result.metrics.final_position_error_m.toFixed(3);drawAll();status(`Predicted ${result.horizon} steps from sample ${result.current}.`);}
async function preview(){const request={...payload(),current:number('current'),horizon:number('horizon')};status('Computing trajectory…');if(state.tab==='dbm'){request.params=params();if($('wheel-speed').value!=='')request.wheel_speed=number('wheel-speed');showPrediction(await api('/api/rollout',request));}else{request.checkpoint=$('checkpoint').value;request.device=$('predict-device').value;showPrediction(await api('/api/predict',request));}}
function setFrame(value){if(!state.segment)return;const current=Math.max(0,Math.min(state.segment.samples-2,Math.round(value)));$('current').value=current;$('frame').value=current;state.prediction=null;clearMetrics();drawAll();}
async function loadSegment(){state.segment=await api('/api/segment',payload());state.prediction=null;state.zoom=1;state.pan=[0,0];$('empty-view').hidden=true;$('frame').max=state.segment.samples-2;$('current').max=state.segment.samples-2;$('fit-end').value=state.segment.samples-1;$('metric-duration').textContent=((state.segment.samples-1)*state.segment.dt).toFixed(1);setFrame(Math.min(number('current'),state.segment.samples-2));}
async function refreshDatasets(){const result=await api('/api/datasets');const tree=$('dataset-tree');tree.replaceChildren();if(!result.datasets.length){tree.textContent='No generated datasets.';return;}const grouped={};for(const dataset of result.datasets)(grouped[dataset.terrain]??=[]).push(dataset);for(const [terrain,datasets]of Object.entries(grouped)){const entry=document.createElement('div');entry.className='terrain-entry';const title=document.createElement('strong');title.textContent=`syn_data / terrains / ${terrain}`;entry.append(title);for(const d of datasets){const row=document.createElement('div');row.className='dataset-row';const label=document.createElement('span');label.textContent=`${d.source} / ${d.path.split('/').pop()}`;const counts=document.createElement('small');counts.textContent=`${d.files.toLocaleString()} files · ${d.episodes.toLocaleString()} trajectories · ${d.stage==='raw'?'raw bins':Object.values(d.windows).reduce((a,b)=>a+b,0).toLocaleString()+' windows'}`;const use=document.createElement('button');use.className='secondary';use.textContent='Use in builder';use.onclick=()=>{addDatasetSource(state.defaults.dataset_root+'/'+d.path,d.terrain,d.source);selectTab('builder');};row.append(label,counts);if(d.source==='sim'){const view=document.createElement('button');view.className='secondary';view.textContent='Visualize';view.onclick=()=>openBinFolder(state.defaults.dataset_root+'/'+d.path,'syn-bins').catch(e=>status(e.message,true));row.append(view);}row.append(use);entry.append(row);}tree.append(entry);}}
async function trackJob(result,title,onComplete,target='jobs',onFinished){
  const row=document.createElement('div');row.className='job';
  const info=document.createElement('div'),heading=document.createElement('strong'),detail=document.createElement('pre'),bar=document.createElement('progress');
  heading.textContent=title;bar.max=1;info.append(heading,detail,bar);
  const cancel=document.createElement('button');cancel.className='secondary';cancel.textContent='Cancel';
  const report=(text,error=false)=>target==='generation-jobs'?generationStatus(text,error):target==='builder-jobs'?builderStatus(text,error):status(text,error);
  cancel.onclick=()=>api(`/api/jobs/${result.job_id}/cancel`,{}).then(()=>{cancel.disabled=true;cancel.textContent='Cancelling…';}).catch(e=>report(e.message,true));
  row.append(info,cancel);const container=$(target);if(!container.querySelector('.job'))container.replaceChildren();container.prepend(row);
  const poll=async()=>{
    try{
      const job=await api(`/api/jobs/${result.job_id}`),p=job.progress;
      if(p.total){bar.value=p.completed/p.total;detail.textContent=`${job.status} · ${p.completed.toLocaleString()} / ${p.total.toLocaleString()}${p.new_windows===undefined?'':' · '+p.new_windows.toLocaleString()+' new windows'}${p.current?'\n'+p.current:''}`;}
      else{bar.removeAttribute('value');detail.textContent=`${job.status} · ${p.evaluations===undefined?(p.current||'Starting…'):p.evaluations+' evaluations'}`;}
      if(['queued','running'].includes(job.status)){setTimeout(poll,1000);return;}
      cancel.remove();bar.value=job.status==='complete'?1:bar.value;
      if(job.status==='failed'){detail.textContent=job.error;report(job.error,true);}
      else{
        const done=job.result;detail.textContent=`${job.status}${done.total?' · '+done.completed+' / '+done.total:''}${done.new_windows===undefined?'':' · '+done.new_windows.toLocaleString()+' new windows'}${done.dataset?'\n'+done.dataset:''}${done.skipped?.length?'\n'+done.skipped.join('\n'):''}`;
        if(target==='jobs')detail.textContent=JSON.stringify(done,null,2);
        if(job.status==='complete'&&onComplete)await onComplete(done);
        else report(`Job ${job.status}. Completed files are retained.`);
        await refreshDatasets();
      }
    }catch(error){detail.textContent=error.message;cancel.remove();report(error.message,true);}
    if(onFinished)onFinished();
  };poll();
}

bind('tab-dataset-viewer',()=>selectTab('dataset-viewer'));bind('tab-raw',()=>selectTab('raw'));bind('tab-dbm',()=>selectTab('dbm'));bind('tab-model',()=>selectTab('model'));bind('tab-generate',()=>selectTab('generate'));bind('tab-builder',()=>selectTab('builder'));bind('tab-syn-bins',()=>selectTab('syn-bins'));
bind('inspect-topics',async()=>{status('Reading recording topics…');const result=await api('/api/topics',{path:$('bag-path').value});$('topics').hidden=false;$('topics').textContent=result.topics.map(t=>`${t.topic} · ${t.type} (${t.count})`).join('\n');$('topic-options').replaceChildren();for(const topic of result.topics){const option=document.createElement('option');option.value=topic.topic;$('topic-options').append(option);}status('Choose pose, twist, and velocity/steering topics, then load the recording.');});
bind('load-bag',async()=>{status('Loading and aligning recording…');state.playing=false;$('play').textContent='▶';const bag={pose_topic:$('pose-topic').value,twist_topic:$('twist-topic').value,command_topic:$('command-topic').value,twist_frame:$('twist-frame').value,command_type:$('command-type').value,command_shift_s:number('command-shift'),max_gap_s:number('max-gap'),timestamp_source:$('timestamp-source').value};state.session=await api('/api/bag',{path:$('bag-path').value,bag,dt:number('dt'),split:$('split').value});$('segment').replaceChildren();for(const segment of state.session.segments){const option=document.createElement('option');option.value=segment.index;option.textContent=`${segment.index+1} · ${segment.start_s.toFixed(1)}–${segment.end_s.toFixed(1)} s · ${segment.samples} samples`;$('segment').append(option);}$('segment-label').hidden=false;$('bag-state').textContent=`${state.session.segments.length} segment(s) · ${state.session.split}`;await loadSegment();status('Recording loaded.');});
$('segment').addEventListener('change',()=>loadSegment().catch(e=>status(e.message,true)));
function dbmConfigLocation(){const folder=$('dbm-config-folder').value.trim(),filename=$('dbm-config-filename').value.trim();if(!folder||!filename)throw new Error('Enter both the DBM config folder and filename.');return {folder,filename};}
bind('load-preset',async()=>{const result=await api(`/api/preset?${new URLSearchParams(dbmConfigLocation())}`);setParams(result.params);state.calibration=result.calibration||null;status(`Loaded ${result.path}`);});
bind('save-preset',async()=>{const result=await api('/api/preset',{...dbmConfigLocation(),params:params(),calibration:state.calibration,...(state.session?payload():{})});status(`Saved ${result.path}`);});
bind('run-dbm',preview);bind('predict',preview);bind('preview',preview);
bind('fit',async()=>{const keys=[...document.querySelectorAll('#fit-keys input:checked')].map(el=>el.value);if(!keys.length)throw new Error('Select at least one parameter to fit.');const result=await api('/api/fit',{...payload(),params:params(),start:number('fit-start'),end:number('fit-end'),keys});trackJob(result,'DBM identification',async fitted=>{setParams(fitted.params);state.calibration=fitted.calibration;status(`Fit ${fitted.success?'converged':'finished without convergence'}: ${fitted.initial_rmse.toFixed(4)} → ${fitted.fitted_rmse.toFixed(4)}. Preview a separate validation recording, then save the terrain preset.`);});});
bind('export',async()=>{const result=await api('/api/export',{...payload(),terrain:$('terrain').value,run:$('export-run').value,data:dataConfig('dt')});trackJob(result,'Export real recording',async()=>status('Recording exported.'));});
bind('refresh-devices',()=>refreshDevices());
bind('generate',async()=>{const device=$('device').value;const commands={...state.defaults.commands,duration:number('duration'),action_bounds:[[number('velocity-min'),number('velocity-max')],[number('steering-min'),number('steering-max')]],noise_std:[number('velocity-noise'),number('steering-noise')],seed:number('seed'),parameter_randomization:0,parameter_ranges:readGenerationRanges(),bezier_full_range:$('generation-bezier-stretch').checked,resample_params_each_run:$('generation-resample').checked};const terrain=$('generation-terrain').value;const result=await api('/api/generate',{terrain,run:$('run-name').value,data:{dt:number('generation-dt')},raw_only:true,commands,runs:number('episodes'),batch_size:number('batch-size'),device,preset_path:$('generation-preset-path').value.trim()});await refreshGenerationRuns();generationStatus(`Started ${result.tmux_session} on ${device}.`);});
bind('inspect-checkpoint',async()=>{status('Loading checkpoint…');const result=await api('/api/checkpoint',{path:$('checkpoint').value});$('checkpoint-info').textContent=`${result.model.architecture} · ${result.stage}\nHistory ${result.data.history_steps} · horizon ${result.data.future_steps}\nTimestep ${result.data.dt} s · epoch ${result.epoch+1}`;$('horizon').value=result.data.future_steps;$('current').value=Math.max(number('current'),result.data.history_steps-1);if(state.segment)setFrame(number('current'));status(`Checkpoint loaded.${state.session&&Math.abs(state.session.dt-result.data.dt)>1e-9?' Reload recording at '+result.data.dt+' s timestep.':''}`);});
bind('refresh-data',refreshDatasets);bind('reset-view',()=>{state.zoom=1;state.pan=[0,0];drawAll();});
$('frame').addEventListener('input',()=>setFrame(number('frame')));$('current').addEventListener('change',()=>setFrame(number('current')));
bind('play',()=>{state.playing=!state.playing;$('play').textContent=state.playing?'Ⅱ':'▶';});
setInterval(()=>{if(state.playing&&state.segment){const next=number('current')+Math.max(1,Math.round(0.08/state.segment.dt));setFrame(next);if(next>=state.segment.samples-2){state.playing=false;$('play').textContent='▶';}}},80);
document.addEventListener('keydown',event=>{if(['INPUT','SELECT','TEXTAREA','BUTTON'].includes(event.target.tagName))return;if(!['dbm','model'].includes(state.tab))return;if(event.code==='Space'){event.preventDefault();$('play').click();}if(event.code==='ArrowRight')setFrame(number('current')+1);if(event.code==='ArrowLeft')setFrame(number('current')-1);});

function canvas(id){const element=$(id);const rect=element.getBoundingClientRect();const ratio=window.devicePixelRatio||1;const w=rect.width,h=rect.height;if(element.width!==Math.round(w*ratio)||element.height!==Math.round(h*ratio)){element.width=Math.round(w*ratio);element.height=Math.round(h*ratio);}const ctx=element.getContext('2d');ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,w,h);return {ctx,w,h};}
function line(ctx,points,color,width=1.5){if(!points.length)return;ctx.beginPath();ctx.moveTo(...points[0]);for(let i=1;i<points.length;i++)ctx.lineTo(...points[i]);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke();}
function drawTrajectory(){const {ctx,w,h}=canvas('trajectory');const recording=state.segment?.states||[];if(!recording.length){ctx.strokeStyle='#323840';for(let x=30;x<w;x+=40){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,h);ctx.stroke();}for(let y=20;y<h;y+=40){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke();}return;}const all=[...recording,...(state.prediction?.predicted||[])];let minX=Infinity,maxX=-Infinity,minY=Infinity,maxY=-Infinity;for(const p of all){minX=Math.min(minX,p[0]);maxX=Math.max(maxX,p[0]);minY=Math.min(minY,p[1]);maxY=Math.max(maxY,p[1]);}const scale=Math.min((w-100)/Math.max(1,maxX-minX),(h-75)/Math.max(1,maxY-minY))*state.zoom;const cx=(minX+maxX)/2,cy=(minY+maxY)/2;const map=p=>[w/2+(p[0]-cx)*scale+state.pan[0],h/2-(p[1]-cy)*scale+state.pan[1]];const spacing=Math.pow(10,Math.floor(Math.log10(80/scale)));const grid=spacing*(80/scale/spacing>5?5:80/scale/spacing>2?2:1);ctx.font='12px system-ui, sans-serif';ctx.fillStyle='#b0bac5';ctx.strokeStyle='#323840';const xlo=cx+(-w/2-state.pan[0])/scale,xhi=cx+(w/2-state.pan[0])/scale;const ylo=cy+(-h/2+state.pan[1])/scale,yhi=cy+(h/2+state.pan[1])/scale;for(let x=Math.ceil(xlo/grid)*grid;x<=xhi;x+=grid){const p=map([x,0]);ctx.beginPath();ctx.moveTo(p[0],0);ctx.lineTo(p[0],h);ctx.stroke();ctx.fillText(x.toFixed(grid<1?1:0),p[0]+4,h-10);}for(let y=Math.ceil(ylo/grid)*grid;y<=yhi;y+=grid){const p=map([0,y]);ctx.beginPath();ctx.moveTo(0,p[1]);ctx.lineTo(w,p[1]);ctx.stroke();ctx.fillText(y.toFixed(grid<1?1:0),8,p[1]-5);}line(ctx,recording.map(map),'#71818e',1.5);if(state.prediction){line(ctx,state.prediction.measured.map(map),'#e7b17a',3);line(ctx,state.prediction.predicted.map(map),'#8bc99b',2.5);}const current=number('current');let idx=0;while(idx+1<state.segment.indices.length&&state.segment.indices[idx+1]<=current)idx++;const p=map(recording[idx]);const yaw=state.segment.yaw[idx];ctx.save();ctx.translate(...p);ctx.rotate(-yaw);ctx.fillStyle='#80cbd3';ctx.beginPath();ctx.moveTo(11,0);ctx.lineTo(-7,-5);ctx.lineTo(-4,0);ctx.lineTo(-7,5);ctx.closePath();ctx.fill();ctx.restore();$('frame-info').textContent=`Sample ${current} · ${(current*state.segment.dt).toFixed(2)} s`;}
function drawSignal(id, series){const {ctx,w,h}=canvas(id);if(!state.segment)return;const {times}=state.segment;const left=60,right=w-20,top=18,bottom=h-30;let lo=0,hi=0;for(const values of series)for(const value of values){lo=Math.min(lo,value);hi=Math.max(hi,value);}const span=Math.max(0.1,hi-lo);lo-=span*.1;hi+=span*.1;const map=(i,value)=>[left+(times[i]-times[0])/Math.max(.01,times[times.length-1]-times[0])*(right-left),bottom-(value-lo)/(hi-lo)*(bottom-top)];ctx.font='12px system-ui, sans-serif';ctx.fillStyle='#b0bac5';for(let i=0;i<=3;i++){const value=lo+(hi-lo)*i/3;const y=map(0,value)[1];ctx.strokeStyle='#323840';ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(right,y);ctx.stroke();ctx.textAlign='right';ctx.fillText(value.toFixed(1),left-8,y+4);ctx.textAlign='left';}series.forEach((values,i)=>line(ctx,values.map((value,j)=>map(j,value)),i===0?'#80cbd3':'#e7b17a',1.4));const cursor=left+(number('current')*state.segment.dt)/(state.segment.dt*(state.segment.samples-1))*(right-left);ctx.strokeStyle='#8bc99b';ctx.setLineDash([3,3]);ctx.beginPath();ctx.moveTo(cursor,top);ctx.lineTo(cursor,bottom);ctx.stroke();ctx.setLineDash([]);ctx.fillText(`${times[0].toFixed(1)}s`,left,h-7);ctx.textAlign='right';ctx.fillText(`${times[times.length-1].toFixed(1)}s`,right,h-7);ctx.textAlign='left';}
function drawAll(){drawTrajectory();if(state.segment){drawSignal('velocity',[state.segment.states.map((s,i)=>s[7]*Math.cos(state.segment.yaw[i])+s[8]*Math.sin(state.segment.yaw[i])),state.segment.commands.map(a=>a[0])]);drawSignal('steering',[state.segment.commands.map(a=>a[1]),state.segment.states.map(s=>s[12])]);}else{canvas('velocity');canvas('steering');}}
const plot=$('trajectory');plot.addEventListener('wheel',event=>{event.preventDefault();state.zoom=Math.max(.1,Math.min(30,state.zoom*Math.exp(-event.deltaY*.001)));drawTrajectory();},{passive:false});let drag=null;plot.addEventListener('pointerdown',event=>{drag=[event.clientX,event.clientY,...state.pan];plot.setPointerCapture(event.pointerId);});plot.addEventListener('pointermove',event=>{if(drag){state.pan=[drag[2]+event.clientX-drag[0],drag[3]+event.clientY-drag[1]];drawTrajectory();}});plot.addEventListener('pointerup',()=>drag=null);plot.addEventListener('pointercancel',()=>drag=null);new ResizeObserver(drawAll).observe($('trajectory'));
async function init(){state.defaults=await api('/api/defaults');setParams(state.defaults.params);$('generation-resample').checked=Boolean(state.defaults.commands.resample_params_each_run);$('generation-bezier-stretch').checked=state.defaults.commands.bezier_full_range;try{state.generationPreset=(await api('/api/preset?terrain='+encodeURIComponent($('generation-terrain').value.trim()))).params;}catch{state.generationPreset=null;}renderGenerationRanges(state.generationPreset||state.defaults.params,state.defaults.commands.parameter_ranges);$('raw-root').value=state.defaults.raw_root||'';$('builder-output').value=state.defaults.dataset_root+'/built/default';updateGenerationSummary();for(const key of ['len_f','len_r','mass','inertia',...state.defaults.fit_keys]){const label=document.createElement('label');const check=document.createElement('input');check.type='checkbox';check.value=key;check.checked=state.defaults.fit_keys.includes(key);label.append(check,document.createTextNode(key));$('fit-keys').append(label);}await refreshDatasets();await refreshDevices(true);drawAll();}
init().catch(error=>status(error.message,true));

bind('dbm-manual',()=>{state.playing=false;$('play').textContent='▶';$('dbm-workspace').hidden=false;$('dynamics-workspace').hidden=true;});
bind('dbm-advanced',()=>{$('dbm-workspace').hidden=true;$('dynamics-workspace').hidden=false;drawAll();});
bind('dbm-return',()=>{state.playing=false;$('play').textContent='▶';$('dbm-workspace').hidden=false;$('dynamics-workspace').hidden=true;});
function sizeDbmWorkspace(){const headerHeight=document.querySelector('header').getBoundingClientRect().height;document.documentElement.style.setProperty('--workbench-header-height',`${headerHeight}px`);const height=Math.max(200,window.innerHeight-headerHeight);$('dbm-workspace').style.height=`${height}px`;$('generation-workspace').style.height=`${height}px`;$('builder-workspace').style.height=`${height}px`;$('dataset-viewer-workspace').style.height=`${height}px`;}
new ResizeObserver(sizeDbmWorkspace).observe(document.querySelector('header'));
window.addEventListener('resize',sizeDbmWorkspace);
sizeDbmWorkspace();

function generationStatus(text,error=false){$('generation-status').textContent=text;$('generation-status').classList.toggle('error',error);}
function updateGenerationSummary(){
  $('generation-episodes').textContent=Number($('episodes').value).toLocaleString()+' / '+(Number($('episodes').value)*Number($('batch-size').value)).toLocaleString();
  $('generation-duration').textContent=`${$('duration').value}s`;
  const dt=Number($('generation-dt').value);
  $('generation-steps').textContent=Math.round(Number($('duration').value)/dt).toLocaleString();
  $('generation-timestep').textContent=`${dt}s`;
  const terrain=$('generation-terrain').value.trim(),run=$('run-name').value.trim();
  const config=$('generation-preset-path').value.trim(),slash=config.lastIndexOf('/');
  const folder=slash<0?'.':config.slice(0,slash);
  $('generation-output').textContent=config?`Output: ${folder}/${run}/`:`Output: ${state.defaults?.dataset_root||'datasets'}/syn_data/terrains/${terrain}/sim/${run}/`;
}
for(const id of ['episodes','batch-size','duration','generation-dt','generation-terrain','run-name'])$(id).addEventListener('input',updateGenerationSummary);
for(const id of ['generation-terrain','generation-preset-path'])$(id).addEventListener('input',()=>{state.generationPreset=null;$('generation-preset-details').hidden=true;$('generation-preset-status').textContent=$('generation-preset-path').value.trim()?'Config not loaded.':'Using the saved terrain preset.';updateGenerationSummary();});
bind('generation-load-preset',async()=>{
  $('generation-preset-details').hidden=true;
  const terrain=$('generation-terrain').value.trim(),path=$('generation-preset-path').value.trim();
  const query=new URLSearchParams({terrain});if(path)query.set('path',path);
  try{
    const result=await api(`/api/preset?${query}`);
    if(path)$('generation-preset-path').value=result.path;
    updateGenerationSummary();
    $('generation-preset-values').replaceChildren();
    for(const [key,value]of Object.entries(result.params)){const term=document.createElement('dt');term.textContent=names[key]||key;const definition=document.createElement('dd');definition.textContent=value;$('generation-preset-values').append(term,definition);}
    state.generationPreset=result.params;renderGenerationRanges(result.params,readGenerationRanges());
    $('generation-preset-details').hidden=false;$('generation-preset-status').textContent=`Loaded ${result.path}`;
  }catch(error){$('generation-preset-status').textContent=`Could not load config: ${error.message}`;throw error;}
});

bind('generation-view',()=>state.generated&&openBinFolder(state.generated.path,'syn-bins'));
bind('generation-build',()=>{if(state.generated){addDatasetSource(state.generated.path,state.generated.terrain,'sim');selectTab('builder');}});


// The backend catalog and WheeledLab sampler define these parameters once.
const generationRangeLabels={start_pos_x:'Starting x (m)',start_pos_y:'Starting y (m)',start_pos_z:'Starting z (m)',start_rot_yaw:'Starting heading (rad)',max_velocity:'Velocity normalization limit (m/s)',max_steering:'Steering limit (rad)',max_lateral_acceleration_mps2:'Lateral acceleration limit (m/s²)',throttle_multiplier:'Throttle gain',throttle_bias:'Throttle bias (m/s)',steer_pos_multiplier:'Positive steering gain',steer_pos_bias:'Positive steering bias (rad)',steer_neg_multiplier:'Negative steering gain',steer_neg_bias:'Negative steering bias (rad)',throttle_deadband_min:'Throttle deadband lower (m/s)',throttle_deadband_max:'Throttle deadband upper (m/s)',steer_deadband_min:'Steering deadband lower (rad)',steer_deadband_max:'Steering deadband upper (rad)',velocity_bias:'Per-trajectory velocity bias (m/s)',steering_bias:'Per-trajectory steering bias (rad)'};
let generationRangeRows=[];
function renderGenerationRanges(params={},ranges={}){
  $('generation-range-grid').replaceChildren();generationRangeRows=[];
  for(const item of [...state.defaults.generation_parameters].sort((a,b)=>['physical','actuator','pose'].indexOf(a.group)-['physical','actuator','pose'].indexOf(b.group))){
    const key=item.key,base=params[key==='max_steering'?'max_steering_angle':key]??item.default;
    const spec=ranges[key],mode=spec?.[2]||'uniform';
    const row=document.createElement('div');row.className='generation-range-card';row.dataset.group=item.group;
    const label=document.createElement('label');label.className='generation-range-name';
    const enabled=document.createElement('input');enabled.type='checkbox';enabled.checked=Boolean(spec);enabled.setAttribute('aria-label',`Randomize ${key}`);
    const title=document.createElement('span');title.textContent=names[key]||generationRangeLabels[key]||key;title.title=`${title.textContent} (${key})`;
    const variable=document.createElement('code');variable.className='generation-range-variable';variable.textContent=key;label.append(enabled,title);
    const distribution=document.createElement('select');distribution.setAttribute('aria-label',`Distribution ${key}`);
    for(const method of key==='velocity_bias'||key==='steering_bias'?['uniform']:['uniform','normal']){const option=new Option(method==='uniform'?'Uniform':'Normal',method);distribution.append(option);}distribution.value=mode;
    const inputs=[0,1].map(index=>{const input=document.createElement('input');input.type='number';input.step='any';input.value=spec?.[index]??base??'';return input;});
    const wrappers=inputs.map(input=>{const wrapper=document.createElement('label');const caption=document.createElement('span');wrapper.append(caption,input);return {wrapper,caption};});
    const refresh=()=>{distribution.disabled=!enabled.checked;inputs.forEach(input=>input.disabled=!enabled.checked);const labels=distribution.value==='normal'?['Mean','Std']:['Min','Max'];wrappers.forEach((part,index)=>{part.caption.textContent=labels[index];inputs[index].setAttribute('aria-label',`${labels[index]} ${key}`);});row.classList.toggle('enabled',enabled.checked);updateGenerationRangeStatus();};
    const fields=document.createElement('div');fields.className='generation-range-fields';const modeLabel=document.createElement('label');const modeCaption=document.createElement('span');modeCaption.textContent='Distribution';modeLabel.append(modeCaption,distribution);fields.append(modeLabel,...wrappers.map(part=>part.wrapper));row.append(label,variable,fields);$('generation-range-grid').append(row);
    generationRangeRows.push({key,enabled,distribution,inputs,refresh});
    enabled.addEventListener('change',refresh);distribution.addEventListener('change',()=>{inputs[0].value=base??'';inputs[1].value=distribution.value==='normal'?0:base??'';refresh();});inputs.forEach(input=>input.addEventListener('input',updateGenerationRangeStatus));refresh();
  }
}
function readGenerationRanges(){
  const ranges={};
  for(const {key,enabled,distribution,inputs}of generationRangeRows){
    if(!enabled.checked)continue;
    if(inputs.some(input=>input.value===''||input.validity.badInput||!Number.isFinite(Number(input.value))))throw new Error(`${key}: enter two finite range values.`);
    const values=inputs.map(input=>Number(input.value));
    if(distribution.value==='uniform'&&values[0]>values[1])throw new Error(`${key}: maximum must be at least the minimum.`);
    if(distribution.value==='normal'&&values[1]<0)throw new Error(`${key}: standard deviation must be nonnegative.`);
    ranges[key]=[...values,distribution.value];
  }
  return ranges;
}
function updateGenerationRangeStatus(){const count=generationRangeRows.filter(row=>row.enabled.checked).length;$('generation-ranges-status').textContent=count?`${count} parameters selected · Equal min/max fixes the value`:'Using preset values.';}
bind('generation-ranges-all',()=>{for(const row of generationRangeRows){row.enabled.checked=true;row.refresh();}});
bind('generation-ranges-clear',()=>{for(const row of generationRangeRows){row.enabled.checked=false;row.refresh();}});
