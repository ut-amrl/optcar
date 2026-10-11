'use strict';
// Source orchestration only: tree rendering, bin reading and FKD processing are shared.
const builderState = {sources: [], plan: null, revision: 0, active: false};
function builderStatus(text, error=false) {
  $('builder-status').textContent=text;
  $('builder-status').classList.toggle('error', error);
}
function invalidateBuilder() {
  builderState.revision++;
  builderState.plan=null;
  $('builder-build').disabled=true;
  $('builder-summary').textContent='Selection or settings changed. Review selected bins before building.';
  $('builder-errors').hidden=true;
  const dt=Number($('builder-dt').value),h=Number($('history-steps').value),f=Number($('future-steps').value);
  $('builder-window-hint').textContent=`${h+f} samples per window · ${(Math.max(0,h-1)*dt).toFixed(2)}s observed history · ${(f*dt).toFixed(2)}s future`;
  const batch=Number($('builder-shard-windows').value);
  $('builder-output-shape-hint').textContent=`Processed shape: [${batch.toLocaleString()}, ${h+f}, 42]. The final bin for each terrain/source/split may be smaller. Synthetic batch size sets trajectories per raw bin.`;
}
function builderRequest(sources=builderState.sources, selected=true) {
  return {output:$('builder-output').value.trim(),device:$('builder-device').value,workers:number('builder-workers'),data:dataConfig(),seed:number('builder-seed'),
    fractions:['builder-train','builder-val','builder-test'].map(id=>number(id)/100),
    sources:sources.map(source=>({path:source.path.value.trim(),terrain:source.terrain.value.trim(),
      source:source.kind,split:source.split.value,...(source.vehicle.value.trim()?{vehicle_id:Number(source.vehicle.value)}:{}),
      ...(source.dt.value.trim()?{dt:Number(source.dt.value)}:{}),
      ...(source.columns.value?{columns:Number(source.columns.value)}:{}),
      ...(source.envs.value.trim()?{num_envs:Number(source.envs.value)}:{}),
      ...(selected?{selected_paths:[...source.selected]}:{})}))};
}
function builderLabel(text, input) {
  const label=document.createElement('label');label.append(document.createTextNode(text),input);return label;
}
function builderInput(value='', type='text') {
  const input=document.createElement('input');input.type=type;input.value=value;return input;
}
function builderSelect(options) {
  const select=document.createElement('select');
  for(const [value,text]of options){const option=document.createElement('option');option.value=value;option.textContent=text;select.append(option);}
  return select;
}
function addDatasetSource(path='', terrain='road', kind='real') {
  const source={kind,selected:new Set(),opened:new Set(),bins:[],scanRevision:0};
  const card=document.createElement('section');card.className='raw-card builder-source';source.card=card;
  const heading=document.createElement('div');heading.className='section-heading';
  const title=document.createElement('h2');title.textContent=kind==='real'?'Real recordings':'Synthetic episodes';
  const remove=document.createElement('button');remove.className='secondary';remove.textContent='Remove';
  remove.onclick=()=>{source.scanRevision++;builderState.sources=builderState.sources.filter(s=>s!==source);card.remove();invalidateBuilder();};
  heading.append(title,remove);card.append(heading);
  source.path=builderInput(path);source.path.placeholder='/path/to/raw/bins';
  source.terrain=builderInput(terrain);
  source.split=builderSelect([['auto','Automatic'],['train','Training'],['val','Validation'],['test','Test']]);
  source.vehicle=builderInput('','number');source.vehicle.placeholder='File index (WheeledLab)';source.vehicle.min='0';source.vehicle.step='1';
  source.dt=builderInput('','number');source.dt.placeholder='Metadata';source.dt.step='any';
  source.columns=builderSelect([['','Metadata / detect'],['15','15 columns'],['17','17 columns']]);
  source.envs=builderInput('','number');source.envs.placeholder='Metadata / 1';source.envs.min='1';
  const two=document.createElement('div');two.className='two';two.append(builderLabel('Terrain',source.terrain),builderLabel('Split',source.split));
  card.append(builderLabel('Raw bin folder or file',source.path),two);
  const advanced=document.createElement('details'),summary=document.createElement('summary');summary.textContent='Vehicle & standalone bin settings';
  advanced.append(summary,builderLabel('Vehicle ID',source.vehicle),builderLabel('Timestep override (s)',source.dt),builderLabel('Layout',source.columns),builderLabel('Environments per bin',source.envs));
  card.append(advanced);
  const buttons=document.createElement('div');buttons.className='button-row';
  const scan=document.createElement('button');scan.className='primary';scan.textContent='Scan folder';
  const all=document.createElement('button');all.className='secondary';all.textContent='Select all';
  const clear=document.createElement('button');clear.className='secondary';clear.textContent='Clear';
  const collapse=document.createElement('button');collapse.className='secondary';collapse.textContent='Collapse';
  all.onclick=()=>{source.selected=new Set(source.bins.filter(e=>!e.error).map(e=>e.path));invalidateBuilder();render();};
  clear.onclick=()=>{source.selected.clear();invalidateBuilder();render();};
  collapse.onclick=()=>{source.opened.clear();source.filter.value='';render();};
  buttons.append(scan,all,clear,collapse);card.append(buttons);
  source.filter=builderInput();source.filter.placeholder='Filter folders or bins';source.filter.setAttribute('aria-label','Filter source bins');
  source.tree=document.createElement('div');source.tree.className='bin-tree builder-tree';
  source.info=document.createElement('p');source.info.className='hint';source.info.textContent='Scan to browse and select recordings.';
  card.append(source.filter,source.info,source.tree);
  function render() {
    OptCarBinExplorer.render({tree:source.tree,entries:source.bins,filter:source.filter.value,opened:source.opened,selected:source.selected,scrollToSelection:false,
      onSelect:file=>{source.selected.has(file)?source.selected.delete(file):source.selected.add(file);invalidateBuilder();render();},
      formatDuration:seconds=>`${Number(seconds).toFixed(1)}s`});
    source.info.textContent=`${source.selected.size} selected / ${source.bins.length} bins · ${source.bins.filter(e=>e.error).length} unavailable`;
  }
  source.filter.oninput=render;
  for(const input of [source.path,source.terrain,source.split,source.vehicle,source.dt,source.columns,source.envs])input.addEventListener('input',()=>{
    source.scanRevision++;source.bins=[];source.selected.clear();source.tree.replaceChildren();source.info.textContent='Source changed. Scan again.';invalidateBuilder();
  });
  scan.onclick=async()=>{
    scan.disabled=true;invalidateBuilder();const revision=++source.scanRevision;
    try{
      const result=await api('/api/dataset-builder/inspect',builderRequest([source],false));
      if(revision!==source.scanRevision)return;
      source.bins=result.bins;source.selected=new Set(result.bins.filter(e=>!e.error).map(e=>e.path));render();
      const unavailable=result.bins.filter(e=>e.error);
      builderStatus(unavailable.length?`Found ${result.bins.length} bins · ${unavailable.length} unavailable. ${unavailable[0].name}: ${unavailable[0].error}`:`Found ${result.bins.length} bins. Choose recordings, then review the dataset.`,Boolean(unavailable.length));
    }catch(error){builderStatus(error.message,true);}finally{scan.disabled=false;}
  };
  builderState.sources.push(source);$('builder-sources').append(card);invalidateBuilder();return source;
}
function displayBuilderPlan(plan) {
  const summary=$('builder-summary');summary.replaceChildren();
  const metrics=document.createElement('div');metrics.className='metrics builder-metrics';
  for(const [title,value]of [['Selected bin files',plan.files],['Training windows',plan.windows.train],['Validation windows',plan.windows.val],['Test windows',plan.windows.test]]){
    const box=document.createElement('div'),label=document.createElement('span'),count=document.createElement('strong');
    label.textContent=title;count.textContent=value.toLocaleString();box.append(label,count);metrics.append(box);
  }
  const explanation=document.createElement('p');explanation.className='hint';explanation.textContent=`Each window has ${plan.config.history_steps+plan.config.future_steps} samples (${plan.config.history_steps} history + ${plan.config.future_steps} future). Starts are ${plan.config.stride} samples apart. Split counts are windows, not raw timesteps. Output shape: [${plan.config.shard_windows.toLocaleString()}, ${plan.config.history_steps+plan.config.future_steps}, 42], with a smaller final bin per terrain/source/split if needed.`;summary.append(metrics,explanation);const errors=$('builder-errors');errors.replaceChildren();
  for(const message of plan.errors){const item=document.createElement('li');item.textContent=message;errors.append(item);}errors.hidden=!plan.errors.length;
}
bind('builder-add-real',()=>addDatasetSource());
bind('builder-add-sim',()=>addDatasetSource('','road','sim'));
bind('builder-review',async()=>{
  const request=builderRequest(),revision=builderState.revision;
  const plan=await api('/api/dataset-builder/inspect',request);
  if(revision!==builderState.revision){builderStatus('Settings changed during review. Review again.');return;}
  displayBuilderPlan(plan);builderState.plan={request,plan};
  $('builder-build').disabled=builderState.active||Boolean(plan.errors.length)||!plan.files;
  builderStatus(plan.errors.length?'Resolve the source errors, then review again.':plan.files?'Review complete. Ready to build.':'No usable bins selected.',Boolean(plan.errors.length));
});
$('builder-build').onclick=async()=>{
  if(!builderState.plan||builderState.active)return;
  const request=builderState.plan.request;builderState.active=true;$('builder-build').disabled=true;
  try{
    const result=await api('/api/dataset-builder/build',request);
    builderStatus(`Started ${result.tmux_session}. Attach with: ${result.attach_command}`);
    await refreshBuilderRuns();
  }catch(error){builderStatus(error.message,true);}
  finally{builderState.active=false;$('builder-build').disabled=!builderState.plan||Boolean(builderState.plan.plan.errors.length)||!builderState.plan.plan.files;}
};
for(const id of ['builder-workers','builder-device','builder-output','history-steps','future-steps','builder-dt','stride','builder-shard-windows','builder-seed','builder-train','builder-val','builder-test'])$(id).addEventListener('input',invalidateBuilder);
invalidateBuilder();

// Reuse the generation dashboard's cards and tmux controls for dataset builds.
const builderRunCards=new Map();
let builderRunsBusy=false;
async function refreshBuilderRuns(){
  if(builderRunsBusy)return;
  builderRunsBusy=true;
  try{
    const {runs}=await api('/api/dataset-builder/runs'),container=$('builder-jobs');
    $('builder-runs-summary').textContent=`${runs.filter(generationRunActive).length} active · ${runs.filter(j=>j.status==='complete').length} complete · ${runs.length} total`;
    if(!runs.length){container.textContent='No tmux builds yet.';return;}
    if(!builderRunCards.size)container.replaceChildren();
    let position=0;
    for(const job of runs){
      let entry=builderRunCards.get(job.job_id);
      if(!entry){entry=generationRunCard(job,builderStatus,refreshBuilderRuns);builderRunCards.set(job.job_id,entry);container.append(entry.card);}
      const current=container.children[position++];if(current!==entry.card)container.insertBefore(entry.card,current||null);
      updateGenerationRun(entry,job);
    }
  }finally{builderRunsBusy=false;}
}
bind('builder-runs-refresh',refreshBuilderRuns);
bind('builder-refresh-devices',()=>refreshDevices());
setInterval(()=>{if(state.tab==='builder')refreshBuilderRuns().catch(e=>builderStatus(e.message,true));},2000);
