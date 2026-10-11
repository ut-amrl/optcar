'use strict';
// Plot the saved processed states directly; the body frame stays at history - 1.
const datasetViewerState={dataset:null,sample:null,cursor:0,request:0,busy:false,configBusy:false};
const datasetViewerStorage='optcar.dataset-viewer.settings';
const datasetViewerColors={history:'#6bd1d6',future:'#eab58a',anchor:'#c2ed80',cursor:'#e1e9eb'};
function datasetViewerStatus(text,error=false){
  $('dataset-viewer-status').textContent=text;
  $('dataset-viewer-status').classList.toggle('error',error);
}
function datasetViewerSettings(){
  return {roots:$('dataset-viewer-roots').value.split(/\r?\n/).map(path=>path.trim()).filter(Boolean),
    config_path:$('dataset-viewer-config').value.trim(),split:$('dataset-viewer-split').value,
    data:{history_steps:Number($('dataset-viewer-history').value),future_steps:Number($('dataset-viewer-future').value),dt:Number($('dataset-viewer-dt').value)}};
}
function validateDatasetViewerSettings(settings){
  const {history_steps:history,future_steps:future,dt}=settings.data;
  if(!settings.roots.length)throw new Error('Enter at least one dataset directory.');
  if(!Number.isInteger(history)||history<2)throw new Error('History samples must be an integer of at least 2.');
  if(!Number.isInteger(future)||future<1)throw new Error('Future transitions must be a positive integer.');
  if(!Number.isFinite(dt)||dt<=0)throw new Error('Timestep must be positive.');
}
function saveDatasetViewerSettings(){
  try{localStorage.setItem(datasetViewerStorage,JSON.stringify(datasetViewerSettings()));}catch{}
}
function datasetViewerControls(){
  const {dataset,sample,busy,configBusy}=datasetViewerState,loaded=Boolean(dataset?.count),index=sample?.index??0;
  const blocked=busy||configBusy;
  $('dataset-viewer-load').disabled=blocked;
  $('dataset-viewer-load').textContent=busy?'Loading…':'Load dataset';
  $('dataset-viewer-config-load').disabled=busy||configBusy;
  $('dataset-viewer-config-save').disabled=busy||configBusy;
  $('dataset-viewer-index').disabled=!loaded||blocked;
  $('dataset-viewer-show').disabled=!loaded||blocked;
  $('dataset-viewer-previous').disabled=!sample||blocked||index<=0;
  $('dataset-viewer-next').disabled=!sample||blocked||index>=dataset.count-1;
  $('dataset-viewer-frame').disabled=!sample||blocked;
  $('dataset-viewer-current').disabled=!sample||blocked;
}
function invalidateDatasetViewer(){
  datasetViewerState.request++;
  datasetViewerState.dataset=null;datasetViewerState.sample=null;datasetViewerState.busy=false;
  $('dataset-viewer-index').value=0;
  $('dataset-viewer-metadata').hidden=true;
  $('dataset-viewer-title').textContent='See the saved training data.';
  $('dataset-viewer-path').textContent='Settings changed. Load the dataset to inspect its trajectories.';
  saveDatasetViewerSettings();datasetViewerControls();drawDatasetViewer();
  datasetViewerStatus('Settings changed. Click Load dataset.');
}
async function loadDatasetViewerConfig(path=$('dataset-viewer-config').value.trim(),scopeRoot=null){
  if(!path)throw new Error('Enter the full path of the config file to load.');
  const revision=datasetViewerState.request;
  datasetViewerState.configBusy=true;datasetViewerControls();datasetViewerStatus('Reading viewer config…');
  try{
    const config=await api('/api/dataset-viewer/config/load',{path});
    if(revision!==datasetViewerState.request||$('dataset-viewer-config').value.trim()!==path)return false;
    const hasRoots=Array.isArray(config.roots)&&config.roots.length>0;
    if(scopeRoot)$('dataset-viewer-roots').value=scopeRoot;
    else if(hasRoots)$('dataset-viewer-roots').value=config.roots.join('\n');
    $('dataset-viewer-history').value=config.data.history_steps;
    $('dataset-viewer-future').value=config.data.future_steps;
    $('dataset-viewer-dt').value=config.data.dt;
    $('dataset-viewer-split').value=config.split;
    $('dataset-viewer-config').value=config.config_path;
    invalidateDatasetViewer();
    datasetViewerStatus(scopeRoot||hasRoots?'Config loaded. Edit settings or click Load dataset.':'Window settings loaded; existing directories were kept. Enter dataset directories, then click Load dataset.');
    return true;
  }catch(error){if(revision===datasetViewerState.request&&$('dataset-viewer-config').value.trim()===path)throw error;return false;}
  finally{datasetViewerState.configBusy=false;datasetViewerControls();}
}
async function saveDatasetViewerConfig(){
  const settings=datasetViewerSettings(),path=settings.config_path;
  if(!path)throw new Error('Choose a full path and YAML or JSON filename to save the viewer config.');
  if(!path.startsWith('/')&&!path.startsWith('~/'))throw new Error('Enter the full path and filename to save the viewer config.');
  if(!/\.(yaml|yml|json)$/i.test(path)||/(^|\/)dataset_build\.json$/i.test(path))throw new Error('Choose a viewer YAML or JSON filename. Generated dataset metadata is read-only.');
  validateDatasetViewerSettings(settings);
  datasetViewerState.configBusy=true;datasetViewerControls();datasetViewerStatus('Saving viewer config…');
  try{
    const config=await api('/api/dataset-viewer/config/save',{path,roots:settings.roots,data:settings.data,split:settings.split});
    if($('dataset-viewer-config').value.trim()===path)$('dataset-viewer-config').value=config.config_path;
    saveDatasetViewerSettings();datasetViewerStatus(`Viewer config saved: ${config.config_path}`);
  }finally{datasetViewerState.configBusy=false;datasetViewerControls();}
}
function showDatasetViewerMetadata(dataset){
  const list=$('dataset-viewer-metadata');list.replaceChildren();list.hidden=false;
  for(const [name,value]of [['Trajectories',dataset.count.toLocaleString()],['Bin files',Number(dataset.files).toLocaleString()],
    ['Window shape',`[B, ${dataset.samples}, ${dataset.features}]`],['History / future',`${dataset.history_steps} / ${dataset.future_steps}`],
    ['Timestep',`${dataset.dt}s`],['Split',dataset.split]]){
    const label=document.createElement('dt'),detail=document.createElement('dd');label.textContent=name;detail.textContent=value;list.append(label,detail);
  }
  $('dataset-viewer-index').max=Math.max(0,dataset.count-1);
}
async function fetchDatasetViewerSample(index,request){
  const dataset=datasetViewerState.dataset;
  if(!dataset)throw new Error('Load a dataset first.');
  if(!Number.isInteger(index)||index<0||index>=dataset.count)throw new Error(`Choose an integer trajectory index from 0 to ${dataset.count-1}.`);
  const sample=await api('/api/dataset-viewer/sample',{dataset_id:dataset.dataset_id,index});
  if(request!==datasetViewerState.request)return;
  datasetViewerState.sample=sample;
  $('dataset-viewer-index').value=sample.index;
  $('dataset-viewer-title').textContent=`Trajectory ${sample.index.toLocaleString()}`;
  $('dataset-viewer-path').textContent=`${sample.path} · file ${sample.file_index} / window ${sample.window_index} · ${sample.source} / ${sample.terrain} · vehicle ${sample.car_number}`;
  $('dataset-viewer-frame').max=sample.world.length-1;$('dataset-viewer-current').max=sample.world.length-1;
  setDatasetViewerFrame(sample.current_step);
  datasetViewerStatus(`Showing trajectory ${sample.index.toLocaleString()} of ${dataset.count.toLocaleString()} (0-based). Body anchor: sample ${sample.current_step}.`);
}
async function loadDatasetViewer(){
  const settings=datasetViewerSettings();
  validateDatasetViewerSettings(settings);
  const previousIndex=Number($('dataset-viewer-index').value),request=++datasetViewerState.request;
  datasetViewerState.busy=true;datasetViewerState.dataset=null;datasetViewerState.sample=null;
  $('dataset-viewer-title').textContent='Loading dataset…';$('dataset-viewer-path').textContent=settings.roots.join(' · ');
  $('dataset-viewer-metadata').hidden=true;datasetViewerControls();drawDatasetViewer();saveDatasetViewerSettings();
  datasetViewerStatus('Indexing dataset with the visible window settings…');
  try{
    const dataset=await api('/api/dataset-viewer/open',settings);
    if(request!==datasetViewerState.request)return;
    datasetViewerState.dataset=dataset;showDatasetViewerMetadata(dataset);
    if(dataset.count){
      const index=Number.isInteger(previousIndex)&&previousIndex>=0&&previousIndex<dataset.count?previousIndex:0;
      await fetchDatasetViewerSample(index,request);
    }else datasetViewerStatus('No trajectories in this split. Choose another split and load again.');
  }catch(error){if(request===datasetViewerState.request){$('dataset-viewer-title').textContent='No trajectory selected';throw error;}}
  finally{if(request===datasetViewerState.request){datasetViewerState.busy=false;datasetViewerControls();}}
}
async function showDatasetViewerTrajectory(index=Number($('dataset-viewer-index').value)){
  const request=++datasetViewerState.request;
  datasetViewerState.busy=true;datasetViewerControls();datasetViewerStatus(`Loading trajectory ${index}…`);
  try{await fetchDatasetViewerSample(index,request);}
  catch(error){if(request===datasetViewerState.request)throw error;}
  finally{if(request===datasetViewerState.request){datasetViewerState.busy=false;datasetViewerControls();}}
}
function setDatasetViewerFrame(value){
  const sample=datasetViewerState.sample;if(!sample)return;
  const cursor=Math.max(0,Math.min(sample.world.length-1,Math.round(Number(value)||0)));
  datasetViewerState.cursor=cursor;$('dataset-viewer-frame').value=cursor;$('dataset-viewer-current').value=cursor;
  drawDatasetViewer();
}
function datasetViewerMarker(ctx,point,color,radius){
  ctx.beginPath();ctx.arc(point[0],point[1],radius,0,Math.PI*2);ctx.fillStyle=color;ctx.fill();
}
function datasetViewerTrajectory(id,rows,yaw){
  const {ctx,w,h}=canvas(id);if(w<1||h<1)return;
  const sample=datasetViewerState.sample;
  if(!sample||!rows?.length){ctx.font='11px system-ui';ctx.fillStyle='#80919d';ctx.textAlign='center';ctx.fillText('Load a dataset to see a trajectory.',w/2,h/2);ctx.textAlign='left';return;}
  const left=44,right=w-18,top=20,bottom=h-30;
  let minX=Infinity,maxX=-Infinity,minY=Infinity,maxY=-Infinity;
  for(const row of rows){if(!Number.isFinite(row[0])||!Number.isFinite(row[1]))continue;minX=Math.min(minX,row[0]);maxX=Math.max(maxX,row[0]);minY=Math.min(minY,row[1]);maxY=Math.max(maxY,row[1]);}
  if(!Number.isFinite(minX))return;
  const spanX=Math.max(.2,maxX-minX)*1.15,spanY=Math.max(.2,maxY-minY)*1.15;
  const scale=Math.min((right-left)/spanX,(bottom-top)/spanY),cx=(minX+maxX)/2,cy=(minY+maxY)/2;
  const map=row=>[(left+right)/2+(row[0]-cx)*scale,(top+bottom)/2-(row[1]-cy)*scale];
  ctx.font='9px monospace';ctx.fillStyle='#748995';ctx.strokeStyle='#25333c';
  for(let tick=0;tick<=4;tick++){
    const x=left+(right-left)*tick/4,y=top+(bottom-top)*tick/4;
    ctx.beginPath();ctx.moveTo(x,top);ctx.lineTo(x,bottom);ctx.moveTo(left,y);ctx.lineTo(right,y);ctx.stroke();
    ctx.fillText((cx+(x-(left+right)/2)/scale).toFixed(1),x-10,h-10);
    ctx.fillText((cy-(y-(top+bottom)/2)/scale).toFixed(1),3,y+3);
  }
  const anchor=sample.current_step;
  line(ctx,rows.slice(0,anchor+1).map(map),datasetViewerColors.history,2);
  line(ctx,rows.slice(anchor).map(map),datasetViewerColors.future,2);
  const current=map(rows[anchor]);
  ctx.save();ctx.translate(...current);ctx.rotate(-(yaw?.[anchor]||0));ctx.fillStyle=datasetViewerColors.anchor;
  ctx.beginPath();ctx.moveTo(10,0);ctx.lineTo(-6,-5);ctx.lineTo(-4,0);ctx.lineTo(-6,5);ctx.closePath();ctx.fill();ctx.restore();
  datasetViewerMarker(ctx,map(rows[datasetViewerState.cursor]),datasetViewerColors.cursor,3.5);
}
function datasetViewerSignal(id,series){
  const {ctx,w,h}=canvas(id),sample=datasetViewerState.sample;if(!sample||w<1||h<1)return;
  const times=sample.times,left=42,right=w-15,top=15,bottom=h-27;
  let low=0,high=0;
  for(const values of series)for(const value of values){if(Number.isFinite(value)){low=Math.min(low,value);high=Math.max(high,value);}}
  const margin=Math.max(.1,high-low)*.12;low-=margin;high+=margin;
  const start=times[0],duration=Math.max(.000001,times[times.length-1]-start);
  const map=(index,value)=>[left+(times[index]-start)/duration*(right-left),bottom-(value-low)/(high-low)*(bottom-top)];
  ctx.font='9px monospace';ctx.fillStyle='#748995';
  for(let tick=0;tick<=3;tick++){
    const value=low+(high-low)*tick/3,y=map(0,value)[1];ctx.strokeStyle='#25333c';ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(right,y);ctx.stroke();ctx.fillText(value.toFixed(2),3,y+3);
  }
  series.forEach((values,index)=>line(ctx,values.map((value,step)=>map(step,value)),index===0?datasetViewerColors.future:datasetViewerColors.history,1.5));
  for(const [step,color,dash]of [[sample.current_step,datasetViewerColors.anchor,[4,3]],[datasetViewerState.cursor,datasetViewerColors.cursor,[]]]){
    const x=map(step,low)[0];ctx.strokeStyle=color;ctx.setLineDash(dash);ctx.beginPath();ctx.moveTo(x,top);ctx.lineTo(x,bottom);ctx.stroke();
  }
  ctx.setLineDash([]);ctx.fillText(`${start.toFixed(2)}s`,left,h-8);ctx.fillText('0s',map(sample.current_step,0)[0]-5,h-8);ctx.fillText(`${times[times.length-1].toFixed(2)}s`,right-38,h-8);
}
function drawDatasetViewer(){
  if($('dataset-viewer-workspace').hidden)return;
  const sample=datasetViewerState.sample;
  datasetViewerTrajectory('dataset-viewer-world',sample?.world,sample?.world_yaw);
  datasetViewerTrajectory('dataset-viewer-body',sample?.body,sample?.body_yaw);
  datasetViewerSignal('dataset-viewer-velocity',sample?[sample.actions.map(row=>row[0]),sample.body.map(row=>row[7])]:[]);
  datasetViewerSignal('dataset-viewer-steering',sample?[sample.actions.map(row=>row[1])]:[]);
  if(!sample){
    $('dataset-viewer-frame-info').textContent='No trajectory selected';
    for(const id of ['dataset-viewer-world-value','dataset-viewer-body-value'])$(id).textContent='—';
    $('dataset-viewer-velocity-value').textContent='Command —';$('dataset-viewer-forward-value').textContent='Stored body vx —';$('dataset-viewer-steering-value').textContent='Command —';return;
  }
  const cursor=datasetViewerState.cursor;
  $('dataset-viewer-frame-info').textContent=`${sample.times[cursor].toFixed(2)}s from current · anchor ${sample.current_step}`;
  for(const frame of ['world','body'])$('dataset-viewer-'+frame+'-value').textContent=`x ${sample[frame][cursor][0].toFixed(3)} · y ${sample[frame][cursor][1].toFixed(3)}`;
  $('dataset-viewer-velocity-value').textContent=`Command ${sample.actions[cursor][0].toFixed(3)}`;
  $('dataset-viewer-forward-value').textContent=`Stored body vx ${sample.body[cursor][7].toFixed(3)}`;
  $('dataset-viewer-steering-value').textContent=`Command ${sample.actions[cursor][1].toFixed(3)}`;
}
async function openProcessedDataset(root,configPath=''){
  const path=configPath||root.replace(/\/+$/,'')+'/dataset_build.json';
  $('dataset-viewer-roots').value=root;$('dataset-viewer-config').value=path;
  $('dataset-viewer-split').value='train';$('dataset-viewer-index').value=0;
  invalidateDatasetViewer();selectTab('dataset-viewer');
  try{
    if(await loadDatasetViewerConfig(path,root))datasetViewerStatus('Output folder and build settings loaded. Click Load dataset. To save a viewer config, choose a YAML or JSON filename.');
  }catch(error){datasetViewerStatus(`Output folder filled in. Enter its window settings or load a config, then click Load dataset. ${error.message}`);}
}
for(const id of ['dataset-viewer-roots','dataset-viewer-history','dataset-viewer-future','dataset-viewer-dt','dataset-viewer-split'])$(id).addEventListener('input',invalidateDatasetViewer);
$('dataset-viewer-config').addEventListener('input',saveDatasetViewerSettings);
const datasetViewerAction=action=>async()=>{try{await action();}catch(error){datasetViewerStatus(error.message,true);}};
$('dataset-viewer-config-load').onclick=datasetViewerAction(()=>loadDatasetViewerConfig());
$('dataset-viewer-config-save').onclick=datasetViewerAction(saveDatasetViewerConfig);
$('dataset-viewer-load').onclick=datasetViewerAction(loadDatasetViewer);
$('dataset-viewer-show').onclick=datasetViewerAction(()=>showDatasetViewerTrajectory());
$('dataset-viewer-previous').onclick=datasetViewerAction(()=>showDatasetViewerTrajectory(datasetViewerState.sample.index-1));
$('dataset-viewer-next').onclick=datasetViewerAction(()=>showDatasetViewerTrajectory(datasetViewerState.sample.index+1));
$('dataset-viewer-index').addEventListener('keydown',event=>{if(event.key==='Enter'){event.preventDefault();$('dataset-viewer-show').click();}});
$('dataset-viewer-frame').addEventListener('input',()=>setDatasetViewerFrame($('dataset-viewer-frame').value));
$('dataset-viewer-current').addEventListener('change',()=>setDatasetViewerFrame($('dataset-viewer-current').value));
try{
  const saved=JSON.parse(localStorage.getItem(datasetViewerStorage)||'null');
  if(saved){
    $('dataset-viewer-roots').value=(saved.roots||[]).join('\n');$('dataset-viewer-config').value=saved.config_path||'';
    if(['train','val','test','all'].includes(saved.split))$('dataset-viewer-split').value=saved.split;
    for(const [id,key,minimum]of [['dataset-viewer-history','history_steps',2],['dataset-viewer-future','future_steps',1],['dataset-viewer-dt','dt',.000001]]){
      const value=saved.data?.[key];if(Number.isFinite(value)&&value>=minimum&&(key==='dt'||Number.isInteger(value)))$(id).value=value;
    }
  }
}catch{}
new ResizeObserver(()=>requestAnimationFrame(drawDatasetViewer)).observe($('dataset-viewer-world'));
new ResizeObserver(()=>requestAnimationFrame(drawDatasetViewer)).observe($('dataset-viewer-body'));
datasetViewerControls();
