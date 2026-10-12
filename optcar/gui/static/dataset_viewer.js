'use strict';
// Index and shuffle the selected dataset without changing stored states.
const datasetViewerState={dataset:null,sample:null,cursor:0,request:0,busy:false,configBusy:false,options:{},rows:{dataset:[],test:[]},browserRevision:0};
const datasetViewerStorage='optcar.dataset-viewer.settings';
const datasetViewerColors={history:'#80cbd3',future:'#e7b17a',anchor:'#8bc99b',cursor:'#edf0f3'};
function datasetViewerStatus(text,error=false){$('dataset-viewer-status').textContent=text;$('dataset-viewer-status').classList.toggle('error',error);}
function datasetViewerData(){return {history_steps:Number($('dataset-viewer-history').value),future_steps:Number($('dataset-viewer-future').value),dt:Number($('dataset-viewer-dt').value)};}
function datasetViewerSettings(){
  const rows=datasetViewerState.rows,seed=$('dataset-viewer-seed').value.trim(),isTest=$('dataset-viewer-set').value==='test';
  const dl_config={...datasetViewerState.options,sim_dt:Number($('dataset-viewer-dt').value),file_selection_seed:seed===''?null:Number(seed)};
  for(const [name,prefix]of [['dataset',''],['test','test_']]){
    dl_config[prefix+'dataset_path']=rows[name].map(row=>row.path.trim());
    dl_config[prefix+'num_runs']=rows[name].map(row=>row.num_runs);
    dl_config[prefix+'num_trajectories']=rows[name].map(row=>row.num_trajectories);
  }
  return {roots:dl_config[isTest?'test_dataset_path':'dataset_path'],dl_config,is_test_set:isTest,
    config_path:$('dataset-viewer-config').value.trim(),split:$('dataset-viewer-split').value,data:datasetViewerData()};
}
function validateDatasetViewerSettings(settings,forSave=false){
  const {history_steps:history,future_steps:future,dt}=settings.data;
  const paths=forSave?settings.dl_config.dataset_path:settings.roots;
  if(!paths.length||paths.some(path=>!path))throw new Error('Add at least one folder and enter its full path.');
  for(const prefix of ['', 'test_']){
    if(settings.dl_config[prefix+'dataset_path'].some(path=>!path))throw new Error('Fill in or remove empty folder rows.');
    for(const field of ['num_runs','num_trajectories'])if(settings.dl_config[prefix+field].some(value=>!Number.isInteger(value)||value < -1))throw new Error('Folder limits must be integers >= 0, or -1 for All.');
  }
  if(!Number.isInteger(history)||history<2)throw new Error('History samples must be an integer of at least 2.');
  if(!Number.isInteger(future)||future<1)throw new Error('Future transitions must be a positive integer.');
  if(!Number.isFinite(dt)||dt<=0)throw new Error('Timestep must be positive.');
  const seed=settings.dl_config.file_selection_seed;
  if(seed!==null&&(!Number.isInteger(seed)||seed<0))throw new Error('Selection seed must be a nonnegative integer or empty.');
}
function saveDatasetViewerSettings(){try{localStorage.setItem(datasetViewerStorage,JSON.stringify({...datasetViewerSettings(),browser_root:$('dataset-browser-root').value}));}catch{}}
function datasetViewerControls(){
  const {dataset,sample,busy,configBusy}=datasetViewerState,loaded=Boolean(dataset?.count),index=sample?.index??0,blocked=busy||configBusy;
  for(const id of ['dataset-viewer-load','dataset-viewer-config-load','dataset-viewer-config-save'])$(id).disabled=blocked;
  $('dataset-viewer-load').textContent=busy?'Loading…':'Load dataset';
  for(const id of ['dataset-viewer-index','dataset-viewer-show','dataset-viewer-shuffle'])$(id).disabled=!loaded||blocked;
  $('dataset-viewer-previous').disabled=!sample||blocked||index<=0;
  $('dataset-viewer-next').disabled=!sample||blocked||index>=dataset?.count-1;
  $('dataset-viewer-frame').disabled=!sample||blocked;$('dataset-viewer-current').disabled=!sample||blocked;
}
function invalidateDatasetViewer(){
  datasetViewerState.request++;datasetViewerState.dataset=null;datasetViewerState.sample=null;datasetViewerState.busy=false;
  $('dataset-viewer-index').value=0;$('dataset-viewer-metadata').hidden=true;$('dataset-viewer-selection').replaceChildren();
  $('dataset-viewer-title').textContent='Dataset viewer';
  $('dataset-viewer-path').textContent='Settings changed. Load the dataset to apply them.';
  saveDatasetViewerSettings();datasetViewerControls();drawDatasetViewer();datasetViewerStatus('Settings changed. Click Load dataset.');
}
function renderDatasetViewerRows(){
  for(const name of ['dataset','test']){
    const container=$(name==='dataset'?'dataset-viewer-datasets':'dataset-viewer-test-datasets');container.replaceChildren();
    if(!datasetViewerState.rows[name].length){const hint=document.createElement('p');hint.className='hint';hint.textContent='Add folders from the explorer or enter a path.';container.append(hint);}
    datasetViewerState.rows[name].forEach((row,index)=>{
      const card=document.createElement('div');card.className='viewer-folder-row';
      const heading=document.createElement('div');heading.className='viewer-row-heading';
      const title=document.createElement('span');title.className='muted';title.textContent=`${name==='test'?'Test folder':'Folder'} ${index+1}`;
      const remove=document.createElement('button');remove.className='secondary';remove.textContent='Remove';
      remove.onclick=()=>{datasetViewerState.rows[name].splice(index,1);renderDatasetViewerRows();invalidateDatasetViewer();};heading.append(title,remove);card.append(heading);
      const pathLabel=document.createElement('label');pathLabel.textContent='Folder path';const input=document.createElement('input');input.value=row.path;input.placeholder='/full/path/to/folder/with/bins';
      input.addEventListener('input',()=>{row.path=input.value;invalidateDatasetViewer();});pathLabel.append(input);card.append(pathLabel);
      const limits=document.createElement('div');limits.className='two';
      for(const [key,text]of [['num_trajectories','Trajectories (-1 = All)'],['num_runs','Bin files (-1 = All)']]){
        const label=document.createElement('label');label.textContent=text;const field=document.createElement('input');field.type='number';field.min=-1;field.step=1;field.value=row[key];
        field.addEventListener('input',()=>{row[key]=field.value===''?NaN:Number(field.value);invalidateDatasetViewer();});label.append(field);limits.append(label);
      }
      card.append(limits);container.append(card);
    });
  }
  if(datasetViewerState.rows.test.length)$('dataset-viewer-test-section').open=true;
}
function addDatasetViewerFolder(path='',target=$('dataset-browser-target').value){
  if(path&&datasetViewerState.rows[target].some(row=>row.path===path)){datasetViewerStatus('That folder is already in this list.');return;}
  datasetViewerState.rows[target].push({path,num_runs:-1,num_trajectories:-1});renderDatasetViewerRows();invalidateDatasetViewer();
}
function applyDatasetViewerLoader(config){
  datasetViewerState.options={...config};
  for(const [name,prefix]of [['dataset',''],['test','test_']]){
    let paths=config[prefix+'dataset_path']||[];if(typeof paths==='string')paths=[paths];
    const runs=config[prefix+'num_runs'],limits=config[prefix+'num_trajectories'];
    datasetViewerState.rows[name]=paths.map((path,i)=>({path,num_runs:Array.isArray(runs)?runs[i]:runs??-1,num_trajectories:Array.isArray(limits)?limits[i]??-1:limits??-1}));
  }
  $('dataset-viewer-seed').value=config.file_selection_seed??'';$('dataset-viewer-dt').value=config.sim_dt;
  renderDatasetViewerRows();renderDatasetViewerOptions();
}
function renderDatasetViewerOptions(){
  const container=$('dataset-viewer-options');container.replaceChildren();
  const skipped=new Set(['dataset_path','test_dataset_path','num_runs','test_num_runs','num_trajectories','test_num_trajectories','sim_dt','file_selection_seed']);
  for(const [key,value]of Object.entries(datasetViewerState.options)){
    if(skipped.has(key))continue;
    const label=document.createElement('label');label.textContent=key;
    const field=document.createElement('input');field.dataset.loaderOption=key;
    if(typeof value==='boolean'){field.type='checkbox';field.checked=value;}
    else{field.type=typeof value==='number'||key==='batch_size'?'number':'text';field.value=value??'';if(field.type==='number'){field.min=1;field.step=1;}}
    if(key==='dtype')field.readOnly=true;
    field.addEventListener('input',()=>{datasetViewerState.options[key]=field.type==='checkbox'?field.checked:field.type==='number'?(field.value===''?null:Number(field.value)):field.value;invalidateDatasetViewer();});
    label.append(field);container.append(label);
  }
}
async function loadDatasetViewerConfig(path=$('dataset-viewer-config').value.trim(),scopeRoot=null){
  if(!path)throw new Error('Enter the full path of the config file to load.');
  await datasetViewerReady;
  const revision=datasetViewerState.request;datasetViewerState.configBusy=true;datasetViewerControls();datasetViewerStatus('Reading dataloader config…');
  try{
    const config=await api('/api/dataset-viewer/config/load',{path});
    if(revision!==datasetViewerState.request||$('dataset-viewer-config').value.trim()!==path)return false;
    if(config.dl_config)applyDatasetViewerLoader(config.dl_config);
    else if(config.roots?.length)applyDatasetViewerLoader({...datasetViewerState.options,dataset_path:config.roots,test_dataset_path:[],num_runs:-1,num_trajectories:null,file_selection_seed:null,sim_dt:config.data.dt});
    if(scopeRoot){datasetViewerState.rows.dataset=[{path:scopeRoot,num_runs:-1,num_trajectories:-1}];renderDatasetViewerRows();}
    if(config.data)for(const [id,key]of [['dataset-viewer-history','history_steps'],['dataset-viewer-future','future_steps'],['dataset-viewer-dt','dt']])$(id).value=config.data[key];
    $('dataset-viewer-split').value=config.split;$('dataset-viewer-set').value='dataset';$('dataset-viewer-config').value=config.config_path;
    invalidateDatasetViewer();datasetViewerStatus(config.dl_config&&!config.data?'Config loaded. Set history/future from your model config, then Load dataset.':'Config loaded. Ready to load dataset.');return true;
  }catch(error){if(revision===datasetViewerState.request&&$('dataset-viewer-config').value.trim()===path)throw error;return false;}
  finally{datasetViewerState.configBusy=false;datasetViewerControls();}
}
async function saveDatasetViewerConfig(){
  await datasetViewerReady;const settings=datasetViewerSettings(),path=settings.config_path;
  if(!path||(!path.startsWith('/')&&!path.startsWith('~/'))||!path.endsWith('.py'))throw new Error('Choose a full path and .py filename for the dataloader config.');
  validateDatasetViewerSettings(settings,true);datasetViewerState.configBusy=true;datasetViewerControls();datasetViewerStatus('Saving dataloader config…');
  try{
    const config=await api('/api/dataset-viewer/config/save',{path,dl_config:settings.dl_config,data:settings.data});
    if($('dataset-viewer-config').value.trim()===path)$('dataset-viewer-config').value=config.config_path;
    saveDatasetViewerSettings();datasetViewerStatus(`Dataloader config saved: ${config.config_path}`);
  }finally{datasetViewerState.configBusy=false;datasetViewerControls();}
}
function showDatasetViewerMetadata(dataset){
  const list=$('dataset-viewer-metadata');list.replaceChildren();list.hidden=false;
  for(const [name,value]of [['Trajectories',dataset.count.toLocaleString()],['Bin files',Number(dataset.files).toLocaleString()],['Window shape',`[B, ${dataset.samples}, ${dataset.features}]`],['History / future',`${dataset.history_steps} / ${dataset.future_steps}`],['Timestep',`${dataset.dt}s`],['Split filter',dataset.split]]){
    const label=document.createElement('dt'),detail=document.createElement('dd');label.textContent=name;detail.textContent=value;list.append(label,detail);
  }
  $('dataset-viewer-index').max=Math.max(0,dataset.count-1);const selection=$('dataset-viewer-selection');selection.replaceChildren();
  for(const row of dataset.selection||[]){const card=document.createElement('div');card.textContent=row.path;const counts=document.createElement('small');counts.textContent=`${row.selected.toLocaleString()} selected / ${row.available.toLocaleString()} available trajectories · ${row.files} / ${row.available_files} bins`;card.append(counts);selection.append(card);}
}
async function fetchDatasetViewerSample(index,request){
  const dataset=datasetViewerState.dataset;if(!dataset)throw new Error('Load a dataset first.');
  if(!Number.isInteger(index)||index<0||index>=dataset.count)throw new Error(`Choose an integer trajectory index from 0 to ${dataset.count-1}.`);
  const sample=await api('/api/dataset-viewer/sample',{dataset_id:dataset.dataset_id,index});if(request!==datasetViewerState.request)return;
  datasetViewerState.sample=sample;$('dataset-viewer-index').value=sample.index;$('dataset-viewer-title').textContent=`Trajectory ${sample.index.toLocaleString()}`;
  $('dataset-viewer-path').textContent=`${sample.path} · file ${sample.file_index} / window ${sample.window_index} · ${sample.source} / ${sample.terrain} · vehicle ${sample.car_number}${sample.shuffled?' · original dataset index '+sample.dataset_index:''}`;
  $('dataset-viewer-frame').max=sample.world.length-1;$('dataset-viewer-current').max=sample.world.length-1;setDatasetViewerFrame(sample.current_step);
  datasetViewerStatus(`Showing trajectory ${sample.index.toLocaleString()} of ${dataset.count.toLocaleString()} (0-based)${sample.shuffled?' · shuffled order':''}. Body anchor: sample ${sample.current_step}.`);
}
async function loadDatasetViewer(){
  await datasetViewerReady;const settings=datasetViewerSettings();validateDatasetViewerSettings(settings);
  const request=++datasetViewerState.request;datasetViewerState.busy=true;datasetViewerState.dataset=null;datasetViewerState.sample=null;
  $('dataset-viewer-title').textContent='Loading dataset…';$('dataset-viewer-path').textContent=settings.roots.join(' · ');$('dataset-viewer-metadata').hidden=true;$('dataset-viewer-selection').replaceChildren();
  datasetViewerControls();drawDatasetViewer();saveDatasetViewerSettings();datasetViewerStatus('Indexing bins and applying per-folder limits…');
  try{const dataset=await api('/api/dataset-viewer/open',settings);if(request!==datasetViewerState.request)return;datasetViewerState.dataset=dataset;showDatasetViewerMetadata(dataset);await fetchDatasetViewerSample(0,request);}
  catch(error){if(request===datasetViewerState.request){$('dataset-viewer-title').textContent='No trajectory selected';throw error;}}
  finally{if(request===datasetViewerState.request){datasetViewerState.busy=false;datasetViewerControls();}}
}
async function showDatasetViewerTrajectory(index=Number($('dataset-viewer-index').value)){
  const request=++datasetViewerState.request;datasetViewerState.busy=true;datasetViewerControls();datasetViewerStatus(`Loading trajectory ${index}…`);
  try{await fetchDatasetViewerSample(index,request);}catch(error){if(request===datasetViewerState.request)throw error;}
  finally{if(request===datasetViewerState.request){datasetViewerState.busy=false;datasetViewerControls();}}
}
async function shuffleDatasetViewer(){
  const dataset=datasetViewerState.dataset;if(!dataset)throw new Error('Load a dataset first.');
  const value=$('dataset-viewer-shuffle-seed').value.trim(),seed=value===''?null:Number(value),request=++datasetViewerState.request;
  if(seed!==null&&(!Number.isInteger(seed)||seed<0))throw new Error('Shuffle seed must be a nonnegative integer or empty.');
  datasetViewerState.busy=true;datasetViewerControls();datasetViewerStatus('Shuffling selected trajectories…');
  try{const result=await api('/api/dataset-viewer/shuffle',{dataset_id:dataset.dataset_id,seed});if(request!==datasetViewerState.request)return;dataset.shuffled=true;await fetchDatasetViewerSample(0,request);if(request===datasetViewerState.request)datasetViewerStatus(`Shuffled ${result.count.toLocaleString()} selected trajectories · seed ${result.seed}.`);}
  finally{if(request===datasetViewerState.request){datasetViewerState.busy=false;datasetViewerControls();}}
}
function setDatasetViewerFrame(value){const sample=datasetViewerState.sample;if(!sample)return;const cursor=Math.max(0,Math.min(sample.world.length-1,Math.round(Number(value)||0)));datasetViewerState.cursor=cursor;$('dataset-viewer-frame').value=cursor;$('dataset-viewer-current').value=cursor;drawDatasetViewer();}
async function browseDatasetViewerFolder(path=$('dataset-browser-root').value.trim()){
  if(!path)throw new Error('Enter a folder to browse.');
  const revision=++datasetViewerState.browserRevision;$('dataset-browser-status').textContent='Reading folder…';
  const browse=folder=>api('/api/dataset-viewer/browse',{path:folder,data:datasetViewerData()});
  const result=await browse(path);if(revision!==datasetViewerState.browserRevision)return;
  $('dataset-browser-root').value=result.path;saveDatasetViewerSettings();const tree=$('dataset-browser-tree');tree.replaceChildren();
  const folder=await OptCarBinExplorer.directory({path:result.path,browse,onCopy:copyDatasetViewerPath,onAdd:addDatasetViewerFolder,isCurrent:()=>revision===datasetViewerState.browserRevision});
  tree.append(folder);folder.open=true;$('dataset-browser-status').textContent='Choose a folder to copy or add.';
}
async function copyDatasetViewerPath(path){try{await navigator.clipboard.writeText(path);$('dataset-browser-status').textContent=`Copied: ${path}`;}catch{$('dataset-browser-status').textContent=`Path: ${path}`;}}
function datasetViewerMarker(ctx,point,color,radius){
  ctx.beginPath();ctx.arc(point[0],point[1],radius,0,Math.PI*2);ctx.fillStyle=color;ctx.fill();
}
function datasetViewerTrajectory(id,rows,yaw){
  const {ctx,w,h}=canvas(id);if(w<1||h<1)return;
  const sample=datasetViewerState.sample;
  if(!sample||!rows?.length){ctx.font='14px system-ui, sans-serif';ctx.fillStyle='#b0bac5';ctx.textAlign='center';ctx.fillText('No dataset loaded',w/2,h/2);ctx.textAlign='left';return;}
  const left=60,right=w-30,top=20,bottom=h-32;
  let minX=Infinity,maxX=-Infinity,minY=Infinity,maxY=-Infinity;
  for(const row of rows){if(!Number.isFinite(row[0])||!Number.isFinite(row[1]))continue;minX=Math.min(minX,row[0]);maxX=Math.max(maxX,row[0]);minY=Math.min(minY,row[1]);maxY=Math.max(maxY,row[1]);}
  if(!Number.isFinite(minX))return;
  const spanX=Math.max(.2,maxX-minX)*1.15,spanY=Math.max(.2,maxY-minY)*1.15;
  const scale=Math.min((right-left)/spanX,(bottom-top)/spanY),cx=(minX+maxX)/2,cy=(minY+maxY)/2;
  const map=row=>[(left+right)/2+(row[0]-cx)*scale,(top+bottom)/2-(row[1]-cy)*scale];
  ctx.font='12px system-ui, sans-serif';ctx.fillStyle='#b0bac5';ctx.strokeStyle='#323840';
  for(let tick=0;tick<=4;tick++){
    const x=left+(right-left)*tick/4,y=top+(bottom-top)*tick/4;
    ctx.beginPath();ctx.moveTo(x,top);ctx.lineTo(x,bottom);ctx.moveTo(left,y);ctx.lineTo(right,y);ctx.stroke();
    ctx.textAlign='center';ctx.fillText((cx+(x-(left+right)/2)/scale).toFixed(1),x,h-10);
    ctx.textAlign='right';ctx.fillText((cy-(y-(top+bottom)/2)/scale).toFixed(1),left-8,y+4);ctx.textAlign='left';
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
  const times=sample.times,left=60,right=w-20,top=15,bottom=h-30;
  let low=0,high=0;
  for(const values of series)for(const value of values){if(Number.isFinite(value)){low=Math.min(low,value);high=Math.max(high,value);}}
  const margin=Math.max(.1,high-low)*.12;low-=margin;high+=margin;
  const start=times[0],duration=Math.max(.000001,times[times.length-1]-start);
  const map=(index,value)=>[left+(times[index]-start)/duration*(right-left),bottom-(value-low)/(high-low)*(bottom-top)];
  ctx.font='12px system-ui, sans-serif';ctx.fillStyle='#b0bac5';
  for(let tick=0;tick<=3;tick++){
    const value=low+(high-low)*tick/3,y=map(0,value)[1];ctx.strokeStyle='#323840';ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(right,y);ctx.stroke();ctx.textAlign='right';ctx.fillText(value.toFixed(2),left-8,y+4);ctx.textAlign='left';
  }
  series.forEach((values,index)=>line(ctx,values.map((value,step)=>map(step,value)),index===0?datasetViewerColors.future:datasetViewerColors.history,1.5));
  for(const [step,color,dash]of [[sample.current_step,datasetViewerColors.anchor,[4,3]],[datasetViewerState.cursor,datasetViewerColors.cursor,[]]]){
    const x=map(step,low)[0];ctx.strokeStyle=color;ctx.setLineDash(dash);ctx.beginPath();ctx.moveTo(x,top);ctx.lineTo(x,bottom);ctx.stroke();
  }
  ctx.setLineDash([]);
  const startLabel=`${start.toFixed(2)}s`,endLabel=`${times[times.length-1].toFixed(2)}s`,anchorX=map(sample.current_step,0)[0];
  ctx.fillText(startLabel,left,h-8);ctx.textAlign='right';ctx.fillText(endLabel,right,h-8);ctx.textAlign='left';
  if(anchorX>left+ctx.measureText(startLabel).width+16&&anchorX<right-ctx.measureText(endLabel).width-16){ctx.textAlign='center';ctx.fillText('0s',anchorX,h-8);ctx.textAlign='left';}
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
  await datasetViewerReady;const path=configPath||root.replace(/\/+$/,'')+'/dataset_build.json';
  datasetViewerState.rows.dataset=[{path:root,num_runs:-1,num_trajectories:-1}];renderDatasetViewerRows();
  $('dataset-viewer-config').value=path;$('dataset-browser-root').value=root;$('dataset-viewer-split').value='all';$('dataset-viewer-set').value='dataset';
  invalidateDatasetViewer();selectTab('dataset-viewer');
  try{await loadDatasetViewerConfig(path,root);}catch(error){datasetViewerStatus(`Output folder filled. Enter window settings or load a config. ${error.message}`);}
}
const datasetViewerAction=action=>async()=>{try{await action();}catch(error){datasetViewerStatus(error.message,true);}};
for(const id of ['dataset-viewer-history','dataset-viewer-future','dataset-viewer-dt','dataset-viewer-split','dataset-viewer-set','dataset-viewer-seed'])$(id).addEventListener('input',invalidateDatasetViewer);
$('dataset-viewer-config').addEventListener('input',saveDatasetViewerSettings);
$('dataset-viewer-add').onclick=()=>addDatasetViewerFolder('','dataset');$('dataset-viewer-test-add').onclick=()=>addDatasetViewerFolder('','test');
$('dataset-viewer-config-load').onclick=datasetViewerAction(()=>loadDatasetViewerConfig());$('dataset-viewer-config-save').onclick=datasetViewerAction(saveDatasetViewerConfig);
$('dataset-viewer-load').onclick=datasetViewerAction(loadDatasetViewer);$('dataset-viewer-shuffle').onclick=datasetViewerAction(shuffleDatasetViewer);
$('dataset-viewer-show').onclick=datasetViewerAction(()=>showDatasetViewerTrajectory());
$('dataset-viewer-previous').onclick=datasetViewerAction(()=>showDatasetViewerTrajectory(datasetViewerState.sample.index-1));
$('dataset-viewer-next').onclick=datasetViewerAction(()=>showDatasetViewerTrajectory(datasetViewerState.sample.index+1));
$('dataset-browser-browse').onclick=datasetViewerAction(()=>browseDatasetViewerFolder());
$('dataset-browser-up').onclick=datasetViewerAction(()=>{const path=$('dataset-browser-root').value.trim().replace(/\/+$/,'');return browseDatasetViewerFolder(path.slice(0,path.lastIndexOf('/'))||'/');});
$('dataset-browser-root').addEventListener('keydown',event=>{if(event.key==='Enter')$('dataset-browser-browse').click();});
$('dataset-viewer-index').addEventListener('keydown',event=>{if(event.key==='Enter'){event.preventDefault();$('dataset-viewer-show').click();}});
$('dataset-viewer-frame').addEventListener('input',()=>setDatasetViewerFrame($('dataset-viewer-frame').value));$('dataset-viewer-current').addEventListener('change',()=>setDatasetViewerFrame($('dataset-viewer-current').value));
const datasetViewerReady=(async()=>{
  try{
    const defaults=await api('/api/dataset-viewer/defaults');let saved=null;try{saved=JSON.parse(localStorage.getItem(datasetViewerStorage)||'null');}catch{}
    const loader={...defaults.dl_config,...saved?.dl_config};
    if(saved?.roots&&!saved.dl_config)loader.dataset_path=saved.roots;
    applyDatasetViewerLoader(loader);
    if(saved){
      $('dataset-viewer-config').value=saved.config_path||'';$('dataset-browser-root').value=saved.browser_root||'';
      if(['train','val','test','all'].includes(saved.split))$('dataset-viewer-split').value=saved.split;
      $('dataset-viewer-set').value=saved.is_test_set?'test':'dataset';
      for(const [id,key,min]of [['dataset-viewer-history','history_steps',2],['dataset-viewer-future','future_steps',1],['dataset-viewer-dt','dt',.000001]]){const value=saved.data?.[key];if(Number.isFinite(value)&&value>=min&&(key==='dt'||Number.isInteger(value)))$(id).value=value;}
    }
  }catch(error){datasetViewerStatus(`Cannot load loader defaults: ${error.message}`,true);throw error;}
})();
new ResizeObserver(()=>requestAnimationFrame(drawDatasetViewer)).observe($('dataset-viewer-world'));
new ResizeObserver(()=>requestAnimationFrame(drawDatasetViewer)).observe($('dataset-viewer-body'));datasetViewerControls();
