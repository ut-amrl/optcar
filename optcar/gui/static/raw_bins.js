'use strict';
// Raw layout and signal groups follow extractor.py / plot_sim_bins.py.
const binState={mode:'bins',scan:null,data:null,current:0,row:null,forward:null,playing:false,zoom:1,pan:[0,0],opened:new Set(),loadVersion:0,frameVersion:0,request:null,actions:null,actionVersion:0};
const binColors=['#80cbd3','#8bc99b','#e7b17a','#bca0df'];
const binGroups=[{title:'Position',unit:'m',columns:[0,1,2],labels:['x','y','z']},{title:'Orientation',unit:'quaternion · wxyz',columns:[3,4,5,6],labels:['w','x','y','z']},{title:'Linear velocity',unit:'world · m/s',columns:[7,8,9],labels:['x','y','z']},{title:'Angular velocity',unit:'world · rad/s',columns:[10,11,12],labels:['x','y','z']}];
const binValue=value=>Number.isFinite(value)?value.toFixed(3):'—';
// Both viewer tabs use this same reader, explorer, frame cursor, and plotting code.
const binViewerInputs=['bin-root','bin-filter','bin-dt','bin-columns','bin-envs','bin-action-mode','bin-action-before','bin-action-after','bin-velocity-min','bin-velocity-max','bin-steering-min','bin-steering-max','bin-speed'];
const binViewerDefaults=Object.fromEntries(binViewerInputs.map(id=>[id,$(id).value]));
const binViewerSaved={};
async function switchBinViewer(mode){
  if(mode===binState.mode)return;
  binViewerSaved[binState.mode]={inputs:Object.fromEntries(binViewerInputs.map(id=>[id,$(id).value])),scan:binState.scan,path:binState.data?.path,environment:binState.data?.env_index||0,current:binState.current,opened:[...binState.opened]};
  clearBin();binState.mode=mode;
  const saved=binViewerSaved[mode];
  const defaults={...binViewerDefaults,'bin-root':mode==='syn-bins'?(localStorage.getItem('optcar-syn-bin-folder')||state.defaults?.dataset_root+'/syn_data/terrains'):(localStorage.getItem('optcar-bin-folder')||'')};
  for(const id of binViewerInputs)$(id).value=(saved?.inputs||defaults)[id];
  const synthetic=mode==='syn-bins';
  $('bin-browser-title').textContent=synthetic?'Synthetic bins':'Raw bins';
  $('bin-viewer-heading').textContent=synthetic?'Synthetic bin viewer':'Raw bin viewer';
  $('bin-browser').setAttribute('aria-label',synthetic?'Synthetic bin folder browser':'Raw bin folder browser');
  $('bin-env-title').textContent=synthetic?'Batch index (0-based)':'Environment';
  $('bin-batch-info').hidden=!synthetic;
  binState.scan=saved?.scan||null;binState.opened=new Set(saved?.opened||[]);clearBin();renderBinTree();refreshBinActions();
  if(saved?.path){
    const version=binState.loadVersion+1;await loadRawBin(saved.path,saved.environment);
    if(version===binState.loadVersion&&mode===binState.mode)await setBinFrame(saved.current);
  }
}
function updateBinBatchInfo(){
  const data=binState.data,synthetic=binState.mode==='syn-bins';
  $('bin-env-label').hidden=!synthetic&&(!data||data.num_envs===1);
  $('bin-env').disabled=!data||data.num_envs===1;
  if(!data){$('bin-env').replaceChildren(new Option('0','0'));}
  $('bin-batch-info').hidden=!synthetic;
  $('bin-batch-info').textContent=!data?'Batch index selects a trajectory within the bin.':data.num_envs===1?'1 trajectory · batch index 0':`${data.num_envs} trajectories · batch index 0–${data.num_envs-1}`;
}
function binOptions(){const out={};for(const [key,id]of [['dt','bin-dt'],['columns','bin-columns'],['num_envs','bin-envs']])if($(id).value!=='')out[key]=number(id);return out;}
function stopBinPlayback(){binState.playing=false;$('bin-play').textContent='▶';$('bin-play').setAttribute('aria-label','Play raw bin');}
function clearBin(){stopBinPlayback();binState.loadVersion++;binState.frameVersion++;binState.actionVersion++;binState.actions=null;$('bin-action-time').textContent='Current: —';binState.data=null;$('bin-tune-dbm').disabled=true;binState.row=null;binState.forward=null;$('bin-forward-source').hidden=true;for(const id of ['bin-command-value','bin-forward-value','bin-steer-value'])$(id).textContent='—';for(const id of ['bin-unshifted-value','bin-unshifted-steer-value'])$(id).hidden=true;$('bin-summary').hidden=true;$('bin-empty').hidden=false;$('bin-stats').hidden=true;$('bin-warning').hidden=true;$('bin-title').textContent='No bin selected';updateBinBatchInfo();$('bin-path').textContent='Select a bin.';$('bin-frame-info').textContent='No sample selected';$('bin-signals').replaceChildren();for(const id of ['bin-play','bin-frame','bin-current'])$(id).disabled=true;drawBins();}
function renderBinTree(){
  const tree=$('bin-tree');
  const count=OptCarBinExplorer.render({tree,entries:binState.scan?.bins||[],filter:$('bin-filter').value,opened:binState.opened,selected:binState.data?.path,onSelect:path=>loadRawBin(path).catch(e=>status(e.message,true)),formatDuration:clockText});
  if(!binState.scan)tree.textContent='No folder loaded.';
  $('bin-count').textContent=`${count} / ${binState.scan?.bins.length||0} bins`;
}
async function scanBinFolder(){
  const version=binState.loadVersion+1;clearBin();status('Reading raw bin folders…');
  const result=await api('/api/raw-bins/scan',{path:$('bin-root').value.trim(),...binOptions()});
  if(version!==binState.loadVersion)return;binState.scan=result;$('bin-root').value=result.root;localStorage.setItem(binState.mode==='syn-bins'?'optcar-syn-bin-folder':'optcar-bin-folder',result.root);binState.opened.clear();
  const first=result.bins.find(b=>!b.error);if(first){const parts=first.relative_path.split('/');parts.pop();let path='';for(const folder of parts){path+=folder+'/';binState.opened.add(path);}}
  renderBinTree();if(first)await loadRawBin(first.path);else status(result.bins.length?'No readable bins. Check the layout settings.':'No .bin files found in this folder.',Boolean(result.bins.length));
}
async function openBinFolder(path,mode='bins'){if(!path)throw new Error('No output folder available.');selectTab(mode);$('bin-root').value=path;$('bin-filter').value='';for(const id of ['bin-dt','bin-columns','bin-envs'])$(id).value='';await scanBinFolder();$('bin-title').scrollIntoView({behavior:'smooth',block:'start'});}
async function loadRawBin(path,environment=0){
  clearBin();const version=binState.loadVersion;status(`Opening ${path.split('/').pop()}…`);
  const data=await api('/api/raw-bins/open',{path,...binOptions(),env_index:environment});if(version!==binState.loadVersion)return;
  binState.data=data;$('bin-tune-dbm').disabled=false;binState.current=0;binState.row=data.frame;binState.forward=data.forward_frame;binState.zoom=1;binState.pan=[0,0];
  binState.request={path:data.path,dt:data.dt,columns:data.columns,num_envs:data.num_envs,env_index:data.env_index};
  $('bin-empty').hidden=true;$('bin-title').textContent=data.name;$('bin-path').textContent=data.path;$('bin-summary').hidden=false;
  $('bin-duration').textContent=`${clockText(data.duration_s)} duration`;$('bin-samples').textContent=`${data.samples.toLocaleString()} samples`;$('bin-layout').textContent=`${data.dt}s timestep · ${data.columns} channels`;
  updateBinBatchInfo();$('bin-env').replaceChildren();for(let i=0;i<data.num_envs;i++){const option=document.createElement('option');option.value=i;option.textContent=i;option.selected=i===data.env_index;$('bin-env').append(option);}
  for(const id of ['bin-frame','bin-current']){$(id).max=data.samples-1;$(id).value=0;$(id).disabled=false;}$('bin-play').disabled=data.samples<2;
  $('bin-signals').replaceChildren();binGroups.forEach((group,i)=>{const panel=document.createElement('div');const heading=document.createElement('div');heading.className='plot-heading';heading.textContent=group.title;const unit=document.createElement('small');unit.textContent=group.unit;heading.append(unit);const legend=document.createElement('div');legend.className='bin-signal-legend';for(const [j,column]of group.columns.entries()){if(column>=data.columns)continue;const item=document.createElement('span');item.style.color=binColors[j];item.id=`bin-value-${column}`;legend.append(item);}const plot=document.createElement('canvas');plot.id=`bin-signal-${i}`;plot.setAttribute('aria-label',`${group.title} over time`);plot.addEventListener('click',event=>{const rect=plot.getBoundingClientRect();const fraction=Math.max(0,Math.min(1,(event.clientX-rect.left-60)/(rect.width-80)));setBinFrame(Math.round(fraction*(data.samples-1)));});panel.append(heading,legend,plot);$('bin-signals').append(panel);});
  $('bin-stat-rows').replaceChildren();for(const stat of data.stats){const row=document.createElement('div');const name=document.createElement('span');name.textContent=stat.name;const value=document.createElement('span');value.textContent=`${binValue(stat.min)} … ${binValue(stat.max)} · ${stat.missing} missing`;row.append(name,value);$('bin-stat-rows').append(row);}$('bin-stats').hidden=false;
  const missing=data.stats.reduce((n,s)=>n+s.missing,0)+data.forward_stats.missing;const warnings=[missing?`${missing.toLocaleString()} missing or nonfinite values found. Gaps are left blank in the plots.`:'',data.aux_warning||''].filter(Boolean);$('bin-warning').hidden=!warnings.length;$('bin-warning').textContent=warnings.join(' ');$('bin-forward-source').hidden=false;$('bin-forward-source').textContent=`Forward velocity: ${data.forward_source}.`;
  renderBinTree();drawBins();refreshBinActions();status(`Loaded ${data.name}.${data.indices.length<data.samples?' Overview sampled; cursor shows exact values.':''}`);
}
async function setBinFrame(value){
  if(!binState.data)return;const index=Math.max(0,Math.min(binState.data.samples-1,Math.round(value)));if(!Number.isFinite(index))return;
  binState.current=index;$('bin-current').value=index;$('bin-frame').value=index;const version=++binState.frameVersion;binState.row=null;binState.forward=null;drawBins();refreshBinActions();
  try{const frame=await api('/api/raw-bins/frame',{...binState.request,index});if(version!==binState.frameVersion)return;binState.row=frame.row;binState.forward=frame.forward_velocity;drawBins();}catch(error){if(version===binState.frameVersion){stopBinPlayback();status(error.message,true);}}
}
function binLine(ctx,points,color,width=1.5){let segment=[];for(const p of points){if(p.every(Number.isFinite))segment.push(p);else{line(ctx,segment,color,width);segment=[];}}line(ctx,segment,color,width);}
function drawBinTrajectory(){
  const {ctx,w,h}=canvas('bin-trajectory');if(!w||!h)return;ctx.strokeStyle='#323840';for(let x=24;x<w;x+=40){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,h);ctx.stroke();}for(let y=24;y<h;y+=40){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke();}
  const data=binState.data;if(!data)return;const finite=data.rows.filter(p=>Number.isFinite(p[0])&&Number.isFinite(p[1]));if(!finite.length)return;
  let minX=Infinity,maxX=-Infinity,minY=Infinity,maxY=-Infinity;for(const p of finite){minX=Math.min(minX,p[0]);maxX=Math.max(maxX,p[0]);minY=Math.min(minY,p[1]);maxY=Math.max(maxY,p[1]);}
  const scale=Math.min((w-90)/Math.max(1,maxX-minX),(h-70)/Math.max(1,maxY-minY))*binState.zoom;const cx=(minX+maxX)/2,cy=(minY+maxY)/2;const map=p=>[Number.isFinite(p[0])?w/2+(p[0]-cx)*scale+binState.pan[0]:NaN,Number.isFinite(p[1])?h/2-(p[1]-cy)*scale+binState.pan[1]:NaN];
  binLine(ctx,data.rows.map(map),'#71818e',1.7);binLine(ctx,data.rows.filter((p,i)=>data.indices[i]<=binState.current).map(map),'#80cbd3',2);
  const first=map(data.rows[0]);if(first.every(Number.isFinite)){ctx.fillStyle='#e7b17a';ctx.beginPath();ctx.arc(...first,4,0,Math.PI*2);ctx.fill();}
  const row=binState.row;if(row){const point=map(row);const [qw,qx,qy,qz]=row.slice(3,7);if(point.every(Number.isFinite)){ctx.save();ctx.translate(...point);ctx.fillStyle='#8bc99b';if([qw,qx,qy,qz].every(Number.isFinite)){const yaw=Math.atan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz));ctx.rotate(-yaw);ctx.beginPath();ctx.moveTo(12,0);ctx.lineTo(-7,-6);ctx.lineTo(-4,0);ctx.lineTo(-7,6);ctx.closePath();ctx.fill();}else{ctx.beginPath();ctx.arc(0,0,5,0,Math.PI*2);ctx.fill();}ctx.restore();}}
  ctx.fillStyle='#b0bac5';ctx.font='12px system-ui, sans-serif';ctx.fillText(`x ${minX.toFixed(1)} … ${maxX.toFixed(1)} m  ·  y ${minY.toFixed(1)} … ${maxY.toFixed(1)} m`,12,h-12);
}
function drawBinSignal(i,group){
  const {ctx,w,h}=canvas(`bin-signal-${i}`);const data=binState.data;if(!w||!h)return;
  const columns=group.columns.filter(c=>c<data.columns);let lo=Infinity,hi=-Infinity;for(const column of columns){const stat=data.stats[column];if(Number.isFinite(stat.min))lo=Math.min(lo,stat.min);if(Number.isFinite(stat.max))hi=Math.max(hi,stat.max);}if(!Number.isFinite(lo)){lo=-1;hi=1;}const span=Math.max(.1,hi-lo);lo-=span*.08;hi+=span*.08;
  const left=60,right=w-20,top=14,bottom=h-30;const map=(sample,value)=>[left+sample/Math.max(1,data.samples-1)*(right-left),Number.isFinite(value)?bottom-(value-lo)/(hi-lo)*(bottom-top):NaN];
  ctx.font='12px system-ui, sans-serif';ctx.fillStyle='#b0bac5';for(let j=0;j<=3;j++){const value=lo+(hi-lo)*j/3;const y=map(0,value)[1];ctx.strokeStyle='#323840';ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(right,y);ctx.stroke();ctx.textAlign='right';ctx.fillText(value.toFixed(2),left-8,y+4);ctx.textAlign='left';}
  columns.forEach((column,j)=>binLine(ctx,data.rows.map((row,k)=>map(data.indices[k],row[column])),binColors[j]));const cursor=map(binState.current,0)[0];ctx.strokeStyle='#8bc99b';ctx.setLineDash([3,3]);ctx.beginPath();ctx.moveTo(cursor,top);ctx.lineTo(cursor,bottom);ctx.stroke();ctx.setLineDash([]);ctx.fillText('0s',left,h-7);ctx.textAlign='right';ctx.fillText(`${data.duration_s.toFixed(1)}s`,right,h-7);ctx.textAlign='left';
  columns.forEach((column,j)=>{$(`bin-value-${column}`).textContent=`${group.labels[j]} ${binValue(binState.row?.[column])}`;});
}
function actionRange(){
  if($('bin-action-mode').value==='full')return null;
  const before=Number($('bin-action-before').value),after=Number($('bin-action-after').value);
  if($('bin-action-before').value===''||$('bin-action-after').value===''||!Number.isFinite(before)||!Number.isFinite(after)||before<0||after<0||before+after<=0)throw new Error('Set nonnegative before/after times with a positive total window.');
  return {before,after};
}
async function refreshBinActions(){
  const version=++binState.actionVersion;const active=$('bin-action-mode').value==='window';$('bin-action-before').disabled=!active;$('bin-action-after').disabled=!active;
  try{const range=actionRange();$('bin-action-error').hidden=true;if(!range||!binState.data){binState.actions=null;drawBins();return;}
    const result=await api('/api/raw-bins/actions',{...binState.request,index:binState.current,before_s:range.before,after_s:range.after});
    if(version!==binState.actionVersion)return;binState.actions=result;drawBins();
  }catch(error){if(version!==binState.actionVersion)return;$('bin-action-error').hidden=false;$('bin-action-error').textContent=error.message;}
}
function actionYLimits(axis,autoMin,autoMax){
  const minInput=$(`bin-${axis}-min`),maxInput=$(`bin-${axis}-max`),error=$(`bin-${axis}-axis-error`);
  const min=minInput.value===''?null:Number(minInput.value),max=maxInput.value===''?null:Number(maxInput.value);
  const invalid=minInput.validity.badInput||maxInput.validity.badInput||(min!==null&&!Number.isFinite(min))||(max!==null&&!Number.isFinite(max))||(min!==null&&max!==null&&min>=max);
  error.hidden=!invalid;error.textContent=invalid?'Enter valid limits with the maximum greater than the minimum.':'';
  for(const input of [minInput,maxInput])input.setAttribute('aria-invalid',String(invalid));
  if(invalid)return [autoMin,autoMax];
  const span=Math.max(.1,autoMax-autoMin);
  let lo=min??autoMin,hi=max??autoMax;
  if(lo>=hi){if(max===null)hi=lo+span;else lo=hi-span;}
  return [lo,hi];
}
function drawBinCommands(id,series,indices,axis){
  const {ctx,w,h}=canvas(id);const data=binState.data;if(!data||!w||!h)return;
  let range;try{range=actionRange();}catch{return;}
  const currentTime=binState.current*data.dt;
  const timeStart=range?currentTime-range.before:0,timeEnd=range?currentTime+range.after:Math.max(data.dt,data.duration_s);
  let lo=0,hi=0;for(const item of series){if(range){item.values.forEach((value,k)=>{const time=indices[k]*data.dt;if(time>=timeStart&&time<=timeEnd&&Number.isFinite(value)){lo=Math.min(lo,value);hi=Math.max(hi,value);}});}else{if(Number.isFinite(item.stats.min))lo=Math.min(lo,item.stats.min);if(Number.isFinite(item.stats.max))hi=Math.max(hi,item.stats.max);}}
  const span=Math.max(.1,hi-lo);lo-=span*.08;hi+=span*.08;
  [lo,hi]=actionYLimits(axis,lo,hi);$(id).setAttribute('aria-description',`Y-axis: ${lo} to ${hi}`);
  const decimals=Math.max(1,Math.min(6,Math.ceil(-Math.log10(hi-lo))+1));
  ctx.font='12px system-ui, sans-serif';
  const left=Math.max(52,Math.ceil(Math.max(ctx.measureText(lo.toFixed(decimals)).width,ctx.measureText(hi.toFixed(decimals)).width))+16),right=w-12,top=14,bottom=h-30;$(id).dataset.plotLeft=left;
  const map=(sample,value)=>[left+(sample*data.dt-timeStart)/Math.max(Number.EPSILON,timeEnd-timeStart)*(right-left),Number.isFinite(value)?bottom-(value-lo)/(hi-lo)*(bottom-top):NaN];ctx.fillStyle='#b0bac5';for(let j=0;j<=2;j++){const value=lo+(hi-lo)*j/2,y=map(0,value)[1];ctx.strokeStyle='#323840';ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(right,y);ctx.stroke();ctx.textAlign='right';ctx.fillText(value.toFixed(decimals),left-8,y+4);ctx.textAlign='left';}
  ctx.save();ctx.beginPath();ctx.rect(left,top,right-left,bottom-top);ctx.clip();for(const item of series)binLine(ctx,item.values.map((value,k)=>map(indices[k],value)),item.color,1.5);ctx.restore();
  const cursor=map(binState.current,0)[0];ctx.strokeStyle='#8bc99b';ctx.setLineDash([3,3]);ctx.beginPath();ctx.moveTo(cursor,top);ctx.lineTo(cursor,bottom);ctx.stroke();ctx.setLineDash([]);
  const startLabel=range?`${-range.before}s`:'0s',endLabel=range?`+${range.after}s`:`${data.duration_s.toFixed(1)}s`;ctx.fillText(startLabel,left,h-7);ctx.fillText(endLabel,Math.max(left,right-ctx.measureText(endLabel).width),h-7);
  if(range&&cursor>left+ctx.measureText(startLabel).width+12&&cursor<right-ctx.measureText(endLabel).width-12){ctx.fillStyle='#8bc99b';ctx.textAlign='center';ctx.fillText('0',cursor,h-7);ctx.textAlign='left';}
}
function drawBinCommandPanels(){
  const data=binState.data;if(!data){canvas('bin-forward-plot');canvas('bin-command-steer');return;}
  const window=$('bin-action-mode').value==='window'?binState.actions:null;const indices=window?.indices||data.indices;const commands=window?.commands||data.rows.map(r=>r.slice(13));const forward=window?.forward_velocity||data.forward_velocity;
  const velocity=[{values:commands.map(r=>r[0]),stats:data.stats[13],color:'#e7b17a'},{values:forward,stats:data.forward_stats,color:'#80cbd3'}];const steering=[{values:commands.map(r=>r[1]),stats:data.stats[14],color:'#e7b17a'}];
  if(data.columns===17){velocity.push({values:commands.map(r=>r[2]),stats:data.stats[15],color:'#bca0df'});steering.push({values:commands.map(r=>r[3]),stats:data.stats[16],color:'#bca0df'});}
  drawBinCommands('bin-forward-plot',velocity,indices,'velocity');drawBinCommands('bin-command-steer',steering,indices,'steering');$('bin-action-time').textContent=`Current: ${(binState.current*data.dt).toFixed(2)} s`;
  $('bin-command-value').textContent=`Command ${binValue(binState.row?.[13])}`;$('bin-forward-value').textContent=`Body x ${binValue(binState.forward)}`;$('bin-steer-value').textContent=`Command ${binValue(binState.row?.[14])}`;
  $('bin-unshifted-value').hidden=data.columns!==17;$('bin-unshifted-steer-value').hidden=data.columns!==17;if(data.columns===17){$('bin-unshifted-value').textContent=`Unshifted ${binValue(binState.row?.[15])}`;$('bin-unshifted-steer-value').textContent=`Unshifted ${binValue(binState.row?.[16])}`;}
}
function drawBins(){if(!['bins','syn-bins'].includes(state.tab))return;drawBinTrajectory();drawBinCommandPanels();if(!binState.data)return;binGroups.forEach((g,i)=>drawBinSignal(i,g));$('bin-frame-info').textContent=`Sample ${binState.current.toLocaleString()} / ${(binState.data.samples-1).toLocaleString()} · ${(binState.current*binState.data.dt).toFixed(2)} s`;}
bind('tab-bins',()=>selectTab('bins'));bind('bin-scan',scanBinFolder);bind('bin-apply',scanBinFolder);
bind('bin-collapse',()=>{binState.opened.clear();$('bin-filter').value='';for(const folder of $('bin-tree').querySelectorAll('details'))folder.open=false;});
$('bin-filter').addEventListener('input',renderBinTree);
$('bin-env').addEventListener('change',async()=>{if(!binState.data)return;const current=binState.current,version=binState.loadVersion+1;try{await loadRawBin(binState.data.path,number('bin-env'));if(version===binState.loadVersion)await setBinFrame(current);}catch(e){status(e.message,true);}});
$('bin-frame').addEventListener('input',()=>{stopBinPlayback();setBinFrame(number('bin-frame'));});$('bin-current').addEventListener('input',()=>{if($('bin-current').value==='')return;stopBinPlayback();setBinFrame(number('bin-current'));});
bind('bin-play',()=>{if(!binState.data)return;if(binState.playing){stopBinPlayback();return;}if(binState.current>=binState.data.samples-1)setBinFrame(0);binState.playing=true;$('bin-play').textContent='Ⅱ';$('bin-play').setAttribute('aria-label','Pause raw bin');});
let binTick=performance.now(),binCarry=0;setInterval(()=>{const now=performance.now(),elapsed=now-binTick;binTick=now;if(!binState.playing||!binState.data||!['bins','syn-bins'].includes(state.tab)){binCarry=0;return;}binCarry+=elapsed/1000*number('bin-speed')/binState.data.dt;const steps=Math.floor(binCarry);if(!steps)return;binCarry-=steps;const next=binState.current+steps;setBinFrame(next);if(next>=binState.data.samples-1)stopBinPlayback();},100);
bind('bin-reset',()=>{binState.zoom=1;binState.pan=[0,0];drawBins();});
bind('bin-toggle-browser',()=>{const closed=$('bins-workspace').classList.toggle('bin-browser-closed');$('bin-toggle-browser').textContent=closed?'Show browser':'Hide browser';$('bin-toggle-browser').setAttribute('aria-expanded',String(!closed));drawBins();});
const binPlot=$('bin-trajectory');let binDrag=null;binPlot.addEventListener('wheel',event=>{event.preventDefault();binState.zoom=Math.max(.1,Math.min(30,binState.zoom*Math.exp(-event.deltaY*.001)));drawBins();},{passive:false});binPlot.addEventListener('pointerdown',event=>{binDrag=[event.clientX,event.clientY,...binState.pan];binPlot.setPointerCapture(event.pointerId);});binPlot.addEventListener('pointermove',event=>{if(binDrag){binState.pan=[binDrag[2]+event.clientX-binDrag[0],binDrag[3]+event.clientY-binDrag[1]];drawBins();}});for(const event of ['pointerup','pointercancel'])binPlot.addEventListener(event,()=>binDrag=null);
new ResizeObserver(drawBins).observe(binPlot);
document.addEventListener('keydown',event=>{if(!['bins','syn-bins'].includes(state.tab)||['INPUT','SELECT','TEXTAREA','BUTTON'].includes(event.target.tagName))return;if(event.code==='Space'){event.preventDefault();$('bin-play').click();}if(event.code==='ArrowRight'||event.code==='ArrowLeft'){event.preventDefault();stopBinPlayback();setBinFrame(binState.current+(event.code==='ArrowRight'?1:-1));}});
$('bin-root').value=localStorage.getItem('optcar-bin-folder')||'';

for(const id of ['bin-forward-plot','bin-command-steer'])$(id).addEventListener('click',event=>{if(!binState.data)return;let range;try{range=actionRange();}catch{return;}const rect=event.currentTarget.getBoundingClientRect();const left=Number(event.currentTarget.dataset.plotLeft)||52;const fraction=Math.max(0,Math.min(1,(event.clientX-rect.left-left)/(rect.width-left-12)));const time=range?binState.current*binState.data.dt-range.before+fraction*(range.before+range.after):fraction*binState.data.duration_s;stopBinPlayback();setBinFrame(Math.round(time/binState.data.dt));});
$('bin-action-mode').addEventListener('change',()=>{binState.actions=null;drawBins();refreshBinActions();});
for(const id of ['bin-action-before','bin-action-after'])$(id).addEventListener('input',()=>{binState.actions=null;drawBins();refreshBinActions();});

for(const axis of ['velocity','steering'])for(const bound of ['min','max'])$(`bin-${axis}-${bound}`).addEventListener('input',drawBins);

bind('bin-tune-dbm',()=>{if(!binState.data)return;$('dbm-tool').src='/dbm/?path='+encodeURIComponent(binState.data.path)+'&folder='+encodeURIComponent(binState.scan?.root||$('bin-root').value)+'&frame='+binState.current;selectTab('dbm');});
