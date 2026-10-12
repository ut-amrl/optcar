// Ported from dbm_estimator/static/app.js. Plot styling follows OptCar.
const originalPlotlyReact = Plotly.react.bind(Plotly);
Plotly.react = (target, traces, layout, config) => originalPlotlyReact(target, traces, {
  ...layout, paper_bgcolor:'#1c2025', plot_bgcolor:'#1c2025', font:{family:'system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif',color:'#b0bac5',size:12}, title:{text:typeof layout.title==='string'?layout.title:layout.title?.text,font:{size:14,color:'#edf0f3'}},
  xaxis:{...layout.xaxis,gridcolor:'#343c46',zerolinecolor:'#667483',automargin:true},
  yaxis:{...layout.yaxis,gridcolor:'#343c46',zerolinecolor:'#667483',automargin:true},
}, config);
const $ = (id) => document.getElementById(id);

const state = {
  session_id: null,
  n_frames: 0,
  dt: 0.02,
  dbmParams: null,
  rollout: null,
  lastFrameData: null,
  lastTrajectoryData: null,
  segmentYamlPath: null,
  refSpecs: [],
  referenceTrajectory: null,
  activeTab: "traj",
};

function showLoader(msg) {
  $("loader_msg").textContent = msg || "Loading...";
  $("loader").classList.remove("hidden");
}
function hideLoader() { $("loader").classList.add("hidden"); }

function readCachedObject(key) {
  try {
    return JSON.parse(localStorage.getItem(key) || "{}") || {};
  } catch {
    return {};
  }
}

async function postJSON(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({error: r.statusText}));
    throw new Error(err.error || r.statusText);
  }
  return r.json();
}

async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) {
    const err = await r.json().catch(() => ({error: r.statusText}));
    throw new Error(err.error || r.statusText);
  }
  return r.json();
}

function paramInputId(name) { return `ref_param_${name}`; }

function formatParamDefault(param) {
  if (param.kind === "list") return JSON.stringify(param.default);
  if (param.default === null || param.default === undefined) return "";
  return String(param.default);
}

async function loadReferenceSpecs() {
  $("ref_traj_status").textContent = "";
  try {
    const res = await getJSON("/dbm/api/reference_trajectories");
    state.refSpecs = res.trajectories || [];
    const sel = $("ref_traj_select");
    sel.innerHTML = "";
    for (const spec of state.refSpecs) {
      const o = document.createElement("option");
      o.value = spec.name;
      o.textContent = spec.name;
      sel.appendChild(o);

    }
    renderReferenceParams();
  } catch (e) {
    $("ref_traj_status").textContent = "Error: " + e.message;
  }
}

function selectedReferenceSpec() {
  const name = $("ref_traj_select").value;
  return state.refSpecs.find((spec) => spec.name === name);
}

function renderReferenceParams() {
  const spec = selectedReferenceSpec();
  const wrap = $("ref_params");
  wrap.innerHTML = "";
  if (!spec) return;

  for (const param of spec.params) {
    const label = document.createElement("label");
    label.textContent = param.name;

    const input = document.createElement("input");
    input.id = paramInputId(param.name);
    input.dataset.kind = param.kind;

    if (param.kind === "bool") {
      input.type = "checkbox";
      input.checked = Boolean(param.default);
    } else if (param.kind === "int" || param.kind === "float") {
      input.type = "number";
      input.step = param.kind === "int" ? "1" : "any";
      input.value = formatParamDefault(param);
    } else {
      input.type = "text";
      input.value = formatParamDefault(param);
    }

    label.appendChild(input);
    wrap.appendChild(label);
  }
}

function readReferenceParams() {
  const spec = selectedReferenceSpec();
  const params = {};
  if (!spec) return params;
  for (const param of spec.params) {
    const input = $(paramInputId(param.name));
    if (!input) continue;
    params[param.name] = param.kind === "bool" ? input.checked : input.value;
  }
  return params;
}

const binLibrary={entries:[],opened:new Set(),selected:null};
function renderBinLibrary(){
  const count=OptCarBinExplorer.render({tree:$("dbm-bin-tree"),entries:binLibrary.entries,filter:$("dbm-bin-filter").value,opened:binLibrary.opened,selected:binLibrary.selected,onSelect:path=>loadDbmBin(path),formatDuration:seconds=>`${seconds.toFixed(1)}s`});
  $("dbm-bin-count").textContent=`${count} / ${binLibrary.entries.length} bins`;
}
$("dbm-bin-filter").oninput=renderBinLibrary;
$("dbm-bin-collapse").onclick=()=>{binLibrary.opened.clear();$("dbm-bin-filter").value='';renderBinLibrary();};
$("dbm-toggle-browser").onclick=()=>{
  const closed=$("dbm-library-layout").classList.toggle('browser-closed');
  $("dbm-toggle-browser").textContent=closed?'Show browser':'Hide browser';
  $("dbm-toggle-browser").setAttribute('aria-expanded',String(!closed));
};
$("btn_inspect").onclick=async()=>{
  const path=$("dataset_path").value.trim();if(!path)return;
  showLoader('Scanning raw bins…');$("inspect_status").textContent='';
  try{
    const res=await postJSON('/dbm/api/inspect',{dataset_path:path});
    binLibrary.entries=res.bins;binLibrary.opened.clear();$("dbm-bin-filter").value='';
    const selected=res.bins.find(b=>b.path===binLibrary.selected)||res.bins.find(b=>!b.error);
    if(selected)OptCarBinExplorer.expandTo(binLibrary.opened,selected.relative_path);
    $("dataset_path").value=res.root;localStorage.setItem('optcar-dbm-folder',res.root);
    renderBinLibrary();$("inspect_status").textContent=`${res.bins.length} bins found.`;
    return res;
  }catch(error){$("inspect_status").textContent='Error: '+error.message;}
  finally{hideLoader();}
};

async function loadDbmBin(bin_path) {
  showLoader("Loading bin...");
  $("load_status").textContent = "";
  try {
    const r = await postJSON("/dbm/api/load", {bin_path});
    frameVersion++;
    state.session_id = r.session_id;
    state.n_frames = r.n_frames;
    state.dt = r.dt;
    state.rollout = null;
    state.lastFrameData = null;
    state.lastTrajectoryData = null;
    state.segmentYamlPath = r.segment_yaml_path;
    $("frame_slider").max = r.n_frames - 1;
    const initialFrame=bin_path===new URLSearchParams(location.search).get("path")?Math.max(0,Math.min(r.n_frames-1,Number(new URLSearchParams(location.search).get("frame")||0))):0;
    $("frame_slider").value = initialFrame;
    $("frame_input").value = initialFrame;
    $("frame_input").max = r.n_frames - 1;
    $("traj_start_slider").max = r.n_frames - 1;
    $("traj_start_slider").value = 0;
    $("traj_start_input").value = 0;
    $("traj_start_input").max = r.n_frames - 1;
    $("traj_end_slider").max = r.n_frames - 1;
    $("traj_end_slider").value = r.n_frames - 1;
    $("traj_end_input").value = r.n_frames - 1;
    $("traj_end_input").max = r.n_frames - 1;
    if (r.saved_segment) {
      $("traj_start_slider").value = r.saved_segment.start_idx;
      $("traj_start_input").value = r.saved_segment.start_idx;
      $("traj_end_slider").value = r.saved_segment.end_idx;
      $("traj_end_input").value = r.saved_segment.end_idx;
      $("segment_status").textContent = `loaded ${r.segment_yaml_path}`;
    } else {
      $("segment_status").textContent = r.segment_yaml_path;
    }
    $("view_card").classList.remove("hidden");
    $("load_status").textContent =
      `${r.n_frames} frames, dt=${r.dt}s, duration=${r.duration.toFixed(2)}s, cols=${r.n_cols}`;
    binLibrary.selected=bin_path;
    $("dbm-recording").textContent=bin_path.split("/").pop();
    renderBinLibrary();
    setActiveTab("dbm");
    await refreshTrajectory();
    await refreshFrame();
  } catch (e) {
    $("load_status").textContent = "Error: " + e.message;
  } finally {
    hideLoader();
  }
};


function setActiveTab(tab) {
  state.activeTab = tab;
  $("tab_traj").classList.toggle("active", tab === "traj");
  $("tab_dbm").classList.toggle("active", tab === "dbm");
  $("traj_tab_panel").classList.toggle("hidden", tab !== "traj");
  $("dbm_tab_panel").classList.toggle("hidden", tab !== "dbm");
  setTimeout(() => {
    if (tab === "traj") Plotly.Plots.resize($("bag_xy_plot"));
    if (tab === "dbm") {
      Plotly.Plots.resize($("xy_plot"));
      Plotly.Plots.resize($("vel_plot"));
      Plotly.Plots.resize($("steer_plot"));
    }
  }, 0);
}

function clampInt(value, minValue, maxValue) {
  const parsed = parseInt(value);
  if (Number.isNaN(parsed)) return minValue;
  return Math.max(minValue, Math.min(parsed, maxValue));
}

function readTrajectoryRange(changed) {
  const maxFrame = Math.max(0, state.n_frames - 1);
  let start = clampInt($("traj_start_input").value, 0, maxFrame);
  let end = clampInt($("traj_end_input").value, 0, maxFrame);
  if (start > end) {
    if (changed === "start") end = start;
    else start = end;
  }
  $("traj_start_input").value = start;
  $("traj_start_slider").value = start;
  $("traj_end_input").value = end;
  $("traj_end_slider").value = end;
  return {start, end};
}

async function refreshTrajectory() {
  if (!state.session_id) return;
  const {start, end} = readTrajectoryRange();
  const r = await postJSON("/dbm/api/trajectory", {
    session_id: state.session_id,
    start_idx: start,
    end_idx: end,
    max_points: 9000,
  });
  state.lastTrajectoryData = r;
  $("traj_info").textContent =
    `frames ${r.start_idx}-${r.end_idx}, t=${r.t0.toFixed(2)}-${r.t1.toFixed(2)}s, ${r.points} plotted point(s)`;
  drawBagXY(r);
}

function drawBagXY(r) {
  const traces = [
    {
      x: r.x,
      y: r.y,
      mode: "lines",
      line: {color: "#80cbd3", width: 2},
      name: "selected trajectory",
    },
  ];
  if (r.x.length > 0) {
    traces.push(
      {
        x: [r.x[0]], y: [r.y[0]], mode: "markers",
        marker: {size: 10, color: "#8bc99b"},
        name: "start",
      },
      {
        x: [r.x[r.x.length - 1]], y: [r.y[r.y.length - 1]], mode: "markers",
        marker: {size: 10, color: "#f29a90"},
        name: "end",
      },
    );
  }

  if (state.referenceTrajectory) {
    traces.push({
      x: state.referenceTrajectory.x,
      y: state.referenceTrajectory.y,
      mode: "lines",
      line: {color: state.referenceTrajectory.color || "#f59e0b", width: 3, dash: "dash"},
      name: "Reference Trajectory",
    });
  }

  const layout = {
    title: "Top-down XY trajectory",
    xaxis: {title: "x [m]", scaleanchor: "y", scaleratio: 1, zeroline: true},
    yaxis: {title: "y [m]", zeroline: true},
    margin: {t: 40, r: 16, b: 48, l: 58},
    dragmode: "pan",
    hovermode: "closest",
    legend: {orientation: "h", x: 0.5, xanchor: "center", y: -0.12},
  };
  Plotly.react("bag_xy_plot", traces, layout,
    {scrollZoom: true, displaylogo: false, responsive: true});
}

let frameVersion=0;
async function refreshFrame() {
  if (!state.session_id) return;
  const version=++frameVersion;
  try {
  const frame = parseInt($("frame_input").value);
  const horizon = parseInt($("horizon").value);
  const cmd_shift_s = parseFloat($("cmd_shift").value) || 0.0;

  const r = await postJSON("/dbm/api/frame", {
    session_id: state.session_id,
    frame_idx: frame,
    horizon: horizon,
    cmd_shift_s: cmd_shift_s,
  });
  if(version!==frameVersion)return;
  state.lastFrameData = r;
  $("frame_info").textContent =
    `frame ${r.frame_idx}/${r.n_frames - 1} (t=${(r.frame_idx * state.dt).toFixed(2)}s, end=${r.end_idx})`;

  let rollout=null;
  if (state.dbmParams) {
    try {
      rollout = await postJSON("/dbm/api/rollout", {
        session_id: state.session_id,
        frame_idx: frame,
        horizon: horizon,
        cmd_shift_s: cmd_shift_s,
        params: state.dbmParams,
      });
    } catch (error) {
      $("rollout_status").textContent="Error: "+error.message;
    }
  }

  if(version!==frameVersion)return;
  state.rollout=rollout;
  drawXY(r);
  drawVel(r);
  drawSteer(r);
  } catch(error){if(version===frameVersion)$("rollout_status").textContent="Error: "+error.message;}
}

let _trajTimer = null;
function scheduleTrajectoryRefresh(changed) {
  readTrajectoryRange(changed);
  clearTimeout(_trajTimer);
  _trajTimer = setTimeout(refreshTrajectory, 35);
}

// Forward is up; the reversed lateral axis puts positive body-y on the left.

function drawXY(r) {
  const traces = [
    {
      x: r.by, y: r.bx,
      mode: "lines+markers",
      line: {color: "#80cbd3", width: 2},
      marker: {size: 3, color: "#80cbd3"},
      name: "measured",
    },
    {
      x: [0], y: [0], mode: "markers",
      marker: {size: 12, color: "#8bc99b"},
      showlegend: false,
    },
    {
      // heading arrow: +x_body (forward) maps to +plot_y (up)
      x: [0, 0], y: [0, 0.3], mode: "lines",
      line: {color: "#8bc99b", width: 3},
      showlegend: false,
    },
  ];

  if (state.rollout) {
    traces.push({
      x: state.rollout.by, y: state.rollout.bx,
      mode: "lines",
      line: {color: "#e7b17a", width: 2, dash: "dash"},
      name: "DBM rollout",
    });
  }

  const layout = {
    title: "Body-frame trajectory",
    xaxis: {title: "y_body [m]  left+", scaleanchor: "y", scaleratio: 1, zeroline: true, autorange: "reversed"},
    yaxis: {title: "x_body [m]  fwd+", zeroline: true},
    margin: {t: 40, r: 10, b: 70, l: 50},
    dragmode: "pan",
    hovermode: "closest",
    showlegend: !!state.rollout,
    legend: {orientation: "h", y: -0.16, yanchor:"top", font:{size:12}},
  };
  Plotly.react("xy_plot", traces, layout,
    {scrollZoom: true, displaylogo: false, responsive: true});
}

function drawVel(r) {
  const layout = {
    title: "Forward velocity",
    xaxis: {title: "Time (s)"},
    yaxis: {title: "m/s"},
    margin: {t: 40, r: 10, b: 70, l: 50},
    legend: {orientation: "h", y: -0.45, yanchor:"top", font:{size:12}},
  };
  Plotly.react("vel_plot", [
    {x: r.t, y: r.cmd_v, mode: "lines", name: "Command", line: {color: "#f29a90"}},
    {x: r.t, y: r.v_long, mode: "lines", name: "Body x", line: {color: "#80cbd3", dash: "dot"}},
  ], layout, {scrollZoom: true, displaylogo: false, responsive: true});
}

function drawSteer(r) {
  const layout = {
    title: "Steering / yaw rate",
    xaxis: {title: "Time (s)"},
    yaxis: {title: "rad or rad/s"},
    margin: {t: 40, r: 10, b: 70, l: 50},
    legend: {orientation: "h", y: -0.45, yanchor:"top", font:{size:12}},
  };
  Plotly.react("steer_plot", [
    {x: r.t, y: r.cmd_s, mode: "lines", name: "Steering", line: {color: "#bca0df"}},
    {x: r.t, y: r.yaw_rate, mode: "lines", name: "Yaw rate", line: {color: "#80cbd3", dash: "dot"}},
  ], layout, {scrollZoom: true, displaylogo: false, responsive: true});
}

function readParams() {
  return {
    len_f:               parseFloat($("p_len_f").value),
    len_r:               parseFloat($("p_len_r").value),
    mass:                parseFloat($("p_mass").value),
    inertia:             parseFloat($("p_inertia").value),
    max_steering_angle:  parseFloat($("p_max_steer").value),
    C_tire_f:            parseFloat($("p_C_tire_f").value),
    C_tire_r:            parseFloat($("p_C_tire_r").value),
    C_throttle_a:        parseFloat($("p_C_throttle_a").value),
    C_rollf:             parseFloat($("p_C_rollf").value),
    C_drag:              parseFloat($("p_C_drag").value),
    wheel_time_constant: parseFloat($("p_wheel_tau").value),
    min_wheel_velocity:  parseFloat($("p_min_wheel").value),
    max_wheel_velocity:  parseFloat($("p_max_wheel").value),
  };
}

$("btn_rollout").onclick = async () => {
  if (!state.session_id) return;
  $("rollout_status").textContent = "Running...";
  $("btn_rollout").disabled = true;
  const version=++frameVersion;
  try {
    state.dbmParams = readParams();
    const frame = parseInt($("frame_input").value);
    const horizon = parseInt($("horizon").value);
    const cmd_shift_s = parseFloat($("cmd_shift").value) || 0.0;
    const rollout = await postJSON("/dbm/api/rollout", {
      session_id: state.session_id,
      frame_idx: frame,
      horizon: horizon,
      cmd_shift_s: cmd_shift_s,
      params: state.dbmParams,
    });
    if(version!==frameVersion)return;
    state.rollout=rollout;
    $("rollout_status").textContent = `${state.rollout.bx.length} pts`;
    if (state.lastFrameData) drawXY(state.lastFrameData);
  } catch (e) {
    $("rollout_status").textContent = "Error: " + e.message;
    state.rollout = null;
    if(state.lastFrameData)drawXY(state.lastFrameData);
  } finally {
    $("btn_rollout").disabled = false;
  }
};

let _refreshTimer = null;
function scheduleRefresh() {
  clearTimeout(_refreshTimer);
  _refreshTimer = setTimeout(refreshFrame, 30);
}
$("frame_slider").addEventListener("input", (e) => {
  $("frame_input").value = e.target.value;
  scheduleRefresh();
});
$("frame_input").addEventListener("input", (e) => {
  $("frame_slider").value = e.target.value;
  scheduleRefresh();
});
$("btn_refresh").onclick = refreshFrame;
$("horizon").addEventListener("input", scheduleRefresh);
$("cmd_shift").addEventListener("input", scheduleRefresh);

$("tab_traj").onclick = () => setActiveTab("traj");
$("tab_dbm").onclick = () => setActiveTab("dbm");
$("ref_traj_select").addEventListener("change", renderReferenceParams);
$("ref_traj_color").addEventListener("input", (e) => {
  if (!state.referenceTrajectory) return;
  state.referenceTrajectory.color = e.target.value;
  if (state.lastTrajectoryData) drawBagXY(state.lastTrajectoryData);
});
$("btn_load_ref_traj").onclick = async () => {
  const spec = selectedReferenceSpec();
  if (!spec) return;
  $("ref_traj_status").textContent = "Loading...";
  $("btn_load_ref_traj").disabled = true;
  try {
    state.referenceTrajectory = await postJSON("/dbm/api/reference_trajectory", {
      trajectory_name: spec.name,
      params: readReferenceParams(),
      max_points: 9000,
    });
    state.referenceTrajectory.color = $("ref_traj_color").value;
    $("ref_traj_status").textContent =
      `${state.referenceTrajectory.label}: ${state.referenceTrajectory.points} point(s)`;
    if (state.lastTrajectoryData) drawBagXY(state.lastTrajectoryData);
  } catch (e) {
    $("ref_traj_status").textContent = "Error: " + e.message;
  } finally {
    $("btn_load_ref_traj").disabled = false;
  }
};
$("btn_clear_ref_traj").onclick = () => {
  state.referenceTrajectory = null;
  $("ref_traj_status").textContent = "";
  if (state.lastTrajectoryData) drawBagXY(state.lastTrajectoryData);
};
$("btn_save_segment").onclick = async () => {
  if (!state.session_id) return;
  $("segment_status").textContent = "Saving...";
  const {start, end} = readTrajectoryRange();
  try {
    const res = await postJSON("/dbm/api/save_segment", {
      session_id: state.session_id,
      start_idx: start,
      end_idx: end,
    });
    state.segmentYamlPath = res.yaml_path;
    $("segment_status").textContent = res.yaml_path;
  } catch (e) {
    $("segment_status").textContent = "Error: " + e.message;
  }
};
$("btn_load_segment").onclick = async () => {
  if (!state.session_id) return;
  $("segment_status").textContent = "Loading...";
  try {
    const res = await postJSON("/dbm/api/load_segment", {session_id: state.session_id});
    $("traj_start_input").value = res.start_idx;
    $("traj_start_slider").value = res.start_idx;
    $("traj_end_input").value = res.end_idx;
    $("traj_end_slider").value = res.end_idx;
    state.segmentYamlPath = res.yaml_path;
    $("segment_status").textContent = res.yaml_path;
    await refreshTrajectory();
  } catch (e) {
    $("segment_status").textContent = "Error: " + e.message;
  }
};
$("traj_start_slider").addEventListener("input", (e) => {
  $("traj_start_input").value = e.target.value;
  scheduleTrajectoryRefresh("start");
});
$("traj_start_input").addEventListener("input", (e) => {
  $("traj_start_slider").value = e.target.value;
  scheduleTrajectoryRefresh("start");
});
$("traj_end_slider").addEventListener("input", (e) => {
  $("traj_end_input").value = e.target.value;
  scheduleTrajectoryRefresh("end");
});
$("traj_end_input").addEventListener("input", (e) => {
  $("traj_end_slider").value = e.target.value;
  scheduleTrajectoryRefresh("end");
});

loadReferenceSpecs();

// Remember the library independently of the recording/transformer controls.
$("dataset_path").value ||= localStorage.getItem("optcar-dbm-folder") || localStorage.getItem("optcar-bin-folder") || "";

const presetInputIds={max_steering_angle:'max_steer',wheel_time_constant:'wheel_tau',min_wheel_velocity:'min_wheel',max_wheel_velocity:'max_wheel'};
function presetLocation(){
  const folder=$("preset_folder").value.trim(),filename=$("preset_filename").value.trim();
  if(!folder||!filename)throw new Error('Enter both the DBM config folder and filename.');
  return {folder,filename};
}
$("load_dbm_preset").onclick=async()=>{
  try{
    const query=new URLSearchParams(presetLocation());
    const response=await fetch(`/api/preset?${query}`),result=await response.json();
    if(!response.ok)throw new Error(result.error);
    for(const [key,value]of Object.entries(result.params))$("p_"+(presetInputIds[key]||key)).value=value;
    state.dbmParams=null;state.rollout=null;if(state.lastFrameData)drawXY(state.lastFrameData);
    $("preset_status").textContent=`Loaded ${result.path}. Click Set & Rollout to apply.`;
  }catch(error){$("preset_status").textContent=error.message;}
};
$("save_dbm_preset").onclick=async()=>{
  try{
    const response=await fetch('/api/preset',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({...presetLocation(),params:readParams()})});
    const result=await response.json();if(!response.ok)throw new Error(result.error);
    $("preset_status").textContent=`Saved ${result.path}`;
  }catch(error){$("preset_status").textContent=error.message;}
};

if($("dataset_path").value){
  const initialPath=new URLSearchParams(location.search).get('path');
  $("btn_inspect").onclick().then(res=>{
    const bin=res?.bins.find(b=>b.path===initialPath&&!b.error);
    if(bin)return loadDbmBin(bin.path);
  });
}
new ResizeObserver(()=>{
  for(const id of ['xy_plot','vel_plot','steer_plot','bag_xy_plot'])if($(id).querySelector('.plot-container'))Plotly.Plots.resize($(id));
}).observe($("dbm-library-main"));
