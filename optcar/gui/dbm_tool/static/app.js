// Ported from WheeledLab dbm_estimator/static/app.js. Plot styling follows OptCar.
const originalPlotlyReact = Plotly.react.bind(Plotly);
Plotly.react = (target, traces, layout, config) => originalPlotlyReact(target, traces, {
  ...layout, paper_bgcolor:'#1c2025', plot_bgcolor:'#1c2025', font:{family:'system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif',color:'#b0bac5',size:12}, title:{text:typeof layout.title==='string'?layout.title:layout.title?.text,font:{size:14,color:'#edf0f3'}},
  xaxis:{...layout.xaxis,gridcolor:'#343c46',zerolinecolor:'#667483',automargin:true},
  yaxis:{...layout.yaxis,gridcolor:'#343c46',zerolinecolor:'#667483',automargin:true},
}, config);
const $ = (id) => document.getElementById(id);
const AGG_LIMITS_CACHE_KEY = "optcar.dbm.aggregate_limits";
const AGG_DISPLAY_CACHE_KEY = "optcar.dbm.aggregate_display";
const MAX_TRACKING_ERROR_SEGMENT_POINTS = 3000;
const MAX_TRACKING_ERROR_REFERENCE_POINTS = 3000;

const state = {
  session_id: null,
  n_frames: 0,
  dt: 0.02,
  dbmParams: null,
  rollout: null,
  lastFrameData: null,
  lastTrajectoryData: null,
  segmentYamlPath: null,
  aggregateSegments: [],
  refSpecs: [],
  referenceTrajectory: null,
  aggregateReferenceTrajectory: null,
  activeTab: "traj",
  activePage: "bag",
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

function writeCachedObject(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value || {}));
  } catch {
    // Local storage may be unavailable in private or restricted browser contexts.
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
function aggParamInputId(name) { return `agg_ref_param_${name}`; }

function formatParamDefault(param) {
  if (param.kind === "list") return JSON.stringify(param.default);
  if (param.default === null || param.default === undefined) return "";
  return String(param.default);
}

async function loadReferenceSpecs() {
  $("ref_traj_status").textContent = "";
  $("agg_ref_traj_status").textContent = "";
  try {
    const res = await getJSON("/dbm/api/reference_trajectories");
    state.refSpecs = res.trajectories || [];
    const sel = $("ref_traj_select");
    sel.innerHTML = "";
    const aggSel = $("agg_ref_traj_select");
    aggSel.innerHTML = "";
    for (const spec of state.refSpecs) {
      const o = document.createElement("option");
      o.value = spec.name;
      o.textContent = spec.name;
      sel.appendChild(o);

      const aggOpt = document.createElement("option");
      aggOpt.value = spec.name;
      aggOpt.textContent = spec.name;
      aggSel.appendChild(aggOpt);
    }
    renderReferenceParams();
    renderAggregateReferenceParams();
  } catch (e) {
    $("ref_traj_status").textContent = "Error: " + e.message;
    $("agg_ref_traj_status").textContent = "Error: " + e.message;
  }
}

function selectedReferenceSpec() {
  const name = $("ref_traj_select").value;
  return state.refSpecs.find((spec) => spec.name === name);
}

function selectedAggregateReferenceSpec() {
  const name = $("agg_ref_traj_select").value;
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

function renderAggregateReferenceParams() {
  const spec = selectedAggregateReferenceSpec();
  const wrap = $("agg_ref_params");
  wrap.innerHTML = "";
  if (!spec) return;

  for (const param of spec.params) {
    const label = document.createElement("label");
    label.textContent = param.name;

    const input = document.createElement("input");
    input.id = aggParamInputId(param.name);
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

function readAggregateReferenceParams() {
  const spec = selectedAggregateReferenceSpec();
  const params = {};
  if (!spec) return params;
  for (const param of spec.params) {
    const input = $(aggParamInputId(param.name));
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
  setPageBag();
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

function setPageBag(){
  state.activePage='bag';
}

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

function appendAggregateYaml(path) {
  if (!path) return;
  const existing = readAggregateRows();
  if (!existing.some((row) => row.yaml_path === path)) {
    addAggregateRow(path, "");
  }
}

function addAggregateRow(yamlPath = "", legend = "", color = undefined, visible = true, lineWidth = 2) {
  const row = document.createElement("div");
  row.className = "aggregate-grid aggregate-row";

  const pathInput = document.createElement("input");
  pathInput.type = "text";
  pathInput.className = "aggregate-yaml";
  pathInput.value = yamlPath;

  const legendInput = document.createElement("input");
  legendInput.type = "text";
  legendInput.className = "aggregate-legend";
  legendInput.value = legend;
  legendInput.oninput = syncAggregateSegmentSettingsFromRows;

  const colorInput = document.createElement("input");
  colorInput.type = "color";
  colorInput.className = "aggregate-color";
  colorInput.value = color || nextAggregateColor();
  colorInput.oninput = syncAggregateSegmentSettingsFromRows;

  const widthInput = document.createElement("input");
  widthInput.type = "number";
  widthInput.className = "aggregate-line-width";
  widthInput.min = "0.1";
  widthInput.step = "0.5";
  widthInput.value = Number.isFinite(Number(lineWidth)) && Number(lineWidth) > 0 ? lineWidth : 2;
  widthInput.oninput = syncAggregateSegmentSettingsFromRows;

  const visibleInput = document.createElement("input");
  visibleInput.type = "checkbox";
  visibleInput.className = "aggregate-visible";
  visibleInput.checked = visible !== false;
  visibleInput.title = "Show in plot and legend";
  visibleInput.onchange = syncAggregateSegmentSettingsFromRows;

  const removeBtn = document.createElement("button");
  removeBtn.type = "button";
  removeBtn.className = "aggregate-remove";
  removeBtn.textContent = "x";
  removeBtn.onclick = () => row.remove();

  row.appendChild(pathInput);
  row.appendChild(legendInput);
  row.appendChild(colorInput);
  row.appendChild(widthInput);
  row.appendChild(visibleInput);
  row.appendChild(removeBtn);
  $("aggregate_rows").appendChild(row);
}

function nextAggregateColor() {
  const palette = ["#2563eb", "#f97316", "#16a34a", "#f29a90", "#bca0df", "#0891b2", "#ca8a04"];
  return palette[document.querySelectorAll(".aggregate-row").length % palette.length];
}

function readAggregateRows() {
  return Array.from(document.querySelectorAll(".aggregate-row"))
    .map((row) => ({
      yaml_path: row.querySelector(".aggregate-yaml").value.trim(),
      legend: row.querySelector(".aggregate-legend").value.trim(),
      color: row.querySelector(".aggregate-color").value,
      line_width: Math.max(0.1, Number(row.querySelector(".aggregate-line-width")?.value) || 2),
      visible: row.querySelector(".aggregate-visible")?.checked !== false,
    }))
    .filter((row) => row.yaml_path);
}

function syncAggregateSegmentSettingsFromRows() {
  if (!state.aggregateSegments.length) return;
  const rows = readAggregateRows();
  state.aggregateSegments = state.aggregateSegments.map((seg, idx) => {
    let row = rows[idx];
    if (!row || row.yaml_path !== seg.yaml_path) {
      row = rows.find((candidate) => candidate.yaml_path === seg.yaml_path);
    }
    if (!row) return seg;
    const fallbackName = seg.bin_name ? seg.bin_name.replace(/\.bin$/i, "") : (seg.name || "segment");
    return {
      ...seg,
      name: row.legend || fallbackName,
      color: row.color,
      visible: row.visible !== false,
      line_width: row.line_width,
    };
  });
  updateAggregateStatus();
  drawAggregateXY();
}

function updateAggregateStatus(errors = []) {
  const total = state.aggregateSegments.length;
  const shown = state.aggregateSegments.filter((seg) => seg.visible !== false).length;
  const errText = errors.length ? `, ${errors.length} error(s)` : "";
  $("aggregate_status").textContent = `${total} segment(s), ${shown} shown${errText}`;
}

function readOptionalNumber(id) {
  const text = $(id).value.trim();
  if (!text) return null;
  const value = Number(text);
  return Number.isFinite(value) ? value : null;
}

function currentAggregatePlotLimits() {
  return {
    x_min: readOptionalNumber("agg_limit_x_min"),
    x_max: readOptionalNumber("agg_limit_x_max"),
    y_min: readOptionalNumber("agg_limit_y_min"),
    y_max: readOptionalNumber("agg_limit_y_max"),
  };
}

function applyAggregatePlotLimits(limits) {
  const values = {...readCachedObject(AGG_LIMITS_CACHE_KEY), ...(limits || {})};
  $("agg_limit_x_min").value = values.x_min ?? "";
  $("agg_limit_x_max").value = values.x_max ?? "";
  $("agg_limit_y_min").value = values.y_min ?? "";
  $("agg_limit_y_max").value = values.y_max ?? "";
}

function cacheAggregatePlotLimits() {
  writeCachedObject(AGG_LIMITS_CACHE_KEY, currentAggregatePlotLimits());
}

function applyAxisRange(axis, minValue, maxValue) {
  if (Number.isFinite(minValue) && Number.isFinite(maxValue) && minValue < maxValue) {
    axis.range = [minValue, maxValue];
  }
}

function currentAggregateDisplayOptions() {
  return {
    show_all: $("agg_show_all_display").checked,
    show_grid: $("agg_show_grid").checked,
    show_axis_labels: $("agg_show_axis_labels").checked,
    show_axes: $("agg_show_axes").checked,
    show_tick_labels: $("agg_show_tick_labels").checked,
    show_heading: $("agg_show_heading").checked,
    show_legend: $("agg_show_legend").checked,
  };
}

function applyAggregateDisplayOptions(options) {
  const values = {...readCachedObject(AGG_DISPLAY_CACHE_KEY), ...(options || {})};
  $("agg_show_all_display").checked = values.show_all !== false;
  $("agg_show_grid").checked = values.show_grid !== false;
  $("agg_show_axis_labels").checked = values.show_axis_labels !== false;
  $("agg_show_axes").checked = values.show_axes !== false;
  $("agg_show_tick_labels").checked = values.show_tick_labels !== false;
  $("agg_show_heading").checked = values.show_heading !== false;
  $("agg_show_legend").checked = values.show_legend !== false;
}

function cacheAggregateDisplayOptions() {
  writeCachedObject(AGG_DISPLAY_CACHE_KEY, currentAggregateDisplayOptions());
}

function setAggregateDisplayToggles(checked) {
  $("agg_show_grid").checked = checked;
  $("agg_show_axis_labels").checked = checked;
  $("agg_show_axes").checked = checked;
  $("agg_show_tick_labels").checked = checked;
  $("agg_show_heading").checked = checked;
  $("agg_show_legend").checked = checked;
}

function syncAggregateDisplayMaster() {
  const allChecked = [
    "agg_show_grid",
    "agg_show_axis_labels",
    "agg_show_axes",
    "agg_show_tick_labels",
    "agg_show_heading",
    "agg_show_legend",
  ].every((id) => $(id).checked);
  $("agg_show_all_display").checked = allChecked;
}

function onAggregateDisplayToggle() {
  syncAggregateDisplayMaster();
  cacheAggregateDisplayOptions();
  drawAggregateXY();
}

function onAggregatePlotLimitInput() {
  cacheAggregatePlotLimits();
  drawAggregateXY();
}

function finiteXYPoints(xValues, yValues, maxPoints) {
  const n = Math.min((xValues || []).length, (yValues || []).length);
  if (!n) return [];
  const step = Math.max(1, Math.ceil(n / Math.max(1, maxPoints)));
  const points = [];
  for (let i = 0; i < n; i += step) {
    const x = Number(xValues[i]);
    const y = Number(yValues[i]);
    if (Number.isFinite(x) && Number.isFinite(y)) points.push([x, y]);
  }
  return points;
}

function closestPointDistanceToPolyline(point, polyline) {
  if (!polyline.length) return Infinity;
  if (polyline.length === 1) {
    const dx = point[0] - polyline[0][0];
    const dy = point[1] - polyline[0][1];
    return Math.sqrt(dx * dx + dy * dy);
  }

  let minSq = Infinity;
  for (let i = 0; i < polyline.length - 1; i += 1) {
    const ax = polyline[i][0];
    const ay = polyline[i][1];
    const bx = polyline[i + 1][0];
    const by = polyline[i + 1][1];
    const abx = bx - ax;
    const aby = by - ay;
    const denom = abx * abx + aby * aby;
    const t = denom > 0
      ? Math.max(0, Math.min(1, ((point[0] - ax) * abx + (point[1] - ay) * aby) / denom))
      : 0;
    const cx = ax + t * abx;
    const cy = ay + t * aby;
    const dx = point[0] - cx;
    const dy = point[1] - cy;
    const d2 = dx * dx + dy * dy;
    if (d2 < minSq) minSq = d2;
  }
  return Math.sqrt(minSq);
}

function meanReferenceTrackingError(segment, referenceTrajectory) {
  const segmentPoints = finiteXYPoints(
    segment.x,
    segment.y,
    MAX_TRACKING_ERROR_SEGMENT_POINTS,
  );
  const referencePoints = finiteXYPoints(
    referenceTrajectory.x,
    referenceTrajectory.y,
    MAX_TRACKING_ERROR_REFERENCE_POINTS,
  );
  if (!segmentPoints.length || !referencePoints.length) return null;

  let sum = 0;
  let count = 0;
  for (const point of segmentPoints) {
    const error = closestPointDistanceToPolyline(point, referencePoints);
    if (!Number.isFinite(error)) continue;
    sum += error;
    count += 1;
  }
  if (!count) return null;
  return sum / count;
}

function meanVelocityTrackingError(segment, referenceTrajectory) {
  if (referenceTrajectory.reference_speed === null || referenceTrajectory.reference_speed === undefined) return null;
  const referenceSpeed = Number(referenceTrajectory.reference_speed);
  if (!Number.isFinite(referenceSpeed)) return null;

  let sum = 0;
  let count = 0;
  for (const value of segment.v_long || []) {
    const speed = Number(value);
    if (!Number.isFinite(speed)) continue;
    sum += Math.abs(speed - referenceSpeed);
    count += 1;
  }
  if (!count) return null;
  return sum / count;
}

function formatErrorMeters(value) {
  if (!Number.isFinite(value)) return "-";
  return value < 10 ? value.toFixed(3) : value.toFixed(2);
}

function renderAggregateTrackingErrorTable() {
  const container = $("aggregate_error_table");
  if (!container) return;
  container.innerHTML = "";

  if (!state.aggregateSegments.length) {
    const empty = document.createElement("div");
    empty.className = "aggregate-error-empty";
    empty.textContent = "No aggregated segments loaded.";
    container.appendChild(empty);
    return;
  }

  if (!state.aggregateReferenceTrajectory) {
    const empty = document.createElement("div");
    empty.className = "aggregate-error-empty";
    empty.textContent = "Load an aggregate reference trajectory to compute tracking error.";
    container.appendChild(empty);
    return;
  }

  const table = document.createElement("table");
  const thead = document.createElement("thead");
  const headerRow = document.createElement("tr");
  for (const label of ["Segment", "Mean Euclidean Distance [m]", "Velocity MAD [m/s]"]) {
    const th = document.createElement("th");
    th.textContent = label;
    headerRow.appendChild(th);
  }
  thead.appendChild(headerRow);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  for (const seg of state.aggregateSegments) {
    const meanError = meanReferenceTrackingError(seg, state.aggregateReferenceTrajectory);
    const velocityError = meanVelocityTrackingError(seg, state.aggregateReferenceTrajectory);
    const tr = document.createElement("tr");
    const values = [
      seg.name || seg.bin_name || "segment",
      formatErrorMeters(meanError),
      formatErrorMeters(velocityError),
    ];
    for (const value of values) {
      const td = document.createElement("td");
      td.textContent = value;
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  container.appendChild(table);
}

function currentAggregateReferenceConfig() {
  const spec = selectedAggregateReferenceSpec();
  if (!spec) return null;
  return {
    trajectory_name: spec.name,
    params: readAggregateReferenceParams(),
    color: $("agg_ref_traj_color").value,
  };
}

function applyAggregateReferenceConfig(config) {
  if (!config || !config.trajectory_name) return;
  $("agg_ref_traj_select").value = config.trajectory_name;
  renderAggregateReferenceParams();
  if (config.color) $("agg_ref_traj_color").value = config.color;
  const params = config.params || {};
  const spec = selectedAggregateReferenceSpec();
  if (!spec) return;
  for (const param of spec.params) {
    if (!(param.name in params)) continue;
    const input = $(aggParamInputId(param.name));
    if (!input) continue;
    if (param.kind === "bool") input.checked = Boolean(params[param.name]);
    else input.value = params[param.name];
  }
}

function applyAggregateRows(rows) {
  $("aggregate_rows").innerHTML = "";
  for (const row of rows || []) {
    addAggregateRow(
      row.yaml_path || "",
      row.legend || "",
      row.color || undefined,
      row.visible !== false,
      row.line_width ?? 2,
    );
  }
  if (document.querySelectorAll(".aggregate-row").length === 0) {
    addAggregateRow();
  }
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

function drawAggregateXY() {
  const traces = [];
  for (const seg of state.aggregateSegments) {
    if (seg.visible === false) continue;
    traces.push({
      x: seg.x,
      y: seg.y,
      mode: "lines",
      line: {width: seg.line_width || 2, color: seg.color || undefined},
      name: seg.name || seg.bin_name || "segment",
    });
  }

  if (state.aggregateReferenceTrajectory) {
    traces.push({
      x: state.aggregateReferenceTrajectory.x,
      y: state.aggregateReferenceTrajectory.y,
      mode: "lines",
      line: {color: state.aggregateReferenceTrajectory.color || "#f59e0b", width: 3, dash: "dash"},
      name: "Reference Trajectory",
    });
  }

  const display = currentAggregateDisplayOptions();
  const layout = {
    title: display.show_heading ? "Aggregate XY trajectories" : "",
    xaxis: {
      title: display.show_axis_labels ? "x [m]" : "",
      scaleanchor: "y",
      scaleratio: 1,
      zeroline: display.show_axes,
      showline: display.show_axes,
      ticks: display.show_axes ? "outside" : "",
      showticklabels: display.show_tick_labels,
      showgrid: display.show_grid,
    },
    yaxis: {
      title: display.show_axis_labels ? "y [m]" : "",
      zeroline: display.show_axes,
      showline: display.show_axes,
      ticks: display.show_axes ? "outside" : "",
      showticklabels: display.show_tick_labels,
      showgrid: display.show_grid,
    },
    margin: {
      t: display.show_heading ? 40 : 14,
      r: 16,
      b: display.show_axis_labels || display.show_tick_labels ? 48 : 28,
      l: display.show_axis_labels || display.show_tick_labels ? 58 : 36,
    },
    dragmode: "pan",
    hovermode: "closest",
    legend: {orientation: "h", x: 0.5, xanchor: "center", y: -0.12},
    showlegend: display.show_legend,
  };
  const limits = currentAggregatePlotLimits();
  applyAxisRange(layout.xaxis, limits.x_min, limits.x_max);
  applyAxisRange(layout.yaxis, limits.y_min, limits.y_max);
  Plotly.react("aggregate_xy_plot", traces, layout,
    {scrollZoom: true, displaylogo: false, responsive: true});
  renderAggregateTrackingErrorTable();
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
$("agg_ref_traj_select").addEventListener("change", renderAggregateReferenceParams);
$("ref_traj_color").addEventListener("input", (e) => {
  if (!state.referenceTrajectory) return;
  state.referenceTrajectory.color = e.target.value;
  if (state.lastTrajectoryData) drawBagXY(state.lastTrajectoryData);
});
$("agg_ref_traj_color").addEventListener("input", (e) => {
  if (!state.aggregateReferenceTrajectory) return;
  state.aggregateReferenceTrajectory.color = e.target.value;
  drawAggregateXY();
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
$("btn_load_agg_ref_traj").onclick = async () => {
  const spec = selectedAggregateReferenceSpec();
  if (!spec) return;
  $("agg_ref_traj_status").textContent = "Loading...";
  $("btn_load_agg_ref_traj").disabled = true;
  try {
    state.aggregateReferenceTrajectory = await postJSON("/dbm/api/reference_trajectory", {
      trajectory_name: spec.name,
      params: readAggregateReferenceParams(),
      max_points: 9000,
    });
    state.aggregateReferenceTrajectory.color = $("agg_ref_traj_color").value;
    $("agg_ref_traj_status").textContent =
      `${state.aggregateReferenceTrajectory.label}: ${state.aggregateReferenceTrajectory.points} point(s)`;
    drawAggregateXY();
  } catch (e) {
    $("agg_ref_traj_status").textContent = "Error: " + e.message;
  } finally {
    $("btn_load_agg_ref_traj").disabled = false;
  }
};
$("btn_clear_agg_ref_traj").onclick = () => {
  state.aggregateReferenceTrajectory = null;
  $("agg_ref_traj_status").textContent = "";
  drawAggregateXY();
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
$("btn_load_aggregate").onclick = async () => {
  $("aggregate_status").textContent = "Loading...";
  try {
    const res = await postJSON("/dbm/api/aggregate_segments", {
      segments: readAggregateRows(),
      max_points: 5000,
    });
    state.aggregateSegments = res.segments || [];
    updateAggregateStatus(res.errors || []);
    drawAggregateXY();
  } catch (e) {
    $("aggregate_status").textContent = "Error: " + e.message;
  }
};
$("btn_clear_aggregate").onclick = () => {
  state.aggregateSegments = [];
  $("aggregate_rows").innerHTML = "";
  addAggregateRow();
  $("aggregate_status").textContent = "";
  drawAggregateXY();
};
$("btn_add_aggregate_row").onclick = () => addAggregateRow();
if ($("btn_save_aggregate_config")) {
  $("btn_save_aggregate_config").onclick = async () => {
    $("aggregate_config_status").textContent = "Saving...";
    try {
      const res = await postJSON("/dbm/api/save_aggregate_config", {
        config_path: $("aggregate_config_path").value,
        segments: readAggregateRows(),
        reference: currentAggregateReferenceConfig(),
        limits: currentAggregatePlotLimits(),
        display: currentAggregateDisplayOptions(),
      });
      $("aggregate_config_path").value = res.config_path;
      $("aggregate_config_status").textContent = res.config_path;
    } catch (e) {
      $("aggregate_config_status").textContent = "Error: " + e.message;
    }
  };
}
if ($("btn_load_aggregate_config")) {
  $("btn_load_aggregate_config").onclick = async () => {
    $("aggregate_config_status").textContent = "Loading...";
    try {
      const res = await postJSON("/dbm/api/load_aggregate_config", {
        config_path: $("aggregate_config_path").value,
      });
      $("aggregate_config_path").value = res.config_path;
      applyAggregateRows(res.segments || []);
      applyAggregateReferenceConfig(res.reference);
      applyAggregatePlotLimits(res.limits);
      applyAggregateDisplayOptions(res.display);
      cacheAggregatePlotLimits();
      cacheAggregateDisplayOptions();
      $("aggregate_config_status").textContent = res.config_path;

      const agg = await postJSON("/dbm/api/aggregate_segments", {
        segments: readAggregateRows(),
        max_points: 5000,
      });
      state.aggregateSegments = agg.segments || [];

      if (res.reference && res.reference.trajectory_name) {
        state.aggregateReferenceTrajectory = await postJSON("/dbm/api/reference_trajectory", {
          trajectory_name: res.reference.trajectory_name,
          params: readAggregateReferenceParams(),
          max_points: 9000,
        });
        state.aggregateReferenceTrajectory.color = $("agg_ref_traj_color").value;
      }

      updateAggregateStatus(agg.errors || []);
      drawAggregateXY();
    } catch (e) {
      $("aggregate_config_status").textContent = "Error: " + e.message;
    }
  };
}
$("agg_limit_x_min").addEventListener("input", onAggregatePlotLimitInput);
$("agg_limit_x_max").addEventListener("input", onAggregatePlotLimitInput);
$("agg_limit_y_min").addEventListener("input", onAggregatePlotLimitInput);
$("agg_limit_y_max").addEventListener("input", onAggregatePlotLimitInput);
$("agg_show_all_display").addEventListener("change", (e) => {
  setAggregateDisplayToggles(e.target.checked);
  cacheAggregateDisplayOptions();
  drawAggregateXY();
});
$("agg_show_grid").addEventListener("change", onAggregateDisplayToggle);
$("agg_show_axis_labels").addEventListener("change", onAggregateDisplayToggle);
$("agg_show_axes").addEventListener("change", onAggregateDisplayToggle);
$("agg_show_tick_labels").addEventListener("change", onAggregateDisplayToggle);
$("agg_show_heading").addEventListener("change", onAggregateDisplayToggle);
$("agg_show_legend").addEventListener("change", onAggregateDisplayToggle);
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

addAggregateRow();
applyAggregatePlotLimits();
applyAggregateDisplayOptions();
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
