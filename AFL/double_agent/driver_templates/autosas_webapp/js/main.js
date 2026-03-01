const state = {
  availableModels: [],
  models: [],
  selectedIndex: null,
  showPd: false,
  showMagnetic: false,
  previewData: null,
  loadedSampleIndex: 0,
  loadedSampleTotal: 0,
  isBootstrapping: false,
};

let previewTimer = null;
let currentTaskId = null;
let previewAbortController = null;
let previewRequestId = 0;

const dom = {
  status: document.getElementById('status'),
  modelsList: document.getElementById('models-list'),
  addSelect: document.getElementById('add-model-select'),
  addBtn: document.getElementById('add-model-btn'),
  removeBtn: document.getElementById('remove-model-btn'),
  editorEmpty: document.getElementById('editor-empty'),
  editorContent: document.getElementById('editor-content'),
  modelName: document.getElementById('model-name'),
  qmin: document.getElementById('model-qmin'),
  qmax: document.getElementById('model-qmax'),
  filterPd: document.getElementById('filter-pd'),
  filterMag: document.getElementById('filter-mag'),
  paramsTable: document.getElementById('params-table'),
  exportBox: document.getElementById('export-box'),
  resultsBox: document.getElementById('results-box'),
  fitMethod: document.getElementById('fit-method'),
  entryIds: document.getElementById('entry-ids'),
  sampleDim: document.getElementById('sample-dim'),
  varPrefix: document.getElementById('var-prefix'),
  qVar: document.getElementById('q-var'),
  sasVar: document.getElementById('sas-var'),
  sasErrVar: document.getElementById('sas-err-var'),
  sasResVar: document.getElementById('sas-res-var'),
  tabModelBtn: document.getElementById('tab-model-btn'),
  tabDataBtn: document.getElementById('tab-data-btn'),
  tabModelContent: document.getElementById('tab-model-content'),
  tabDataContent: document.getElementById('tab-data-content'),
  dataPrevBtn: document.getElementById('data-prev-btn'),
  dataNextBtn: document.getElementById('data-next-btn'),
  dataIndexIndicator: document.getElementById('data-index-indicator'),
  datasetSummary: document.getElementById('dataset-summary'),
  datasetHtml: document.getElementById('dataset-html'),
};

document.addEventListener('DOMContentLoaded', async () => {
  wireEvents();
  initializePlot();
  await bootstrap();
});

function wireEvents() {
  document.getElementById('bootstrap-btn').addEventListener('click', bootstrap);
  document.getElementById('add-model-btn').addEventListener('click', addModel);
  document.getElementById('remove-model-btn').addEventListener('click', removeModel);
  document.getElementById('apply-btn').addEventListener('click', applyModelInputs);
  document.getElementById('export-btn').addEventListener('click', refreshExportBox);
  document.getElementById('set-data-btn').addEventListener('click', setDataContext);
  document.getElementById('run-fit-btn').addEventListener('click', runFit);
  document.getElementById('refresh-summary-btn').addEventListener('click', refreshSummary);
  dom.tabModelBtn.addEventListener('click', () => setActiveTab('model'));
  dom.tabDataBtn.addEventListener('click', () => setActiveTab('data'));
  dom.dataPrevBtn.addEventListener('click', () => stepLoadedSample(-1));
  dom.dataNextBtn.addEventListener('click', () => stepLoadedSample(1));

  dom.modelName.addEventListener('input', () => {
    const model = selectedModel();
    if (!model) return;
    model.name = dom.modelName.value;
    renderModelsList();
    refreshExportBox();
  });

  dom.qmin.addEventListener('change', () => {
    const model = selectedModel();
    if (!model) return;
    model.q_min = Number(dom.qmin.value);
    refreshExportBox();
    schedulePreview();
  });

  dom.qmax.addEventListener('change', () => {
    const model = selectedModel();
    if (!model) return;
    model.q_max = Number(dom.qmax.value);
    refreshExportBox();
    schedulePreview();
  });

  dom.filterPd.addEventListener('change', () => {
    state.showPd = dom.filterPd.checked;
    renderParamEditor();
  });

  dom.filterMag.addEventListener('change', () => {
    state.showMagnetic = dom.filterMag.checked;
    renderParamEditor();
  });
}

function setActiveTab(tabName) {
  const modelActive = tabName === 'model';
  dom.tabModelBtn.classList.toggle('active', modelActive);
  dom.tabDataBtn.classList.toggle('active', !modelActive);
  dom.tabModelContent.classList.toggle('active', modelActive);
  dom.tabDataContent.classList.toggle('active', !modelActive);
}

function isNumericValue(value) {
  return typeof value === 'number' && Number.isFinite(value);
}

function defaultBounds(value) {
  if (!isNumericValue(value)) return null;
  if (value > 0) return [value / 10, value * 10];
  if (value < 0) return [value * 10, value / 10];
  return [-1, 1];
}

function normalizeBounds(bounds, fallbackValue = 0) {
  if (!Array.isArray(bounds) || bounds.length !== 2) {
    return defaultBounds(fallbackValue);
  }

  let minV = Number(bounds[0]);
  let maxV = Number(bounds[1]);
  if (!Number.isFinite(minV) || !Number.isFinite(maxV) || minV === maxV) {
    return defaultBounds(fallbackValue);
  }
  if (minV > maxV) {
    const tmp = minV;
    minV = maxV;
    maxV = tmp;
  }
  return [minV, maxV];
}

function clampToBounds(value, bounds) {
  if (!isNumericValue(value) || !Array.isArray(bounds) || bounds.length !== 2) return value;
  return Math.max(bounds[0], Math.min(bounds[1], value));
}

function sliderStep(bounds) {
  if (!Array.isArray(bounds) || bounds.length !== 2) return 0.01;
  const span = Math.abs(bounds[1] - bounds[0]);
  if (span === 0) return 0.01;
  return span / 200;
}

function initializePlot() {
  if (!window.Plotly) {
    setStatus('Plotly failed to load from local static assets.', true);
    return;
  }

  Plotly.newPlot('plot', [
    { x: [], y: [], mode: 'lines', name: 'Model', line: { color: '#0b6e6a', width: 2 } },
    { x: [], y: [], mode: 'markers', name: 'Data', marker: { color: '#111', size: 5 } },
  ], {
    margin: { l: 55, r: 15, t: 20, b: 50 },
    xaxis: { type: 'log', title: 'q (Å⁻¹)' },
    yaxis: { type: 'log', title: 'Intensity (cm⁻¹)' },
    showlegend: true,
    paper_bgcolor: '#fff',
    plot_bgcolor: '#fff',
  }, { responsive: true });
}

async function ensureAuthenticated() {
  let token = localStorage.getItem('token');
  if (token) return token;

  const response = await fetch('/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username: 'user', password: 'domo_arigato' }),
  });

  if (!response.ok) {
    throw new Error('Login failed');
  }

  const data = await response.json();
  token = data.token;
  localStorage.setItem('token', token);
  return token;
}

async function authenticatedFetch(url, options = {}) {
  const token = await ensureAuthenticated();
  const headers = {
    'Content-Type': 'application/json',
    Authorization: `Bearer ${token}`,
    ...(options.headers || {}),
  };

  return fetch(url, { ...options, headers });
}

async function bootstrap() {
  if (state.isBootstrapping) return;
  state.isBootstrapping = true;
  try {
    const res = await authenticatedFetch('/autosas_get_bootstrap', { method: 'GET' });
    const data = await res.json();
    if (data.status !== 'success') {
      throw new Error(data.message || 'Bootstrap failed');
    }

    state.availableModels = data.available_models || [];
    state.previewData = data.data_preview || null;
    dom.addSelect.innerHTML = '';
    state.availableModels.forEach((modelName) => {
      const option = document.createElement('option');
      option.value = modelName;
      option.textContent = modelName;
      dom.addSelect.appendChild(option);
    });

    dom.fitMethod.value = JSON.stringify(data.fit_method || {}, null, 2);
    const tiled = data.tiled_context || {};
    dom.entryIds.value = Array.isArray(tiled.entry_ids) ? tiled.entry_ids.join('\n') : '';
    dom.sampleDim.value = tiled.concat_dim || 'sample';
    dom.varPrefix.value = tiled.variable_prefix || '';
    dom.qVar.value = tiled.q_variable || 'q';
    dom.sasVar.value = tiled.sas_variable || 'I';
    dom.sasErrVar.value = tiled.sas_err_variable || 'dI';
    dom.sasResVar.value = tiled.sas_resolution_variable || '';
    state.loadedSampleTotal = Number(data.loaded_sample_total || 0);
    state.loadedSampleIndex = 0;

    state.models = [];
    const existingInputs = Array.isArray(data.model_inputs) ? data.model_inputs : [];
    if (existingInputs.length > 0) {
      for (let i = 0; i < existingInputs.length; i += 1) {
        const rebuilt = await expandModelInput(existingInputs[i], i + 1);
        state.models.push(rebuilt);
      }
      state.selectedIndex = 0;
    } else {
      state.selectedIndex = null;
    }

    renderAll();
    renderDatasetSummary(data.loaded_dataset);
    updateDataNavigator();
    if (state.loadedSampleTotal > 0) {
      await loadSampleByIndex(0);
    } else if ((tiled.entry_ids || []).length > 0 && !data.has_fitter) {
      await setDataContext({ silent: true, refreshBootstrap: false });
    }

    if (data.last_fit_summary) {
      dom.resultsBox.textContent = JSON.stringify(data.last_fit_summary, null, 2);
    }

    setStatus('AutoSAS web app loaded.');
  } catch (err) {
    setStatus(`Bootstrap error: ${err.message}`, true);
  } finally {
    state.isBootstrapping = false;
  }
}

async function expandModelInput(modelInput, index) {
  const sasmodel = modelInput.sasmodel;
  const res = await authenticatedFetch(`/autosas_get_model_template?sasmodel=${encodeURIComponent(sasmodel)}&index=${index}`, { method: 'GET' });
  const data = await res.json();
  if (data.status !== 'success') {
    throw new Error(data.message || `Failed to load template for ${sasmodel}`);
  }

  const model = data.model;
  model.name = modelInput.name;
  model.q_min = modelInput.q_min;
  model.q_max = modelInput.q_max;

  const fitParams = modelInput.fit_params || {};
  Object.entries(fitParams).forEach(([baseName, p]) => {
    if (!model.params[baseName]) return;

    model.params[baseName].value = p.value;
    model.params[baseName].autosas = true;
    model.params[baseName].use_bounds = !!p.bounds;
    if (p.bounds) model.params[baseName].bounds = p.bounds;

    if (p.pd !== undefined && model.params[`${baseName}_pd`]) {
      model.params[`${baseName}_pd`].value = p.pd;
      model.params[`${baseName}_pd`].autosas = true;
    }
    if (p.pd_type !== undefined && model.params[`${baseName}_pd_type`]) {
      model.params[`${baseName}_pd_type`].value = p.pd_type;
      model.params[`${baseName}_pd_type`].autosas = true;
    }
    if (p.pd_n !== undefined && model.params[`${baseName}_pd_n`]) {
      model.params[`${baseName}_pd_n`].value = p.pd_n;
      model.params[`${baseName}_pd_n`].autosas = true;
    }
    if (p.pd_nsigma !== undefined && model.params[`${baseName}_pd_nsigma`]) {
      model.params[`${baseName}_pd_nsigma`].value = p.pd_nsigma;
      model.params[`${baseName}_pd_nsigma`].autosas = true;
    }
  });

  return model;
}

function renderAll() {
  renderModelsList();
  renderEditor();
  refreshExportBox();
  schedulePreview();
}

function renderModelsList() {
  if (state.models.length === 0) {
    dom.modelsList.innerHTML = '<div class="hint" style="padding:8px;">No models added.</div>';
    updateModelButtons();
    return;
  }

  dom.modelsList.innerHTML = state.models
    .map((m, i) => `<div class="model-item ${i === state.selectedIndex ? 'active' : ''}" data-index="${i}">${escapeHtml(m.name)}<br><small>${escapeHtml(m.sasmodel)}</small></div>`)
    .join('');

  dom.modelsList.querySelectorAll('.model-item').forEach((el) => {
    el.addEventListener('click', () => {
      state.selectedIndex = Number(el.getAttribute('data-index'));
      renderEditor();
      renderModelsList();
      schedulePreview();
    });
  });
  updateModelButtons();
}

function updateModelButtons() {
  const hasSelection = state.selectedIndex !== null && state.selectedIndex >= 0 && state.selectedIndex < state.models.length;
  dom.removeBtn.disabled = !hasSelection;
}

function renderEditor() {
  const model = selectedModel();
  if (!model) {
    dom.editorEmpty.classList.remove('hidden');
    dom.editorContent.classList.add('hidden');
    updateModelButtons();
    return;
  }

  dom.editorEmpty.classList.add('hidden');
  dom.editorContent.classList.remove('hidden');

  dom.modelName.value = model.name;
  dom.qmin.value = model.q_min;
  dom.qmax.value = model.q_max;

  renderParamEditor();
  updateModelButtons();
}

function renderParamEditor() {
  const model = selectedModel();
  if (!model) {
    dom.paramsTable.innerHTML = '';
    return;
  }

  const rows = [];
  for (const [paramName, info] of Object.entries(model.params)) {
    const isPd = paramName.includes('_pd');
    const isMagnetic = paramName.startsWith('up_') || paramName.includes('_M0') || paramName.includes('_mphi') || paramName.includes('_mtheta');

    if (isPd && !state.showPd) continue;
    if (isMagnetic && !state.showMagnetic) continue;

    const boundsText = Array.isArray(info.bounds) ? `${info.bounds[0]}, ${info.bounds[1]}` : '';

    rows.push(`
      <div class="param-row" data-param="${escapeHtml(paramName)}">
        <div>
          <strong>${escapeHtml(paramName)}</strong><br />
          <small>${escapeHtml(boundsText)}</small>
        </div>
        <div class="param-controls">
          ${renderValueInput(paramName, info)}
          ${renderBoundsEditor(paramName, info)}
        </div>
        <label><input type="checkbox" class="param-autosas" ${info.autosas ? 'checked' : ''} /> AutoSAS</label>
        <label><input type="checkbox" class="param-bounds" ${info.use_bounds ? 'checked' : ''} /> Bounds</label>
      </div>
    `);
  }

  dom.paramsTable.innerHTML = rows.join('') || '<div class="hint" style="padding:8px;">No parameters in current filter.</div>';

  dom.paramsTable.querySelectorAll('.param-row').forEach((row) => {
    const paramName = row.getAttribute('data-param');
    const info = model.params[paramName];

    const valueInput = row.querySelector('.param-value');
    const autosas = row.querySelector('.param-autosas');
    const bounds = row.querySelector('.param-bounds');
    const slider = row.querySelector('.param-slider');
    const boundMinInput = row.querySelector('.bound-min');
    const boundMaxInput = row.querySelector('.bound-max');

    valueInput.addEventListener('change', () => {
      if (valueInput.tagName === 'SELECT') {
        if (valueInput.value === 'true') info.value = true;
        else if (valueInput.value === 'false') info.value = false;
        else info.value = valueInput.value;
      } else {
        if (typeof info.value === 'number') info.value = Number(valueInput.value);
        else info.value = valueInput.value;
      }
      if (info.use_bounds && isNumericValue(info.value)) {
        info.bounds = normalizeBounds(info.bounds, info.value);
        info.value = clampToBounds(info.value, info.bounds);
        renderParamEditor();
      }
      refreshExportBox();
      schedulePreview();
    });

    autosas.addEventListener('change', () => {
      info.autosas = autosas.checked;
      refreshExportBox();
    });

    bounds.addEventListener('change', () => {
      info.use_bounds = bounds.checked;
      if (info.use_bounds && isNumericValue(info.value)) {
        info.bounds = normalizeBounds(info.bounds, info.value);
        info.value = clampToBounds(info.value, info.bounds);
      }
      renderParamEditor();
      refreshExportBox();
      schedulePreview();
    });

    if (slider) {
      slider.addEventListener('input', () => {
        const nextValue = Number(slider.value);
        info.value = nextValue;
        valueInput.value = String(nextValue);
        refreshExportBox();
        schedulePreviewImmediate();
      });
    }

    if (boundMinInput && boundMaxInput) {
      const applyBoundsInputs = () => {
        const minV = Number(boundMinInput.value);
        const maxV = Number(boundMaxInput.value);
        if (!Number.isFinite(minV) || !Number.isFinite(maxV) || minV === maxV) return;

        info.bounds = minV < maxV ? [minV, maxV] : [maxV, minV];
        info.value = clampToBounds(Number(info.value), info.bounds);
        renderParamEditor();
        refreshExportBox();
        schedulePreview();
      };

      boundMinInput.addEventListener('change', applyBoundsInputs);
      boundMaxInput.addEventListener('change', applyBoundsInputs);
    }
  });
}

function renderValueInput(paramName, info) {
  if (Array.isArray(info.options)) {
    const opts = info.options
      .map((o) => `<option value="${escapeHtml(o)}" ${o === info.value ? 'selected' : ''}>${escapeHtml(o)}</option>`)
      .join('');
    return `<select class="param-value">${opts}</select>`;
  }

  if (typeof info.value === 'boolean') {
    const opts = ['true', 'false']
      .map((o) => `<option value="${o}" ${String(info.value) === o ? 'selected' : ''}>${o}</option>`)
      .join('');
    return `<select class="param-value">${opts}</select>`;
  }

  if (typeof info.value !== 'number') {
    return `<input class="param-value" type="text" value="${escapeHtml(String(info.value))}" />`;
  }

  const step = paramName.endsWith('_pd_n') ? '1' : 'any';
  return `<input class="param-value" type="number" step="${step}" value="${escapeHtml(String(info.value))}" />`;
}

function renderBoundsEditor(paramName, info) {
  if (!info.use_bounds || !isNumericValue(info.value)) return '';

  const bounds = normalizeBounds(info.bounds, Number(info.value));
  if (!bounds) return '';
  info.bounds = bounds;
  info.value = clampToBounds(Number(info.value), info.bounds);

  const step = sliderStep(bounds);
  const integerLike = paramName.endsWith('_pd_n');
  const minAttr = integerLike ? Math.round(bounds[0]) : bounds[0];
  const maxAttr = integerLike ? Math.round(bounds[1]) : bounds[1];
  const valueAttr = integerLike ? Math.round(info.value) : info.value;
  const stepAttr = integerLike ? 1 : step;

  return `
    <div class="bounds-editor">
      <input
        class="param-slider"
        type="range"
        min="${escapeHtml(String(minAttr))}"
        max="${escapeHtml(String(maxAttr))}"
        step="${escapeHtml(String(stepAttr))}"
        value="${escapeHtml(String(valueAttr))}"
      />
      <div class="bounds-inline">
        <label>min <input class="bound-min" type="number" step="any" value="${escapeHtml(String(bounds[0]))}" /></label>
        <label>max <input class="bound-max" type="number" step="any" value="${escapeHtml(String(bounds[1]))}" /></label>
      </div>
    </div>
  `;
}

function selectedModel() {
  if (state.selectedIndex === null) return null;
  if (state.selectedIndex < 0 || state.selectedIndex >= state.models.length) return null;
  return state.models[state.selectedIndex];
}

async function addModel() {
  const sasmodel = (dom.addSelect.value || '').trim();
  if (!sasmodel) return;

  try {
    const index = state.models.length + 1;
    const res = await authenticatedFetch(`/autosas_get_model_template?sasmodel=${encodeURIComponent(sasmodel)}&index=${index}`, { method: 'GET' });
    const data = await res.json();
    if (data.status !== 'success') {
      throw new Error(data.message || 'Failed to add model');
    }

    state.models.push(data.model);
    state.selectedIndex = state.models.length - 1;
    renderAll();
    setStatus(`Added model: ${data.model.name}`);
  } catch (err) {
    setStatus(`Add model error: ${err.message}`, true);
  }
}

function removeModel() {
  if (state.selectedIndex === null) return;
  const removedName = state.models[state.selectedIndex]?.name || 'model';
  state.models.splice(state.selectedIndex, 1);
  if (state.models.length === 0) {
    state.selectedIndex = null;
  } else {
    state.selectedIndex = Math.min(state.selectedIndex, state.models.length - 1);
  }

  // Renumber names like widget behavior.
  state.models.forEach((m, idx) => {
    m.name = `${m.sasmodel}_${idx + 1}`;
  });

  renderAll();
  setStatus(`Removed model: ${removedName}`);
}

function buildModelInputs() {
  return state.models.map((model) => {
    const fitParams = {};
    const pdGroups = {};

    Object.entries(model.params).forEach(([name, info]) => {
      if (!info.autosas) return;

      if (name.endsWith('_pd') || name.endsWith('_pd_type') || name.endsWith('_pd_n') || name.endsWith('_pd_nsigma')) {
        const base = name.split('_pd')[0];
        pdGroups[base] = pdGroups[base] || {};

        if (name.endsWith('_pd')) pdGroups[base].pd = info.value;
        else if (name.endsWith('_pd_type')) pdGroups[base].pd_type = info.value;
        else if (name.endsWith('_pd_n')) pdGroups[base].pd_n = info.value;
        else if (name.endsWith('_pd_nsigma')) pdGroups[base].pd_nsigma = info.value;
        return;
      }

      const p = { value: typeof info.value === 'number' ? Number(info.value) : info.value };
      if (info.use_bounds && Array.isArray(info.bounds) && info.bounds.length === 2) {
        p.bounds = [Number(info.bounds[0]), Number(info.bounds[1])];
      }
      fitParams[name] = p;
    });

    Object.entries(pdGroups).forEach(([base, pdVals]) => {
      if (fitParams[base]) {
        Object.assign(fitParams[base], pdVals);
      }
    });

    return {
      name: model.name,
      sasmodel: model.sasmodel,
      q_min: Number(model.q_min),
      q_max: Number(model.q_max),
      fit_params: fitParams,
    };
  });
}

function refreshExportBox() {
  const modelInputs = buildModelInputs();
  dom.exportBox.value = JSON.stringify(modelInputs, null, 2);
}

function schedulePreview() {
  if (previewTimer) window.clearTimeout(previewTimer);
  previewTimer = window.setTimeout(updatePreview, 250);
}

function schedulePreviewImmediate() {
  if (previewTimer) {
    window.clearTimeout(previewTimer);
    previewTimer = null;
  }
  updatePreview();
}

async function updatePreview() {
  const model = selectedModel();
  if (!model || !window.Plotly) return;

  try {
    if (previewAbortController) {
      previewAbortController.abort();
    }
    previewAbortController = new AbortController();
    const requestId = ++previewRequestId;

    const payload = encodeURIComponent(JSON.stringify(model));
    const res = await authenticatedFetch(`/autosas_preview_model?model_config=${payload}`, {
      method: 'GET',
      signal: previewAbortController.signal,
    });
    const data = await res.json();
    if (requestId !== previewRequestId) return;

    if (data.status !== 'success') {
      throw new Error(data.message || 'Preview failed');
    }

    const traces = [
      { x: data.q, y: data.intensity, mode: 'lines', name: 'Model', line: { color: '#0b6e6a', width: 2 } },
      {
        x: state.previewData?.q || [],
        y: state.previewData?.I || [],
        mode: 'markers',
        name: 'Data',
        marker: { color: '#111', size: 5 },
      },
    ];

    Plotly.react('plot', traces, {
      margin: { l: 55, r: 15, t: 20, b: 50 },
      xaxis: { type: 'log', title: 'q (Å⁻¹)' },
      yaxis: { type: 'log', title: 'Intensity (cm⁻¹)' },
      showlegend: true,
      paper_bgcolor: '#fff',
      plot_bgcolor: '#fff',
    }, { responsive: true });
  } catch (err) {
    if (err && err.name === 'AbortError') return;
    setStatus(`Preview error: ${err.message}`, true);
  } finally {
    previewAbortController = null;
  }
}

async function queueTask(taskName, payload) {
  const body = { task_name: taskName, ...payload };
  const res = await authenticatedFetch('/enqueue', {
    method: 'POST',
    body: JSON.stringify(body),
  });
  const taskId = (await res.text()).trim();
  if (!res.ok || !taskId) {
    throw new Error(`Failed to queue task ${taskName}`);
  }
  currentTaskId = taskId;
  return taskId;
}

async function waitForTask(taskId, timeoutMs = 180000) {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    const res = await authenticatedFetch('/get_queue', { method: 'GET' });
    const [history, runningTask, queue] = await res.json();

    const done = history.find((t) => t.uuid === taskId);
    if (done) {
      return done.meta?.return_val;
    }

    const stillQueued = (runningTask && runningTask.uuid === taskId) || queue.some((t) => t.uuid === taskId);
    if (!stillQueued) break;

    await sleep(1000);
  }

  throw new Error(`Task ${taskId} timed out.`);
}

async function applyModelInputs() {
  try {
    const modelInputs = buildModelInputs();
    const validateRes = await authenticatedFetch(`/autosas_validate_model_inputs?model_inputs=${encodeURIComponent(JSON.stringify(modelInputs))}`, { method: 'GET' });
    const validateData = await validateRes.json();
    if (validateData.status !== 'success') {
      throw new Error(validateData.message || 'Validation failed');
    }

    setStatus('Applying model_inputs...');
    const taskId = await queueTask('autosas_apply_model_inputs', { model_inputs: modelInputs });
    const result = await waitForTask(taskId);
    setStatus(result?.message || 'Applied model_inputs.');
  } catch (err) {
    setStatus(`Apply error: ${err.message}`, true);
  }
}

function renderDatasetSummary(dataset) {
  if (!dataset) {
    dom.datasetSummary.textContent = 'No dataset loaded.';
    dom.datasetHtml.innerHTML = '';
    return;
  }

  const dims = dataset.dims || {};
  const dataVars = dataset.data_vars || [];
  dom.datasetSummary.textContent = `dims: ${Object.entries(dims).map(([k, v]) => `${k}:${v}`).join(', ')} | data_vars: ${dataVars.length}`;
  dom.datasetHtml.innerHTML = dataset.html || '<em>No HTML dataset preview available.</em>';
}

function updateDataNavigator() {
  const total = state.loadedSampleTotal || 0;
  const displayIndex = total > 0 ? state.loadedSampleIndex + 1 : 0;
  dom.dataIndexIndicator.textContent = `Data ${displayIndex}/${total}`;
  dom.dataPrevBtn.disabled = total === 0 || state.loadedSampleIndex <= 0;
  dom.dataNextBtn.disabled = total === 0 || state.loadedSampleIndex >= total - 1;
}

async function loadSampleByIndex(index) {
  const res = await authenticatedFetch(`/autosas_get_loaded_sample?sample_index=${index}`, { method: 'GET' });
  const data = await res.json();
  if (data.status !== 'success') {
    throw new Error(data.message || 'Failed to load sample.');
  }

  state.loadedSampleIndex = Number(data.sample_index || 0);
  state.loadedSampleTotal = Number(data.sample_total || 0);
  state.previewData = { q: data.q || [], I: data.I || [] };
  updateDataNavigator();
  schedulePreviewImmediate();
}

async function stepLoadedSample(direction) {
  try {
    const target = state.loadedSampleIndex + direction;
    if (target < 0 || target >= state.loadedSampleTotal) return;
    await loadSampleByIndex(target);
  } catch (err) {
    setStatus(`Sample navigation error: ${err.message}`, true);
  }
}

async function setDataContext(options = {}) {
  const { silent = false, refreshBootstrap = true } = options;
  try {
    const entryIds = (dom.entryIds.value || '')
      .replaceAll(',', '\n')
      .split('\n')
      .map((v) => v.trim())
      .filter((v) => v.length > 0);
    if (entryIds.length === 0) {
      throw new Error('At least one tiled entry id is required.');
    }

    const payload = {
      entry_ids: entryIds,
      concat_dim: dom.sampleDim.value.trim() || 'sample',
      variable_prefix: dom.varPrefix.value.trim() || '',
      q_variable: dom.qVar.value.trim() || 'q',
      sas_variable: dom.sasVar.value.trim() || 'I',
      sas_err_variable: dom.sasErrVar.value.trim() || 'dI',
      sas_resolution_variable: dom.sasResVar.value.trim() || null,
    };

    if (!silent) setStatus('Assembling tiled datasets and setting SAS data context...');
    const taskId = await queueTask('autosas_set_tiled_data_context', payload);
    const result = await waitForTask(taskId);
    if (result) {
      state.loadedSampleTotal = Number(result.n_samples || 0);
      state.loadedSampleIndex = 0;
      renderDatasetSummary(result);
      updateDataNavigator();
      if (state.loadedSampleTotal > 0) {
        await loadSampleByIndex(0);
      }
    }
    if (!silent) setStatus(result?.message || 'SAS tiled data context set.');

    if (refreshBootstrap) {
      await bootstrap();
    }
  } catch (err) {
    setStatus(`Set data error: ${err.message}`, true);
  }
}

async function runFit() {
  try {
    let fitMethod = null;
    const rawFitMethod = dom.fitMethod.value.trim();
    if (rawFitMethod) fitMethod = JSON.parse(rawFitMethod);

    setStatus('Running fit...');
    const payload = fitMethod ? { fit_method: fitMethod } : {};
    const taskId = await queueTask('autosas_run_fit', payload);
    const result = await waitForTask(taskId, 600000);

    if (result?.summary) {
      dom.resultsBox.textContent = JSON.stringify(result.summary, null, 2);
    }
    setStatus(`Fit completed. UUID: ${result?.fit_uuid || 'n/a'}`);
  } catch (err) {
    setStatus(`Run fit error: ${err.message}`, true);
  }
}

async function refreshSummary() {
  try {
    const res = await authenticatedFetch('/autosas_last_fit_summary', { method: 'GET' });
    const data = await res.json();
    if (data.status !== 'success') {
      throw new Error(data.message || 'No summary available');
    }

    dom.resultsBox.textContent = JSON.stringify(data, null, 2);
    setStatus('Fit summary refreshed.');
  } catch (err) {
    setStatus(`Summary error: ${err.message}`, true);
  }
}

function setStatus(message, isError = false) {
  const prefix = currentTaskId ? `[task ${currentTaskId}] ` : '';
  dom.status.textContent = `${prefix}${message}`;
  dom.status.style.borderColor = isError ? '#b3261e' : '#d7e0e8';
  dom.status.style.color = isError ? '#7d1b14' : '#103b54';
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function escapeHtml(s) {
  return s
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}
