// State management
let state = {
  groups: [],
  selectedGroupIndex: null
};

// Outputs management
let outputs = [];

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
  setupEventListeners();
  loadConfig();
  // Pane starts collapsed (already set in HTML)
});

// Authentication functions (matching pipeline_builder pattern)
async function ensureAuthenticated() {
  let token = localStorage.getItem('token');
  if (!token) {
    try {
      const response = await fetch('/login', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          username: 'user',
          password: 'domo_arigato'
        })
      });
      
      if (response.ok) {
        const data = await response.json();
        token = data.token;
        localStorage.setItem('token', token);
      } else {
        throw new Error('Login failed');
      }
    } catch (error) {
      console.error('Authentication error:', error);
      throw new Error('Failed to authenticate');
    }
  }
  return token;
}

async function authenticatedFetch(url, options = {}) {
  const token = await ensureAuthenticated();
  const headers = {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${token}`,
    ...options.headers
  };
  
  return fetch(url, {
    ...options,
    headers
  });
}

function setupEventListeners() {
  // Toolbar buttons
  document.getElementById('add-group-btn').addEventListener('click', addGroup);
  document.getElementById('remove-group-btn').addEventListener('click', removeSelectedGroup);
  document.getElementById('load-config-btn').addEventListener('click', loadConfig);
  document.getElementById('save-config-btn').addEventListener('click', saveConfig);
  document.getElementById('preview-json-btn').addEventListener('click', showJsonPreview);
  document.getElementById('assemble-input-btn').addEventListener('click', assembleInput);
  document.getElementById('predict-btn').addEventListener('click', runPredict);

  // JSON Preview Modal
  document.getElementById('close-json-modal').addEventListener('click', () => {
    document.getElementById('json-preview-modal').classList.add('hidden');
  });
  document.getElementById('copy-json-btn').addEventListener('click', copyJsonToClipboard);

  // Close modal on outside click
  document.getElementById('json-preview-modal').addEventListener('click', (e) => {
    if (e.target.id === 'json-preview-modal') {
      document.getElementById('json-preview-modal').classList.add('hidden');
    }
  });

  // Dataset Preview Modal
  document.getElementById('close-dataset-modal').addEventListener('click', () => {
    document.getElementById('dataset-preview-modal').classList.add('hidden');
  });
  document.getElementById('dataset-preview-modal').addEventListener('click', (e) => {
    if (e.target.id === 'dataset-preview-modal') {
      document.getElementById('dataset-preview-modal').classList.add('hidden');
    }
  });

  // Prediction Results Modal
  document.getElementById('close-predict-modal').addEventListener('click', () => {
    document.getElementById('predict-results-modal').classList.add('hidden');
  });
  document.getElementById('predict-results-modal').addEventListener('click', (e) => {
    if (e.target.id === 'predict-results-modal') {
      document.getElementById('predict-results-modal').classList.add('hidden');
    }
  });

  // Return Value Modal
  document.getElementById('close-returnval-modal').addEventListener('click', () => {
    document.getElementById('returnval-modal').classList.add('hidden');
  });
  document.getElementById('returnval-modal').addEventListener('click', (e) => {
    if (e.target.id === 'returnval-modal') {
      document.getElementById('returnval-modal').classList.add('hidden');
    }
  });

  // Outputs Pane
  document.getElementById('toggle-outputs-pane').addEventListener('click', toggleOutputsPane);
  document.getElementById('outputs-pane-header').addEventListener('click', (e) => {
    if (e.target.id !== 'clear-outputs-btn') {
      toggleOutputsPane();
    }
  });
  document.getElementById('clear-outputs-btn').addEventListener('click', (e) => {
    e.stopPropagation();
    clearOutputs();
  });
}

// Load configuration from server
async function loadConfig() {
  showLoading(true);
  try {
    const res = await authenticatedFetch('/get_tiled_input_config', {
      method: 'GET'
    });
    const data = await res.json();
    
    if (data.status === 'success') {
      state.groups = data.config || [];
      renderGroups();
      showStatus('Config loaded successfully', 'success');
    } else {
      showStatus('Failed to load config: ' + (data.message || 'Unknown error'), 'error');
    }
  } catch (error) {
    const fullError = error.stack || error.toString();
    showStatus('Error loading config: ' + error.message, 'error', fullError);
  } finally {
    showLoading(false);
  }
}

// Save configuration to server
async function saveConfig() {
  const config = state.groups.map(g => ({
    concat_dim: g.concat_dim,
    variable_prefix: g.variable_prefix,
    entry_ids: g.entry_ids
  }));

  showLoading(true);
  try {
    const configParam = encodeURIComponent(JSON.stringify(config));
    const url = `/set_tiled_input_config?config=${configParam}`;
    const res = await authenticatedFetch(url, { method: 'GET' });
    const data = await res.json();
    
    if (data.status === 'success') {
      showStatus('Config saved successfully', 'success');
    } else {
      showStatus('Failed to save config: ' + (data.message || 'Unknown error'), 'error');
    }
  } catch (error) {
    const fullError = error.stack || error.toString();
    showStatus('Error saving config: ' + error.message, 'error', fullError);
  } finally {
    showLoading(false);
  }
}

// Add a new group
function addGroup() {
  const newGroup = {
    concat_dim: '',
    variable_prefix: '',
    entry_ids: []
  };
  
  state.groups.push(newGroup);
  state.selectedGroupIndex = state.groups.length - 1;
  renderGroups();
  renderEditor();
  updateToolbarState();
}

// Remove selected group
function removeSelectedGroup() {
  if (state.selectedGroupIndex === null) return;
  
  state.groups.splice(state.selectedGroupIndex, 1);
  state.selectedGroupIndex = state.groups.length > 0 ? Math.min(state.selectedGroupIndex, state.groups.length - 1) : null;
  renderGroups();
  renderEditor();
  updateToolbarState();
}

// Select a group
function selectGroup(index) {
  state.selectedGroupIndex = index;
  renderGroups();
  renderEditor();
  updateToolbarState();
}

// Render group list
function renderGroups() {
  const container = document.getElementById('group-list');
  
  if (state.groups.length === 0) {
    container.innerHTML = '<div class="empty-state">No groups configured. Click "Add Group" to get started.</div>';
    return;
  }
  
  container.innerHTML = state.groups.map((group, index) => {
    const isSelected = index === state.selectedGroupIndex;
    const entryCount = group.entry_ids.length;
    
    return `
      <div class="group-card ${isSelected ? 'selected' : ''}" onclick="selectGroup(${index})">
        <div class="group-card-header">
          <div class="group-card-title">${group.concat_dim || 'Unnamed'}</div>
          <div class="group-card-badge">${entryCount} entries</div>
        </div>
        <div class="group-card-info">
          <span><strong>Prefix:</strong> ${group.variable_prefix || '(none)'}</span>
          <span><strong>Concat Dim:</strong> ${group.concat_dim || '(not set)'}</span>
        </div>
      </div>
    `;
  }).join('');
}

// Render group editor
function renderEditor() {
  const container = document.getElementById('editor-content');
  
  if (state.selectedGroupIndex === null || state.selectedGroupIndex >= state.groups.length) {
    container.innerHTML = '<div class="empty-state">Select a group to edit</div>';
    return;
  }
  
  const group = state.groups[state.selectedGroupIndex];
  const entryIdsText = group.entry_ids.join('\n');
  
  container.innerHTML = `
    <div class="form-group">
      <label for="concat-dim-input">Concat Dimension *</label>
      <input type="text" id="concat-dim-input" value="${escapeHtml(group.concat_dim)}" 
             placeholder="e.g., saxs, sans" />
      <div class="help-text">Name of the dimension to concatenate along</div>
    </div>
    
    <div class="form-group">
      <label for="variable-prefix-input">Variable Prefix *</label>
      <input type="text" id="variable-prefix-input" value="${escapeHtml(group.variable_prefix)}" 
             placeholder="e.g., saxs_" />
      <div class="help-text">Prefix to apply to all variable names (data vars, coords, dims)</div>
    </div>
    
    <div class="form-group">
      <label for="entry-ids-textarea">Entry IDs *</label>
      <textarea id="entry-ids-textarea" placeholder="Enter entry IDs, one per line or comma-separated">${escapeHtml(entryIdsText)}</textarea>
      <div class="help-text">Tiled entry IDs to fetch. One per line or comma-separated.</div>
      <button class="test-entry-btn" onclick="testFetchEntries()">Test Fetch Entries</button>
    </div>
  `;
  
  // Add event listeners for inputs
  document.getElementById('concat-dim-input').addEventListener('input', (e) => {
    group.concat_dim = e.target.value.trim();
    renderGroups(); // Update card display
  });
  
  document.getElementById('variable-prefix-input').addEventListener('input', (e) => {
    group.variable_prefix = e.target.value.trim();
    renderGroups(); // Update card display
  });
  
  document.getElementById('entry-ids-textarea').addEventListener('input', (e) => {
    const text = e.target.value.trim();
    // Parse entry IDs: split by newlines or commas
    group.entry_ids = text
      .split(/[\n,]+/)
      .map(id => id.trim())
      .filter(id => id.length > 0);
    renderGroups(); // Update card display
  });
}

// Update toolbar button states
function updateToolbarState() {
  const removeBtn = document.getElementById('remove-group-btn');
  removeBtn.disabled = state.selectedGroupIndex === null;
}

// Show JSON preview modal
function showJsonPreview() {
  const config = state.groups.map(g => ({
    concat_dim: g.concat_dim,
    variable_prefix: g.variable_prefix,
    entry_ids: g.entry_ids
  }));
  
  const jsonStr = JSON.stringify(config, null, 2);
  document.getElementById('json-preview-textarea').value = jsonStr;
  
  // Validate JSON
  const validationStatus = document.getElementById('json-validation-status');
  try {
    JSON.parse(jsonStr);
    validationStatus.textContent = '✓ Valid JSON';
    validationStatus.className = 'valid';
  } catch (e) {
    validationStatus.textContent = '✗ Invalid JSON: ' + e.message;
    validationStatus.className = 'invalid';
  }
  
  document.getElementById('json-preview-modal').classList.remove('hidden');
}

// Copy JSON to clipboard
async function copyJsonToClipboard() {
  const textarea = document.getElementById('json-preview-textarea');
  textarea.select();
  try {
    await navigator.clipboard.writeText(textarea.value);
    showStatus('JSON copied to clipboard', 'success');
  } catch (err) {
    showStatus('Failed to copy: ' + err.message, 'error');
  }
}

// Test fetch entries
async function testFetchEntries() {
  if (state.selectedGroupIndex === null) return;
  
  const group = state.groups[state.selectedGroupIndex];
  if (group.entry_ids.length === 0) {
    showStatus('No entry IDs to test', 'error');
    return;
  }
  
  showLoading(true);
  const results = [];
  
  for (const entryId of group.entry_ids) {
    try {
      const res = await authenticatedFetch(`/test_fetch_entry?entry_id=${encodeURIComponent(entryId)}`, {
        method: 'GET'
      });
      const data = await res.json();
      
      if (data.status === 'success') {
        results.push({ entryId, success: true, metadata: data.metadata });
      } else {
        results.push({ entryId, success: false, error: data.message || 'Unknown error' });
      }
    } catch (error) {
      results.push({ entryId, success: false, error: error.message });
    }
  }
  
  showLoading(false);
  
  const successCount = results.filter(r => r.success).length;
  const failCount = results.length - successCount;
  
  if (failCount === 0) {
    showStatus(`✓ All ${results.length} entries validated successfully`, 'success');
  } else {
    const failedEntries = results.filter(r => !r.success).map(r => r.entryId).join(', ');
    showStatus(`✗ ${failCount} of ${results.length} entries failed: ${failedEntries}`, 'error');
    console.log('Test results:', results);
  }
}

// Assemble input from tiled
async function assembleInput() {
  // Validate config first
  const config = state.groups.map(g => ({
    concat_dim: g.concat_dim,
    variable_prefix: g.variable_prefix,
    entry_ids: g.entry_ids
  }));
  
  // Check for empty groups
  if (config.length === 0) {
    showStatus('No groups configured. Add at least one group.', 'error');
    return;
  }
  
  // Check each group has required fields
  for (let i = 0; i < config.length; i++) {
    const g = config[i];
    if (!g.concat_dim || !g.variable_prefix || !g.entry_ids || g.entry_ids.length === 0) {
      showStatus(`Group ${i + 1} is incomplete. Please fill all fields.`, 'error');
      return;
    }
  }
  
  // First save the config
  await saveConfig();
  
  showLoading(true);
  try {
    const res = await authenticatedFetch('/enqueue', {
      method: 'POST',
      body: JSON.stringify({
        task_name: 'assemble_input_from_tiled'
      })
    });
    
    // The /enqueue endpoint returns a plain text UUID, not JSON
    const taskUuid = await res.text();
    
    if (res.ok && taskUuid) {
      showStatus('Input assembly queued. Waiting for completion...', 'info');
      
      // Poll for completion
      pollTaskStatus(taskUuid);
    } else {
      showStatus('Failed to queue assembly: ' + (taskUuid || 'Unknown error'), 'error');
      showLoading(false);
    }
  } catch (error) {
    showStatus('Error assembling input: ' + error.message, 'error');
    showLoading(false);
  }
}

// Poll task status
async function pollTaskStatus(taskId) {
  const maxAttempts = 60;
  let attempts = 0;
  
  const poll = async () => {
    attempts++;
    
    try {
      // Get queue state: returns [history, running_task, queue]
      const res = await authenticatedFetch('/get_queue', { method: 'GET' });
      const queueData = await res.json();
      
      const [history, runningTask, queue] = queueData;
      
      // Check if our task is in history (completed)
      const completedTask = history.find(t => t.uuid === taskId);
      
      if (completedTask) {
        showLoading(false);
        
        // The return value is stored in meta.return_val
        const returnVal = completedTask.meta?.return_val;
        console.log('Completed task:', completedTask);
        console.log('Return value:', returnVal);
        
        if (returnVal && returnVal.status === 'success') {
          showStatus(`Input assembled successfully!`, 'success', null, returnVal);
          
          // Show the dataset preview modal
          showDatasetPreview(returnVal);
        } else if (returnVal && returnVal.status === 'error') {
          const fullError = JSON.stringify(returnVal, null, 2);
          showStatus('Assembly failed: ' + (returnVal.message || 'Unknown error'), 'error', fullError, returnVal);
        } else if (completedTask.meta?.exit_state === 'Success!') {
          // Task completed successfully but return_val might not have expected structure
          showStatus('Assembly completed', 'success', null, returnVal);
          if (returnVal) {
            showDatasetPreview(returnVal);
          }
        } else {
          // Task might have failed
          const exitState = completedTask.meta?.exit_state || 'Unknown';
          showStatus(`Assembly finished with state: ${exitState}`, 'info', null, returnVal);
        }
        return;
      }
      
      // Check if our task is currently running
      if (runningTask && runningTask.uuid === taskId) {
        // Still running, continue polling
        if (attempts < maxAttempts) {
          setTimeout(poll, 1000);
        } else {
          showLoading(false);
          showStatus('Assembly timed out. Check server logs.', 'error');
        }
        return;
      }
      
      // Check if our task is still in the queue
      const queuedTask = queue.find(t => t.uuid === taskId);
      if (queuedTask) {
        // Still in queue, continue polling
        if (attempts < maxAttempts) {
          setTimeout(poll, 1000);
        } else {
          showLoading(false);
          showStatus('Assembly timed out. Check server logs.', 'error');
        }
        return;
      }
      
      // Task not found anywhere - might have completed between polls
      // Try one more time or give up
      if (attempts < maxAttempts) {
        setTimeout(poll, 1000);
      } else {
        showLoading(false);
        showStatus('Assembly completed (task not found in queue)', 'info');
      }
    } catch (error) {
      showLoading(false);
      const fullError = error.stack || error.toString();
      showStatus('Error checking status: ' + error.message, 'error', fullError);
    }
  };
  
  poll();
}

// Show dataset preview modal
function showDatasetPreview(returnVal) {
  const modal = document.getElementById('dataset-preview-modal');
  const summaryDiv = document.getElementById('dataset-summary');
  const htmlDiv = document.getElementById('dataset-html-content');
  
  // Build summary
  const dims = returnVal.dims || {};
  const dataVars = returnVal.data_vars || [];
  const coords = returnVal.coords || [];
  
  let summaryHtml = '<h4>Dataset Summary</h4><ul>';
  summaryHtml += `<li><strong>Dimensions:</strong> ${Object.entries(dims).map(([k, v]) => `${k}: ${v}`).join(', ') || 'none'}</li>`;
  summaryHtml += `<li><strong>Data Variables (${dataVars.length}):</strong> ${dataVars.join(', ') || 'none'}</li>`;
  summaryHtml += `<li><strong>Coordinates (${coords.length}):</strong> ${coords.join(', ') || 'none'}</li>`;
  summaryHtml += '</ul>';
  
  summaryDiv.innerHTML = summaryHtml;
  
  // Set HTML content
  if (returnVal.html) {
    htmlDiv.innerHTML = returnVal.html;
  } else {
    htmlDiv.innerHTML = '<p>No HTML representation available</p>';
  }
  
  // Show modal
  modal.classList.remove('hidden');
}

// Show status message
let statusTimeout = null;
function showStatus(message, type = 'info', fullDetails = null, returnVal = null) {
  const statusEl = document.getElementById('status-message');
  
  // Clear any existing timeout
  if (statusTimeout) {
    clearTimeout(statusTimeout);
    statusTimeout = null;
  }
  
  // Create short message for status bar (truncate if too long)
  const shortMessage = message.length > 100 ? message.substring(0, 97) + '...' : message;
  
  statusEl.textContent = shortMessage;
  statusEl.className = type;
  statusEl.classList.remove('hidden');
  
  // Add to outputs pane with full details and returnVal
  addOutput(message, type, fullDetails || message, returnVal);
  
  // Use longer timeout for errors (15 seconds), shorter for success/info (5 seconds)
  const timeout = type === 'error' ? 15000 : 5000;
  
  statusTimeout = setTimeout(() => {
    statusEl.classList.add('hidden');
    statusTimeout = null;
  }, timeout);
}

// Add output to outputs pane
function addOutput(message, type = 'info', fullDetails = null, returnVal = null) {
  const timestamp = new Date().toLocaleTimeString();
  const details = fullDetails || message;
  
  // Auto-enable pretty print for errors with structured data
  let prettyPrinted = false;
  if (type === 'error' && details) {
    try {
      JSON.parse(details);
      prettyPrinted = true;
    } catch (e) {
      // Not JSON, check if it looks structured
      if (details.includes('{') && details.includes('}')) {
        prettyPrinted = true;
      }
    }
  }
  
  const output = {
    id: Date.now(),
    timestamp,
    type,
    message,
    fullDetails: details,
    prettyPrinted: prettyPrinted,
    returnVal: returnVal
  };
  
  outputs.unshift(output); // Add to beginning
  updateOutputsDisplay();
  updateOutputsCount();
  
  // Auto-expand pane if it's collapsed and this is an error
  if (type === 'error') {
    const pane = document.getElementById('outputs-pane');
    if (pane.classList.contains('collapsed')) {
      pane.classList.remove('collapsed');
      document.getElementById('toggle-outputs-pane').textContent = '▼';
    }
  }
}

// Pretty print JSON or other structured data
function prettyPrint(text) {
  // Try to parse as JSON first
  try {
    const parsed = JSON.parse(text);
    return JSON.stringify(parsed, null, 2);
  } catch (e) {
    // If not JSON, try to detect and format other structures
    // Check if it looks like a Python traceback or error
    if (text.includes('Traceback') || text.includes('File "')) {
      return text; // Already formatted
    }
    // Try to format as key-value pairs if it looks structured
    if (text.includes('{') && text.includes('}')) {
      // Try to extract and format JSON-like structures
      const jsonMatches = text.match(/\{[^}]*\}/g);
      if (jsonMatches) {
        let formatted = text;
        jsonMatches.forEach(match => {
          try {
            const parsed = JSON.parse(match);
            const pretty = JSON.stringify(parsed, null, 2);
            formatted = formatted.replace(match, pretty);
          } catch (e) {
            // Ignore if can't parse
          }
        });
        return formatted;
      }
    }
    return text;
  }
}

// Update outputs display
function updateOutputsDisplay() {
  const container = document.getElementById('outputs-list');
  
  if (outputs.length === 0) {
    container.innerHTML = '<div class="empty-state">No outputs yet</div>';
    return;
  }
  
  container.innerHTML = outputs.map(output => {
    // Default to expanded for errors, collapsed for others
    const isExpanded = output.expanded !== undefined ? output.expanded : (output.type === 'error');
    const detailsDisplay = isExpanded ? 'block' : 'none';
    const toggleText = isExpanded ? 'Hide details' : 'Show details';
    const isPrettyPrinted = output.prettyPrinted || false;
    const displayDetails = isPrettyPrinted ? prettyPrint(output.fullDetails) : output.fullDetails;
    const detailsClass = isPrettyPrinted ? 'output-item-details pretty-printed' : 'output-item-details';
    const hasReturnVal = output.returnVal !== null && output.returnVal !== undefined;
    
    return `
      <div class="output-item ${output.type}" data-id="${output.id}">
        <div class="output-item-header">
          <span class="output-item-type">${output.type}</span>
          <span class="output-item-time">${output.timestamp}</span>
        </div>
        <div class="output-item-message">${escapeHtml(output.message)}</div>
        <div class="output-item-actions">
          ${output.fullDetails !== output.message ? `
            <button class="output-item-toggle" onclick="toggleOutputDetails(${output.id})">${toggleText}</button>
            <button class="output-item-toggle" onclick="togglePrettyPrint(${output.id})">${isPrettyPrinted ? 'Raw' : 'Pretty'}</button>
          ` : ''}
          ${hasReturnVal ? `
            <button class="output-item-toggle" onclick="showReturnVal(${output.id})" style="background: #6f42c1; color: white; padding: 4px 8px; border-radius: 3px;">View ReturnVal</button>
          ` : ''}
        </div>
        ${output.fullDetails !== output.message ? `
          <div class="${detailsClass}" style="display: ${detailsDisplay}">${escapeHtml(displayDetails)}</div>
        ` : ''}
      </div>
    `;
  }).join('');
}

// Toggle output details
function toggleOutputDetails(id) {
  const output = outputs.find(o => o.id === id);
  if (output) {
    output.expanded = !output.expanded;
    updateOutputsDisplay();
  }
}

// Toggle pretty print for output
function togglePrettyPrint(id) {
  const output = outputs.find(o => o.id === id);
  if (output) {
    output.prettyPrinted = !output.prettyPrinted;
    updateOutputsDisplay();
  }
}

// Toggle outputs pane
function toggleOutputsPane() {
  const pane = document.getElementById('outputs-pane');
  const toggleBtn = document.getElementById('toggle-outputs-pane');
  
  if (pane.classList.contains('collapsed')) {
    pane.classList.remove('collapsed');
    toggleBtn.textContent = '▼';
  } else {
    pane.classList.add('collapsed');
    toggleBtn.textContent = '▲';
  }
}

// Clear outputs
function clearOutputs() {
  if (confirm('Clear all outputs?')) {
    outputs = [];
    updateOutputsDisplay();
    updateOutputsCount();
  }
}

// Update outputs count badge
function updateOutputsCount() {
  const countEl = document.getElementById('outputs-count');
  const errorCount = outputs.filter(o => o.type === 'error').length;
  countEl.textContent = outputs.length;
  
  if (errorCount > 0) {
    countEl.style.background = '#dc3545';
  } else {
    countEl.style.background = '#6f42c1';
  }
}

// Show/hide loading overlay
function showLoading(show) {
  const overlay = document.getElementById('loading-overlay');
  if (show) {
    overlay.classList.remove('hidden');
  } else {
    overlay.classList.add('hidden');
  }
}

// Escape HTML
function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// Run predict
async function runPredict() {
  // First check if predict is ready
  showLoading(true);
  try {
    const checkRes = await authenticatedFetch('/check_predict_ready', {
      method: 'GET'
    });
    const checkData = await checkRes.json();
    
    if (!checkData.ready) {
      showLoading(false);
      showStatus('Predict failed: ' + (checkData.error || 'Unknown error'), 'error');
      return;
    }
    
    // If ready, enqueue the predict task
    const res = await authenticatedFetch('/enqueue', {
      method: 'POST',
      body: JSON.stringify({
        task_name: 'predict'
      })
    });
    
    // The /enqueue endpoint returns a plain text UUID, not JSON
    const taskUuid = await res.text();
    
    if (res.ok && taskUuid) {
      showStatus('Prediction queued. Waiting for completion...', 'info');
      
      // Poll for completion
      pollPredictStatus(taskUuid);
    } else {
      showStatus('Failed to queue prediction: ' + (taskUuid || 'Unknown error'), 'error');
      showLoading(false);
    }
  } catch (error) {
    const fullError = error.stack || error.toString();
    showStatus('Error running predict: ' + error.message, 'error', fullError);
    showLoading(false);
  }
}

// Poll predict task status
async function pollPredictStatus(taskId) {
  const maxAttempts = 120; // Predict might take longer than assemble
  let attempts = 0;
  
  const poll = async () => {
    attempts++;
    
    try {
      // Get queue state: returns [history, running_task, queue]
      const res = await authenticatedFetch('/get_queue', { method: 'GET' });
      const queueData = await res.json();
      
      const [history, runningTask, queue] = queueData;
      
      // Check if our task is in history (completed)
      const completedTask = history.find(t => t.uuid === taskId);
      
      if (completedTask) {
        showLoading(false);
        
        // Check if task succeeded or failed
        const exitState = completedTask.meta?.exit_state || 'Unknown';
        
        if (exitState === 'Success!') {
          const returnVal = completedTask.meta?.return_val;
          showStatus('Prediction completed successfully!', 'success', null, returnVal);
          
          // Fetch and display results
          try {
            const resultRes = await authenticatedFetch('/last_result', {
              method: 'GET'
            });
            
            if (resultRes.ok) {
              const resultHtml = await resultRes.text();
              showPredictResults(resultHtml);
            } else {
              showStatus('Prediction completed but could not fetch results', 'info');
            }
          } catch (error) {
            showStatus('Prediction completed but error fetching results: ' + error.message, 'error');
          }
        } else {
          // Task failed - try to get error message
          const returnVal = completedTask.meta?.return_val;
          let errorMsg = 'Prediction failed';
          
          if (returnVal && typeof returnVal === 'string') {
            errorMsg = returnVal;
          } else if (returnVal && returnVal.error) {
            errorMsg = returnVal.error;
          } else if (completedTask.meta?.error) {
            errorMsg = completedTask.meta.error;
          }
          
          // Capture full error details
          const fullError = JSON.stringify({
            returnVal,
            meta: completedTask.meta,
            error: errorMsg
          }, null, 2);
          
          showStatus('Prediction failed: ' + errorMsg, 'error', fullError, returnVal);
        }
        return;
      }
      
      // Check if our task is currently running
      if (runningTask && runningTask.uuid === taskId) {
        // Still running, continue polling
        if (attempts < maxAttempts) {
          setTimeout(poll, 1000);
        } else {
          showLoading(false);
          showStatus('Prediction timed out. Check server logs.', 'error');
        }
        return;
      }
      
      // Check if our task is still in the queue
      const queuedTask = queue.find(t => t.uuid === taskId);
      if (queuedTask) {
        // Still in queue, continue polling
        if (attempts < maxAttempts) {
          setTimeout(poll, 1000);
        } else {
          showLoading(false);
          showStatus('Prediction timed out. Check server logs.', 'error');
        }
        return;
      }
      
      // Task not found anywhere - might have completed between polls
      // Try one more time or give up
      if (attempts < maxAttempts) {
        setTimeout(poll, 1000);
      } else {
        showLoading(false);
        showStatus('Prediction completed (task not found in queue)', 'info');
      }
    } catch (error) {
      showLoading(false);
      const fullError = error.stack || error.toString();
      showStatus('Error checking prediction status: ' + error.message, 'error', fullError);
    }
  };
  
  poll();
}

// Show prediction results modal
function showPredictResults(htmlContent) {
  const modal = document.getElementById('predict-results-modal');
  const contentDiv = document.getElementById('predict-results-content');
  
  // Set HTML content
  if (htmlContent) {
    contentDiv.innerHTML = htmlContent;
  } else {
    contentDiv.innerHTML = '<p>No results available</p>';
  }
  
  // Show modal
  modal.classList.remove('hidden');
}

// Show return value in modal
function showReturnVal(id) {
  const output = outputs.find(o => o.id === id);
  if (!output || !output.returnVal) {
    return;
  }
  
  const modal = document.getElementById('returnval-modal');
  const contentDiv = document.getElementById('returnval-content');
  
  // Pretty print the return value
  let formatted;
  try {
    formatted = JSON.stringify(output.returnVal, null, 2);
    // Convert \n escape sequences to actual newlines in the JSON string
    // This handles cases where string values contain \n
    formatted = formatted.replace(/\\n/g, '\n');
  } catch (e) {
    formatted = String(output.returnVal);
    // Also handle \n in non-JSON strings
    formatted = formatted.replace(/\\n/g, '\n');
  }
  
  contentDiv.innerHTML = `
    <div style="font-family: monospace; white-space: pre-wrap; background: #f8f9fa; padding: 15px; border-radius: 4px; overflow-x: auto; max-height: 70vh; overflow-y: auto;">
      ${escapeHtml(formatted)}
    </div>
  `;
  
  modal.classList.remove('hidden');
}

// Make selectGroup available globally
window.selectGroup = selectGroup;
window.testFetchEntries = testFetchEntries;
window.toggleOutputDetails = toggleOutputDetails;
window.togglePrettyPrint = togglePrettyPrint;
window.showReturnVal = showReturnVal;
