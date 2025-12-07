// State management
let state = {
  groups: [],
  selectedGroupIndex: null
};

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
  setupEventListeners();
  loadConfig();
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
    showStatus('Error loading config: ' + error.message, 'error');
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
    showStatus('Error saving config: ' + error.message, 'error');
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
          showStatus(`Input assembled successfully!`, 'success');
          
          // Show the dataset preview modal
          showDatasetPreview(returnVal);
        } else if (returnVal && returnVal.status === 'error') {
          showStatus('Assembly failed: ' + (returnVal.message || 'Unknown error'), 'error');
        } else if (completedTask.meta?.exit_state === 'Success!') {
          // Task completed successfully but return_val might not have expected structure
          showStatus('Assembly completed', 'success');
          if (returnVal) {
            showDatasetPreview(returnVal);
          }
        } else {
          // Task might have failed
          const exitState = completedTask.meta?.exit_state || 'Unknown';
          showStatus(`Assembly finished with state: ${exitState}`, 'info');
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
      showStatus('Error checking status: ' + error.message, 'error');
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
function showStatus(message, type = 'info') {
  const statusEl = document.getElementById('status-message');
  statusEl.textContent = message;
  statusEl.className = type;
  statusEl.classList.remove('hidden');
  
  setTimeout(() => {
    statusEl.classList.add('hidden');
  }, 5000);
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

// Make selectGroup available globally
window.selectGroup = selectGroup;
window.testFetchEntries = testFetchEntries;
