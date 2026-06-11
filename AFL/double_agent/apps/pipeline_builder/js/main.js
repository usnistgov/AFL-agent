// D3 setup
const svg = d3.select('#connection-svg');
const connectionsGroup = svg.append('g').attr('class', 'connections');

const opList = document.getElementById('op-list');
const canvas = document.getElementById('canvas');
const prefabSelect = document.getElementById('prefab-select');
const paramTiles = document.getElementById('param-tiles');
let counter = 0;
let selectedNode = null;
let lastDeletedNode = null;
let isLoadingPipeline = false;
let isProgrammaticConnection = false;

// Zoom variables - declare early to avoid initialization issues
let currentZoom = 1.0;
const minZoom = 0.1;
const maxZoom = 3.0;
const zoomStep = 0.1;

// Connection data
let connections = [];
let nodes = [];

// D3 helper functions
function updateSVGSize() {
  const rect = canvas.getBoundingClientRect();
  // Keep SVG at full canvas size since we're not scaling the canvas
  svg.attr('width', rect.width).attr('height', rect.height);
}

function createConnection(sourceNode, targetNode) {
  const connectionId = `conn_${sourceNode.id}_${targetNode.id}`;
  const connection = {
    id: connectionId,
    source: sourceNode,
    target: targetNode
  };
  connections.push(connection);
  return connection;
}

function removeConnection(connection) {
  const index = connections.indexOf(connection);
  if (index > -1) {
    connections.splice(index, 1);
  }
}

function drawConnections() {
  const connectionPaths = connectionsGroup.selectAll('.connection-path')
    .data(connections, d => d.id);
  
  connectionPaths.exit().remove();
  
  const pathSelection = connectionPaths.enter()
    .append('path')
    .attr('class', 'connection-path')
    .attr('marker-end', 'url(#arrowhead)')
    .on('click', function(event, d) {
      // Delete connection on click
      event.stopPropagation();
      removeConnection(d);
      drawConnections();
    })
    .merge(connectionPaths); // Merge new and existing elements
  
  // Apply path calculation to both new and existing elements
  pathSelection
    .attr('d', d => {
      const sourceRect = d.source.getBoundingClientRect();
      const targetRect = d.target.getBoundingClientRect();
      const canvasRect = canvas.getBoundingClientRect();
      
      const sourceX = sourceRect.left - canvasRect.left + sourceRect.width / 2;
      const sourceY = sourceRect.bottom - canvasRect.top;
      const targetX = targetRect.left - canvasRect.left + targetRect.width / 2;
      const targetY = targetRect.top - canvasRect.top;
      
      const midY = sourceY + (targetY - sourceY) * 0.5;
      
      return `M ${sourceX} ${sourceY} C ${sourceX} ${midY} ${targetX} ${midY} ${targetX} ${targetY}`;
    });
}

// Initialize SVG arrowhead marker
svg.append('defs').append('marker')
  .attr('id', 'arrowhead')
  .attr('viewBox', '0 -5 10 10')
  .attr('refX', 8)
  .attr('refY', 0)
  .attr('markerWidth', 6)
  .attr('markerHeight', 6)
  .attr('orient', 'auto')
  .append('path')
  .attr('d', 'M0,-5L10,0L0,5')
  .attr('fill', '#333');

// Connection drag state
let connectionDragState = {
  active: false,
  sourceNode: null,
  tempLine: null
};

function startConnectionDrag(sourceNode, event) {
  connectionDragState.active = true;
  connectionDragState.sourceNode = sourceNode;
  
  // Create temporary line for visual feedback
  const sourceRect = sourceNode.getBoundingClientRect();
  const canvasRect = canvas.getBoundingClientRect();
  const startX = sourceRect.left - canvasRect.left + sourceRect.width / 2;
  const startY = sourceRect.bottom - canvasRect.top;
  
  connectionDragState.tempLine = connectionsGroup.append('path')
    .attr('class', 'temp-connection')
    .attr('stroke', '#007acc')
    .attr('stroke-width', 2)
    .attr('fill', 'none')
    .attr('stroke-dasharray', '5,5');
  
  // Add mouse move listener to canvas
  canvas.addEventListener('mousemove', updateTempConnection);
  canvas.addEventListener('mouseup', cancelConnectionDrag);
}

function updateTempConnection(event) {
  if (!connectionDragState.active || !connectionDragState.tempLine) return;
  
  const sourceRect = connectionDragState.sourceNode.getBoundingClientRect();
  const canvasRect = canvas.getBoundingClientRect();
  const startX = sourceRect.left - canvasRect.left + sourceRect.width / 2;
  const startY = sourceRect.bottom - canvasRect.top;
  const endX = event.clientX - canvasRect.left;
  const endY = event.clientY - canvasRect.top;
  
  const midY = startY + (endY - startY) * 0.5;
  const path = `M ${startX} ${startY} C ${startX} ${midY} ${endX} ${midY} ${endX} ${endY}`;
  
  connectionDragState.tempLine.attr('d', path);
}

function finishConnectionDrag(targetNode) {
  if (!connectionDragState.active || !connectionDragState.sourceNode) return;
  
  const sourceNode = connectionDragState.sourceNode;
  
  // Only create connection if source and target are different
  if (sourceNode !== targetNode) {
    // Check if connection already exists
    const existingConnection = connections.find(c => 
      c.source === sourceNode && c.target === targetNode
    );
    
    if (!existingConnection) {
      if (!isLoadingPipeline && !isProgrammaticConnection) {
        handleNewConnection(sourceNode, targetNode);
      }
      createConnection(sourceNode, targetNode);
      drawConnections();
    }
  }
  
  cancelConnectionDrag();
}

function cancelConnectionDrag() {
  if (connectionDragState.tempLine) {
    connectionDragState.tempLine.remove();
  }
  
  connectionDragState.active = false;
  connectionDragState.sourceNode = null;
  connectionDragState.tempLine = null;
  
  canvas.removeEventListener('mousemove', updateTempConnection);
  canvas.removeEventListener('mouseup', cancelConnectionDrag);
}

// Initialize SVG size
updateSVGSize();
window.addEventListener('resize', updateSVGSize);

// Dictionary editor functionality
let dictEditorState = {
  currentInput: null,
  currentParamName: '',
  currentData: {},
  isRawView: false
};

let listEditorState = {
  currentInput: null,
  currentParamName: '',
  currentData: [],
  isRawView: false
};

function isDictParameter(paramName, paramType) {
  // Check if parameter type indicates it's a dictionary
  return paramType === 'dict' || paramType === 'Dict' || 
         paramName.toLowerCase().includes('dict') ||
         paramName.toLowerCase().includes('config') ||
         paramName.toLowerCase().includes('params') ||
         paramName.toLowerCase().includes('options');
}

function isListParameter(paramName, paramType) {
  // Check if parameter type indicates it's a list/array
  return paramType === 'list' || paramType === 'List' ||
         paramName.toLowerCase().includes('list') ||
         paramName.toLowerCase().includes('array') ||
         paramName.toLowerCase().includes('items') ||
         paramName.toLowerCase().includes('tags') ||
         paramName.toLowerCase().includes('variables') && paramName.toLowerCase().includes('s');
}

function updateRequiredFieldStyling(inputElement) {
  // Find the corresponding label for this input
  const paramGroup = inputElement.closest('.param-group');
  if (!paramGroup) return;
  
  const label = paramGroup.querySelector('label');
  if (!label) return;
  
  // Check if this is a required field (has asterisk in label)
  const isRequired = label.textContent.includes('*');
  if (!isRequired) return;
  
  // Update styling based on whether the field has content
  if (!inputElement.value || inputElement.value.trim() === '') {
    inputElement.style.borderColor = '#d32f2f';
    inputElement.style.borderWidth = '2px';
    label.style.color = '#d32f2f';
  } else {
    inputElement.style.borderColor = '';
    inputElement.style.borderWidth = '';
    label.style.color = '';
  }
}

function showDictEditor(inputElement, paramName, currentValue) {
  dictEditorState.currentInput = inputElement;
  dictEditorState.currentParamName = paramName;
  
  // Parse current value
  try {
    if (typeof currentValue === 'string') {
      dictEditorState.currentData = currentValue ? JSON.parse(currentValue) : {};
    } else {
      dictEditorState.currentData = currentValue || {};
    }
  } catch (e) {
    dictEditorState.currentData = {};
  }
  
  // Update modal title
  document.getElementById('dict-editor-title').textContent = `Edit Dictionary: ${paramName}`;
  
  // Render the editor
  renderDictEditor();
  
  // Show modal
  document.getElementById('dict-editor-modal').style.display = 'block';
}

function renderDictEditor() {
  const treeView = document.getElementById('dict-tree-view');
  const rawEditor = document.getElementById('dict-raw-editor');
  
  if (dictEditorState.isRawView) {
    rawEditor.value = JSON.stringify(dictEditorState.currentData, null, 2);
  } else {
    treeView.innerHTML = '';
    renderDictTree(dictEditorState.currentData, treeView, '');
  }
  
  validateDictData();
}

function renderDictTree(data, container, path) {
  if (typeof data !== 'object' || data === null) {
    return;
  }
  
  Object.keys(data).forEach(key => {
    const value = data[key];
    const fullPath = path ? `${path}.${key}` : key;
    
    const row = document.createElement('div');
    row.className = 'dict-key-row';
    
    const keyName = document.createElement('div');
    keyName.className = 'dict-key-name';
    keyName.textContent = key;
    row.appendChild(keyName);
    
    const keyValue = document.createElement('div');
    keyValue.className = 'dict-key-value';
    
    if (typeof value === 'object' && value !== null) {
      if (Array.isArray(value)) {
        keyValue.innerHTML = `<em>[Array with ${value.length} items]</em>`;
      } else {
        keyValue.innerHTML = `<em>{Object with ${Object.keys(value).length} keys}</em>`;
      }
    } else {
      const input = document.createElement('input');
      input.type = 'text';
      input.value = typeof value === 'string' ? value : JSON.stringify(value);
      input.style.width = '100%';
      input.addEventListener('change', (e) => {
        updateDictValue(fullPath, e.target.value);
      });
      keyValue.appendChild(input);
    }
    row.appendChild(keyValue);
    
    const keyType = document.createElement('div');
    keyType.className = 'dict-key-type';
    keyType.textContent = Array.isArray(value) ? 'array' : typeof value;
    row.appendChild(keyType);
    
    const actions = document.createElement('div');
    actions.className = 'dict-key-actions';
    
    const deleteBtn = document.createElement('button');
    deleteBtn.textContent = '×';
    deleteBtn.style.cssText = 'background: #dc3545; color: white; border: none; border-radius: 3px; padding: 2px 6px; cursor: pointer;';
    deleteBtn.addEventListener('click', () => {
      deleteDictKey(fullPath);
    });
    actions.appendChild(deleteBtn);
    
    if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
      const expandBtn = document.createElement('button');
      expandBtn.textContent = '+';
      expandBtn.style.cssText = 'background: #28a745; color: white; border: none; border-radius: 3px; padding: 2px 6px; cursor: pointer; margin-left: 5px;';
      expandBtn.addEventListener('click', () => {
        addNestedKey(fullPath);
      });
      actions.appendChild(expandBtn);
    }
    
    row.appendChild(actions);
    container.appendChild(row);
    
    // Render nested objects
    if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
      const nested = document.createElement('div');
      nested.className = 'dict-nested';
      renderDictTree(value, nested, fullPath);
      container.appendChild(nested);
    }
  });
}

function updateDictValue(path, value) {
  const keys = path.split('.');
  let current = dictEditorState.currentData;
  
  // Navigate to parent object
  for (let i = 0; i < keys.length - 1; i++) {
    if (!current[keys[i]]) current[keys[i]] = {};
    current = current[keys[i]];
  }
  
  // Parse value
  try {
    const parsed = JSON.parse(value);
    current[keys[keys.length - 1]] = parsed;
  } catch (e) {
    current[keys[keys.length - 1]] = value;
  }
  
  validateDictData();
}

function deleteDictKey(path) {
  const keys = path.split('.');
  let current = dictEditorState.currentData;
  
  // Navigate to parent object
  for (let i = 0; i < keys.length - 1; i++) {
    current = current[keys[i]];
  }
  
  delete current[keys[keys.length - 1]];
  renderDictEditor();
}

function addNestedKey(path) {
  showAddKeyDialog((keyName, valueType) => {
    const keys = path.split('.');
    let current = dictEditorState.currentData;
    
    // Navigate to target object
    for (const key of keys) {
      current = current[key];
    }
    
    current[keyName] = getDefaultValueForType(valueType);
    renderDictEditor();
  });
}

function addTopLevelKey() {
  showAddKeyDialog((keyName, valueType) => {
    dictEditorState.currentData[keyName] = getDefaultValueForType(valueType);
    renderDictEditor();
  });
}

function getDefaultValueForType(valueType) {
  switch (valueType) {
    case 'dict':
      return {};
    case 'array':
      return [];
    case 'number':
      return 0;
    case 'boolean':
      return false;
    case 'null':
      return null;
    case 'string':
    default:
      return '';
  }
}

function showAddKeyDialog(callback) {
  // Create modal overlay
  const overlay = document.createElement('div');
  overlay.style.cssText = 'position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); z-index: 4000; display: flex; align-items: center; justify-content: center;';
  
  const dialog = document.createElement('div');
  dialog.style.cssText = 'background: white; padding: 20px; border-radius: 8px; min-width: 300px; box-shadow: 0 4px 20px rgba(0,0,0,0.3);';
  
  dialog.innerHTML = `
    <h4 style="margin-top: 0; margin-bottom: 15px;">Add New Key</h4>
    <div style="margin-bottom: 15px;">
      <label style="display: block; margin-bottom: 5px; font-weight: 500;">Key Name:</label>
      <input type="text" id="add-key-name" style="width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box;" placeholder="Enter key name">
    </div>
    <div style="margin-bottom: 20px;">
      <label style="display: block; margin-bottom: 5px; font-weight: 500;">Value Type:</label>
      <select id="add-key-type" style="width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box;">
        <option value="string">String (text)</option>
        <option value="number">Number</option>
        <option value="boolean">Boolean (true/false)</option>
        <option value="null">Null</option>
        <option value="dict">Dictionary (nested object)</option>
        <option value="array">Array (list)</option>
      </select>
    </div>
    <div style="text-align: right;">
      <button id="add-key-cancel" style="margin-right: 10px; padding: 8px 16px; border: 1px solid #ccc; background: white; border-radius: 4px; cursor: pointer;">Cancel</button>
      <button id="add-key-confirm" style="padding: 8px 16px; background: #28a745; color: white; border: none; border-radius: 4px; cursor: pointer;">Add Key</button>
    </div>
  `;
  
  overlay.appendChild(dialog);
  document.body.appendChild(overlay);
  
  const keyNameInput = dialog.querySelector('#add-key-name');
  const keyTypeSelect = dialog.querySelector('#add-key-type');
  const cancelBtn = dialog.querySelector('#add-key-cancel');
  const confirmBtn = dialog.querySelector('#add-key-confirm');
  
  // Focus the key name input
  keyNameInput.focus();
  
  const cleanup = () => {
    document.body.removeChild(overlay);
  };
  
  cancelBtn.addEventListener('click', cleanup);
  
  overlay.addEventListener('click', (e) => {
    if (e.target === overlay) cleanup();
  });
  
  confirmBtn.addEventListener('click', () => {
    const keyName = keyNameInput.value.trim();
    if (!keyName) {
      keyNameInput.style.borderColor = '#dc3545';
      keyNameInput.focus();
      return;
    }
    const valueType = keyTypeSelect.value;
    cleanup();
    callback(keyName, valueType);
  });
  
  keyNameInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      confirmBtn.click();
    } else if (e.key === 'Escape') {
      cleanup();
    }
  });
}

function validateDictData() {
  const status = document.getElementById('dict-validation-status');
  try {
    JSON.stringify(dictEditorState.currentData);
    status.textContent = '✓ Valid';
    status.className = 'dict-validation-success';
    document.getElementById('dict-editor-save').disabled = false;
  } catch (e) {
    status.textContent = '✗ Invalid: ' + e.message;
    status.className = 'dict-validation-error';
    document.getElementById('dict-editor-save').disabled = true;
  }
}

function saveDictEditor() {
  if (dictEditorState.isRawView) {
    try {
      dictEditorState.currentData = JSON.parse(document.getElementById('dict-raw-editor').value);
    } catch (e) {
      alert('Invalid JSON: ' + e.message);
      return;
    }
  }
  
  const jsonString = JSON.stringify(dictEditorState.currentData);
  dictEditorState.currentInput.value = jsonString;
  
  // Trigger change event to update the node and tile
  const changeEvent = new Event('change', { bubbles: true });
  dictEditorState.currentInput.dispatchEvent(changeEvent);
  
  closeDictEditor();
}

function closeDictEditor() {
  document.getElementById('dict-editor-modal').style.display = 'none';
  dictEditorState = {
    currentInput: null,
    currentParamName: '',
    currentData: {},
    isRawView: false
  };
}

function toggleDictView() {
  dictEditorState.isRawView = !dictEditorState.isRawView;
  
  const treeView = document.getElementById('dict-tree-view');
  const rawView = document.getElementById('dict-raw-view');
  const toggleBtn = document.getElementById('dict-view-toggle');
  
  if (dictEditorState.isRawView) {
    treeView.style.display = 'none';
    rawView.style.display = 'block';
    toggleBtn.textContent = 'Tree View';
    document.getElementById('dict-raw-editor').value = JSON.stringify(dictEditorState.currentData, null, 2);
  } else {
    // Parse raw editor content back to data
    try {
      dictEditorState.currentData = JSON.parse(document.getElementById('dict-raw-editor').value);
    } catch (e) {
      alert('Invalid JSON, reverting to tree view');
    }
    treeView.style.display = 'block';
    rawView.style.display = 'none';
    toggleBtn.textContent = 'Raw JSON View';
    // Clear tree view before rendering to prevent duplication
    treeView.innerHTML = '';
    renderDictTree(dictEditorState.currentData, treeView, '');
  }
  validateDictData();
}

// List editor functionality
function showListEditor(inputElement, paramName, currentValue) {
  listEditorState.currentInput = inputElement;
  listEditorState.currentParamName = paramName;
  
  // Parse current value
  try {
    if (typeof currentValue === 'string') {
      listEditorState.currentData = currentValue ? JSON.parse(currentValue) : [];
    } else if (Array.isArray(currentValue)) {
      listEditorState.currentData = currentValue;
    } else {
      listEditorState.currentData = [];
    }
  } catch (e) {
    listEditorState.currentData = [];
  }
  
  // Update modal title
  document.getElementById('list-editor-title').textContent = `Edit List: ${paramName}`;
  
  // Render the editor
  renderListEditor();
  
  // Show modal
  document.getElementById('list-editor-modal').style.display = 'block';
}

function renderListEditor() {
  const itemsView = document.getElementById('list-items-view');
  const rawEditor = document.getElementById('list-raw-editor');
  
  if (listEditorState.isRawView) {
    rawEditor.value = JSON.stringify(listEditorState.currentData, null, 2);
  } else {
    itemsView.innerHTML = '';
    renderListItems(listEditorState.currentData, itemsView);
  }
  
  validateListData();
}

function renderListItems(data, container) {
  if (!Array.isArray(data)) {
    return;
  }
  
  data.forEach((item, index) => {
    const row = document.createElement('div');
    row.className = 'list-item-row';
    
    const itemIndex = document.createElement('div');
    itemIndex.className = 'list-item-index';
    itemIndex.textContent = `[${index}]`;
    row.appendChild(itemIndex);
    
    const itemValue = document.createElement('div');
    itemValue.className = 'list-item-value';
    
    const input = document.createElement('input');
    input.type = 'text';
    input.value = typeof item === 'string' ? item : JSON.stringify(item);
    input.style.width = '100%';
    input.addEventListener('change', (e) => {
      updateListItem(index, e.target.value);
    });
    itemValue.appendChild(input);
    row.appendChild(itemValue);
    
    const itemType = document.createElement('div');
    itemType.className = 'list-item-type';
    itemType.textContent = Array.isArray(item) ? 'array' : typeof item;
    row.appendChild(itemType);
    
    const actions = document.createElement('div');
    actions.className = 'list-item-actions';
    
    const deleteBtn = document.createElement('button');
    deleteBtn.textContent = '×';
    deleteBtn.style.cssText = 'background: #dc3545; color: white; border: none; border-radius: 3px; padding: 2px 6px; cursor: pointer;';
    deleteBtn.addEventListener('click', () => {
      deleteListItem(index);
    });
    actions.appendChild(deleteBtn);
    
    const moveUpBtn = document.createElement('button');
    moveUpBtn.textContent = '↑';
    moveUpBtn.style.cssText = 'background: #6c757d; color: white; border: none; border-radius: 3px; padding: 2px 6px; cursor: pointer; margin-left: 5px;';
    moveUpBtn.disabled = index === 0;
    moveUpBtn.addEventListener('click', () => {
      moveListItem(index, index - 1);
    });
    actions.appendChild(moveUpBtn);
    
    const moveDownBtn = document.createElement('button');
    moveDownBtn.textContent = '↓';
    moveDownBtn.style.cssText = 'background: #6c757d; color: white; border: none; border-radius: 3px; padding: 2px 6px; cursor: pointer; margin-left: 5px;';
    moveDownBtn.disabled = index === data.length - 1;
    moveDownBtn.addEventListener('click', () => {
      moveListItem(index, index + 1);
    });
    actions.appendChild(moveDownBtn);
    
    row.appendChild(actions);
    container.appendChild(row);
  });
}

function updateListItem(index, value) {
  try {
    const parsed = JSON.parse(value);
    listEditorState.currentData[index] = parsed;
  } catch (e) {
    listEditorState.currentData[index] = value;
  }
  validateListData();
}

function deleteListItem(index) {
  listEditorState.currentData.splice(index, 1);
  renderListEditor();
}

function moveListItem(fromIndex, toIndex) {
  if (toIndex >= 0 && toIndex < listEditorState.currentData.length) {
    const item = listEditorState.currentData.splice(fromIndex, 1)[0];
    listEditorState.currentData.splice(toIndex, 0, item);
    renderListEditor();
  }
}

function addListItem() {
  const value = prompt('Enter new item value:');
  if (value !== null) {
    try {
      const parsed = JSON.parse(value);
      listEditorState.currentData.push(parsed);
    } catch (e) {
      listEditorState.currentData.push(value);
    }
    renderListEditor();
  }
}

function validateListData() {
  const status = document.getElementById('list-validation-status');
  try {
    JSON.stringify(listEditorState.currentData);
    status.textContent = '✓ Valid';
    status.className = 'list-validation-success';
    document.getElementById('list-editor-save').disabled = false;
  } catch (e) {
    status.textContent = '✗ Invalid: ' + e.message;
    status.className = 'list-validation-error';
    document.getElementById('list-editor-save').disabled = true;
  }
}

function saveListEditor() {
  if (listEditorState.isRawView) {
    try {
      listEditorState.currentData = JSON.parse(document.getElementById('list-raw-editor').value);
    } catch (e) {
      alert('Invalid JSON: ' + e.message);
      return;
    }
  }
  
  const jsonString = JSON.stringify(listEditorState.currentData);
  listEditorState.currentInput.value = jsonString;
  
  // Trigger change event to update the node and tile
  const changeEvent = new Event('change', { bubbles: true });
  listEditorState.currentInput.dispatchEvent(changeEvent);
  
  closeListEditor();
}

function closeListEditor() {
  document.getElementById('list-editor-modal').style.display = 'none';
  listEditorState = {
    currentInput: null,
    currentParamName: '',
    currentData: [],
    isRawView: false
  };
}

function toggleListView() {
  listEditorState.isRawView = !listEditorState.isRawView;
  
  const itemsView = document.getElementById('list-items-view');
  const rawView = document.getElementById('list-raw-view');
  const toggleBtn = document.getElementById('list-view-toggle');
  
  if (listEditorState.isRawView) {
    itemsView.style.display = 'none';
    rawView.style.display = 'block';
    toggleBtn.textContent = 'Items View';
    document.getElementById('list-raw-editor').value = JSON.stringify(listEditorState.currentData, null, 2);
  } else {
    // Parse raw editor content back to data
    try {
      listEditorState.currentData = JSON.parse(document.getElementById('list-raw-editor').value);
    } catch (e) {
      alert('Invalid JSON, reverting to items view');
    }
    itemsView.style.display = 'block';
    rawView.style.display = 'none';
    toggleBtn.textContent = 'Raw JSON View';
    // Clear items view before rendering to prevent duplication
    itemsView.innerHTML = '';
    renderListItems(listEditorState.currentData, itemsView);
  }
  validateListData();
}

// Copy-paste functionality variables
let selectedNodes = new Set();
let clipboard = [];
let lastMouseX = 0;
let lastMouseY = 0;

// UI elements
const prefabModal = document.getElementById('prefab-modal');
const loadPrefabModalBtn = document.getElementById('load-prefab-btn');


canvas.addEventListener('dragover', e => e.preventDefault());
canvas.addEventListener('drop', e => {
  e.preventDefault();
  const fqcn = e.dataTransfer.getData('text/plain');
  const opTemplate = document.querySelector(`[data-fqcn="${fqcn}"]`);
  const params = JSON.parse(opTemplate.dataset.params);
  const metadata = JSON.parse(opTemplate.dataset.metadata);
  
  // Use zoom-aware coordinates
  const coords = getCanvasMouseCoordinates(e);
  const node = addNode(fqcn, params, coords.x, coords.y, metadata);
  
  // Assign a temporary index for new nodes
  const existingNodes = document.querySelectorAll('.node');
  node.dataset.opIndex = existingNodes.length - 1;
  
  // Don't automatically update connections when dropping new nodes
  // Let users manually connect them via the connection dialog
});

function addNode(fqcn, params, x, y, metadata = null) {
  const node = document.createElement('div');
  node.className = 'node';
  node.id = 'node' + (counter++);
  
  // Set base position (unzoomed coordinates)
  const baseX = x - currentPanX;
  const baseY = y - currentPanY;
  node.dataset.baseLeft = baseX;
  node.dataset.baseTop = baseY;
  
  // Set actual position (zoomed coordinates)
  node.style.left = (baseX * currentZoom) + 'px';
  node.style.top = (baseY * currentZoom) + 'px';
  node.style.transform = `scale(${currentZoom})`;
  node.style.transformOrigin = '0 0';
  
  node.dataset.fqcn = fqcn;

  const title = document.createElement('div');
  title.className = 'node-title';
  title.textContent = params.name || fqcn.split('.').pop();
  node.appendChild(title);

  // Info button for docstring
  if (metadata && metadata.docstring !== undefined) {
    const infoBtnNode = document.createElement('div');
    infoBtnNode.className = 'info-btn';
    infoBtnNode.textContent = 'i';
    infoBtnNode.addEventListener('click', (e) => {
      e.stopPropagation();
      showDocModal(metadata.docstring || '', fqcn.split('.').pop());
    });
    node.appendChild(infoBtnNode);
  }

  const varsDiv = document.createElement('div');
  varsDiv.className = 'node-vars';
  
  // Create input fields for input parameters dynamically
  if (metadata && metadata.input_params && metadata.input_params.length > 0) {
    metadata.input_params.forEach(paramName => {
      const inputGroup = document.createElement('div');
      inputGroup.innerHTML = `<label>${paramName}:</label><input data-input="${paramName}" type="text" placeholder="${paramName}">`;
      varsDiv.appendChild(inputGroup);
      
      const inputField = inputGroup.querySelector('input');
      inputField.addEventListener('change', (e) => {
        updateNodeAndTileInputs(node.id, paramName, e.target.value);
        // Don't automatically update connections when changing input variables
        // Users should manually connect nodes via the connection dialog
      });
    });
  }
  
  // Create input fields for output parameters dynamically
  if (metadata && metadata.output_params && metadata.output_params.length > 0) {
    metadata.output_params.forEach(paramName => {
      const outputGroup = document.createElement('div');
      outputGroup.innerHTML = `<label>${paramName}:</label><input data-output="${paramName}" type="text" placeholder="${paramName}">`;
      varsDiv.appendChild(outputGroup);
      
      const outputField = outputGroup.querySelector('input');
      outputField.addEventListener('change', (e) => {
        updateNodeAndTileInputs(node.id, paramName, e.target.value);
        // Don't automatically update connections when changing output variables
        // Users should manually connect nodes via the connection dialog
      });
    });
  }
  
  // Fallback: if no metadata, create generic input/output fields
  if (!metadata || ((!metadata.input_params || metadata.input_params.length === 0) && 
                    (!metadata.output_params || metadata.output_params.length === 0))) {
    const inputGroup = document.createElement('div');
    inputGroup.innerHTML = '<label>Input Variable:</label><input data-input="input_variable" type="text" placeholder="input_var">';
    varsDiv.appendChild(inputGroup);
    
    const outputGroup = document.createElement('div');
    outputGroup.innerHTML = '<label>Output Variable:</label><input data-output="output_variable" type="text" placeholder="output_var">';
    varsDiv.appendChild(outputGroup);
    
    // Add change listeners to generic input/output variable fields
    const inputField = inputGroup.querySelector('input');
    const outputField = outputGroup.querySelector('input');
    inputField.addEventListener('change', (e) => {
      updateNodeAndTileInputs(node.id, 'input_variable', e.target.value);
      // Don't automatically update connections when changing input variables
      // Users should manually connect nodes via the connection dialog
    });
    outputField.addEventListener('change', (e) => {
      updateNodeAndTileInputs(node.id, 'output_variable', e.target.value);
      // Don't automatically update connections when changing output variables
      // Users should manually connect nodes via the connection dialog
    });
  }
  
  node.appendChild(varsDiv);

  // Create connectors
  const outAnchor = document.createElement('div');
  outAnchor.className = 'connector output';
  outAnchor.dataset.role = 'out';
  node.appendChild(outAnchor);

  const inAnchor = document.createElement('div');
  inAnchor.className = 'connector input';
  inAnchor.dataset.role = 'in';
  node.appendChild(inAnchor);
  
  // Add connection event handlers
  outAnchor.addEventListener('mousedown', (e) => {
    e.stopPropagation();
    startConnectionDrag(node, e);
  });
  
  inAnchor.addEventListener('mouseenter', (e) => {
    if (connectionDragState.active) {
      inAnchor.classList.add('connection-target');
    }
  });
  
  inAnchor.addEventListener('mouseleave', (e) => {
    inAnchor.classList.remove('connection-target');
  });
  
  inAnchor.addEventListener('mouseup', (e) => {
    if (connectionDragState.active) {
      e.stopPropagation();
      finishConnectionDrag(node);
    }
  });

  // Create delete button
  const deleteBtn = document.createElement('div');
  deleteBtn.className = 'delete-node';
  deleteBtn.textContent = '×';
  deleteBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    deleteNode(node);
  });
  node.appendChild(deleteBtn);

  // Add click handler for selection with multi-select support
  node.addEventListener('click', (e) => {
    e.stopPropagation();
    
    if (e.ctrlKey || e.metaKey) {
      // Multi-select mode
      toggleNodeSelection(node, true);
    } else {
      // Single select mode
      toggleNodeSelection(node, false);
    }
  });

  canvas.appendChild(node);
  
  // Add to nodes array
  nodes.push(node);
  
  // Make node draggable with D3
  const drag = d3.drag()
    .on('start', function(event) {
      if (selectedNodes.size > 1 && selectedNodes.has(node)) {
        // Capture original base positions for each selected node
        groupDragData.active = true;
        groupDragData.draggingNode = node;
        groupDragData.originals.clear();
        selectedNodes.forEach(n => {
          groupDragData.originals.set(n, {
            baseLeft: parseFloat(n.dataset.baseLeft) || 0,
            baseTop: parseFloat(n.dataset.baseTop) || 0
          });
        });
      } else {
        groupDragData.active = false;
      }
    })
    .on('drag', function(event) {
      // Update base position first
      const currentBaseLeft = parseFloat(node.dataset.baseLeft) || 0;
      const currentBaseTop = parseFloat(node.dataset.baseTop) || 0;
      const newBaseLeft = currentBaseLeft + (event.dx / currentZoom);
      const newBaseTop = currentBaseTop + (event.dy / currentZoom);
      
      node.dataset.baseLeft = newBaseLeft;
      node.dataset.baseTop = newBaseTop;
      
      // Update visual position
      node.style.left = (newBaseLeft * currentZoom) + 'px';
      node.style.top = (newBaseTop * currentZoom) + 'px';
      
      if (groupDragData.active && groupDragData.draggingNode === node) {
        const orig = groupDragData.originals.get(node);
        if (orig) {
          const deltaBaseX = newBaseLeft - orig.baseLeft;
          const deltaBaseY = newBaseTop - orig.baseTop;
          groupDragData.originals.forEach((pos, n) => {
            if (n !== node) {
              const newGroupBaseLeft = pos.baseLeft + deltaBaseX;
              const newGroupBaseTop = pos.baseTop + deltaBaseY;
              
              n.dataset.baseLeft = newGroupBaseLeft;
              n.dataset.baseTop = newGroupBaseTop;
              n.style.left = (newGroupBaseLeft * currentZoom) + 'px';
              n.style.top = (newGroupBaseTop * currentZoom) + 'px';
            }
          });
        }
      }
      
      drawConnections();
    })
    .on('end', function(event) {
      groupDragData.active = false;
      groupDragData.draggingNode = null;
      updateEdgeGlow();
    });
    
  d3.select(node).call(drag);
  
  // Create parameter tile
  const tile = createParamTile(node, fqcn, params, metadata);
  
  // Sync tile inputs with node inputs for input/output parameters
  if (metadata && metadata.input_params) {
    metadata.input_params.forEach(paramName => {
      const nodeInput = node.querySelector(`input[data-input="${paramName}"]`);
      const tileInput = tile.querySelector(`input[data-param="${paramName}"]`);
      if (params[paramName] && nodeInput && tileInput) {
        nodeInput.value = params[paramName];
        tileInput.value = params[paramName];
      }
    });
  }
  
  if (metadata && metadata.output_params) {
    metadata.output_params.forEach(paramName => {
      const nodeOutput = node.querySelector(`input[data-output="${paramName}"]`);
      const tileOutput = tile.querySelector(`input[data-param="${paramName}"]`);
      if (params[paramName] && nodeOutput && tileOutput) {
        nodeOutput.value = params[paramName];
        tileOutput.value = params[paramName];
      }
    });
  }
  
  // Fallback for generic input/output variables
  if (!metadata || ((!metadata.input_params || metadata.input_params.length === 0) && 
                    (!metadata.output_params || metadata.output_params.length === 0))) {
    const nodeInput = node.querySelector('input[data-input="input_variable"]');
    const nodeOutput = node.querySelector('input[data-output="output_variable"]');
    const tileInput = tile.querySelector('input[data-param="input_variable"]');
    const tileOutput = tile.querySelector('input[data-param="output_variable"]');

    if (params.input_variable && nodeInput && tileInput) {
      nodeInput.value = params.input_variable;
      tileInput.value = params.input_variable;
    }
    if (params.output_variable && nodeOutput && tileOutput) {
      nodeOutput.value = params.output_variable;
      tileOutput.value = params.output_variable;
    }
  }
  
  return node;
}

function buildOps() {
  const nodes = Array.from(document.querySelectorAll('.node:not(.deleted)'));
  
  // Sort nodes by their current visual order (top to bottom)
  nodes.sort((a, b) => {
    const aTop = parseInt(a.style.top);
    const bTop = parseInt(b.style.top);
    return aTop - bTop;
  });
  
  const ops = [];
  nodes.forEach((node, index) => {
    // Update the node's index to match the current order
    node.dataset.opIndex = index;
    
    const args = {};
    
    // Get parameters from the param tile
    const paramInputs = document.querySelectorAll(`input[data-node-id="${node.id}"][data-param]`);
    paramInputs.forEach(inp => {
      if (inp.value) args[inp.dataset.param] = parseValue(inp.value);
    });
    
    ops.push({class: node.dataset.fqcn, args: args});
  });
  return ops;
}

function parseValue(value) {
    const trimmed = value.trim();
    
    // Handle JSON objects and arrays
    if ((trimmed.startsWith('{') && trimmed.endsWith('}')) || (trimmed.startsWith('[') && trimmed.endsWith(']'))) {
        try {
            // Attempt to parse as JSON
            return JSON.parse(trimmed);
        } catch (e) {
            // Fallback to string if not valid JSON
            return value;
        }
    }
    
    // Handle null - must check before other primitives
    if (trimmed === 'null') {
        return null;
    }
    
    // Handle booleans
    if (trimmed === 'true') {
        return true;
    }
    if (trimmed === 'false') {
        return false;
    }
    
    // Handle numbers (integers and floats)
    if (/^-?\d+$/.test(trimmed)) {
        // Integer
        return parseInt(trimmed, 10);
    }
    if (/^-?\d+\.?\d*(?:[eE][+-]?\d+)?$/.test(trimmed) && !isNaN(parseFloat(trimmed))) {
        // Float or scientific notation
        return parseFloat(trimmed);
    }
    
    return value;
}

// Zoom functionality
function applyZoom(zoomLevel = currentZoom) {
  // Apply zoom to all nodes instead of scaling the canvas
  const nodes = document.querySelectorAll('.node');
  nodes.forEach(node => {
    let baseLeft, baseTop;
    
    // Initialize base positions if not already set
    if (!node.dataset.baseLeft || !node.dataset.baseTop) {
      // If this is the first time, use current visual position divided by old zoom as base
      const currentLeft = parseFloat(node.style.left) || 0;
      const currentTop = parseFloat(node.style.top) || 0;
      baseLeft = currentLeft / (currentZoom || 1);
      baseTop = currentTop / (currentZoom || 1);
      node.dataset.baseLeft = baseLeft;
      node.dataset.baseTop = baseTop;
    } else {
      baseLeft = parseFloat(node.dataset.baseLeft);
      baseTop = parseFloat(node.dataset.baseTop);
    }
    
    // Apply zoom to position and scale
    node.style.left = (baseLeft * zoomLevel) + 'px';
    node.style.top = (baseTop * zoomLevel) + 'px';
    node.style.transform = `scale(${zoomLevel})`;
    node.style.transformOrigin = '0 0';
  });
  
  // Update SVG size to match canvas (no scaling needed)
  updateSVGSize();
  
  // Redraw connections with new positions
  drawConnections();
  
  // Update zoom level display
  document.getElementById('zoom-level').textContent = Math.round(zoomLevel * 100) + '%';
  
  // Update button states
  document.getElementById('zoom-in').disabled = zoomLevel >= maxZoom;
  document.getElementById('zoom-out').disabled = zoomLevel <= minZoom;
}

function zoomIn() {
  if (currentZoom < maxZoom) {
    currentZoom = Math.min(maxZoom, currentZoom + zoomStep);
    applyZoom(currentZoom);
    updateEdgeGlow();
  }
}

function zoomOut() {
  if (currentZoom > minZoom) {
    currentZoom = Math.max(minZoom, currentZoom - zoomStep);
    applyZoom(currentZoom);
    updateEdgeGlow();
  }
}

function resetZoom() {
  currentZoom = 1.0;
  applyZoom(currentZoom);
  updateEdgeGlow();
}

function fitToView() {
  const nodes = document.querySelectorAll('.node:not(.deleted)');
  if (nodes.length === 0) {
    resetZoom();
    return;
  }
  
  // Calculate bounding box using base positions (unzoomed coordinates)
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  
  nodes.forEach(node => {
    const baseLeft = parseFloat(node.dataset.baseLeft) || 0;
    const baseTop = parseFloat(node.dataset.baseTop) || 0;
    const width = 150; // Use standard node width since we're working with base coordinates
    const height = 100; // Use standard node height
    
    minX = Math.min(minX, baseLeft);
    minY = Math.min(minY, baseTop);
    maxX = Math.max(maxX, baseLeft + width);
    maxY = Math.max(maxY, baseTop + height);
  });
  
  // Add padding
  const padding = 50;
  minX -= padding;
  minY -= padding;
  maxX += padding;
  maxY += padding;
  
  // Calculate required zoom to fit all nodes
  const canvasRect = canvas.getBoundingClientRect();
  const contentWidth = maxX - minX;
  const contentHeight = maxY - minY;
  
  const zoomX = canvasRect.width / contentWidth;
  const zoomY = canvasRect.height / contentHeight;
  
  // Use the smaller zoom to ensure everything fits
  const targetZoom = Math.min(zoomX, zoomY, maxZoom);
  currentZoom = Math.max(minZoom, targetZoom);
  
  // Pan to center the content (working in base coordinate space)
  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;
  const targetCenterX = canvasRect.width / (2 * currentZoom);
  const targetCenterY = canvasRect.height / (2 * currentZoom);
  
  const panDeltaX = (targetCenterX - centerX) * currentZoom;
  const panDeltaY = (targetCenterY - centerY) * currentZoom;
  
  // Apply zoom first
  applyZoom(currentZoom);
  
  // Then pan to center
  panAllNodes(panDeltaX, panDeltaY);
  
  updateEdgeGlow();
}

function getCanvasMouseCoordinates(event) {
  const canvasRect = canvas.getBoundingClientRect();
  
  return {
    x: (event.clientX - canvasRect.left) / currentZoom,
    y: (event.clientY - canvasRect.top) / currentZoom
  };
}

// Canvas panning functionality
function animatedPanAllNodes(deltaX, deltaY, duration = 500) {
  const nodes = document.querySelectorAll('.node');
  const startTime = performance.now();
  
  // Temporarily enable transitions
  nodes.forEach(node => {
    node.style.transition = `left ${duration}ms ease-out, top ${duration}ms ease-out`;
  });
  
  // Apply the movement using base positions
  nodes.forEach(node => {
    const baseLeft = parseFloat(node.dataset.baseLeft) || 0;
    const baseTop = parseFloat(node.dataset.baseTop) || 0;
    const newBaseLeft = baseLeft + (deltaX / currentZoom);
    const newBaseTop = baseTop + (deltaY / currentZoom);
    
    node.dataset.baseLeft = newBaseLeft;
    node.dataset.baseTop = newBaseTop;
    
    node.style.left = (newBaseLeft * currentZoom) + 'px';
    node.style.top = (newBaseTop * currentZoom) + 'px';
  });
  
  // Update current pan position
  currentPanX += deltaX;
  currentPanY += deltaY;
  
  // Continuously update connections during animation
  function updateConnectionsAnimation() {
    const elapsed = performance.now() - startTime;
    if (elapsed < duration) {
      // Redraw connections
      drawConnections();
      
      // Continue animation
      requestAnimationFrame(updateConnectionsAnimation);
    } else {
      // Animation complete - final cleanup
      nodes.forEach(node => {
        node.style.transition = '';
      });
      drawConnections();
      updateEdgeGlow();
    }
  }
  
  // Start the connection update loop
  requestAnimationFrame(updateConnectionsAnimation);
}

function panAllNodes(deltaX, deltaY) {
  const nodeElements = document.querySelectorAll('.node');
  nodeElements.forEach(node => {
    // Update base positions (unzoomed coordinates)
    const baseLeft = parseFloat(node.dataset.baseLeft) || 0;
    const baseTop = parseFloat(node.dataset.baseTop) || 0;
    const newBaseLeft = baseLeft + (deltaX / currentZoom);
    const newBaseTop = baseTop + (deltaY / currentZoom);
    
    node.dataset.baseLeft = newBaseLeft;
    node.dataset.baseTop = newBaseTop;
    
    // Update actual positions (zoomed coordinates)
    node.style.left = (newBaseLeft * currentZoom) + 'px';
    node.style.top = (newBaseTop * currentZoom) + 'px';
  });

  
  // Update current pan position
  currentPanX += deltaX;
  currentPanY += deltaY;
  
  // Redraw connections
  drawConnections();
  updateEdgeGlow();
}

canvas.addEventListener('mousedown', (e) => {
  if (e.target === canvas) {
    isPanning = true;
    panStartX = e.clientX;
    panStartY = e.clientY;
    canvas.classList.add('dragging');
    e.preventDefault();
  }
});

canvas.addEventListener('mousemove', (e) => {
  if (isPanning) {
    const deltaX = e.clientX - panStartX;
    const deltaY = e.clientY - panStartY;
    
    panAllNodes(deltaX, deltaY);
    
    panStartX = e.clientX;
    panStartY = e.clientY;
    e.preventDefault();
  }
});

canvas.addEventListener('mouseup', (e) => {
  if (isPanning) {
    isPanning = false;
    canvas.classList.remove('dragging');
  }
});

canvas.addEventListener('mouseleave', (e) => {
  if (isPanning) {
    isPanning = false;
    canvas.classList.remove('dragging');
  }
});

// Add mouse wheel support for zoom (with Ctrl) and scrolling (without Ctrl)
canvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  
  if (e.ctrlKey || e.metaKey) {
    // Zoom functionality with Ctrl/Cmd key
    const delta = e.deltaY > 0 ? -zoomStep : zoomStep;
    const newZoom = Math.max(minZoom, Math.min(maxZoom, currentZoom + delta));
    
    if (newZoom !== currentZoom) {
      currentZoom = newZoom;
      applyZoom(currentZoom);
      updateEdgeGlow();
    }
  } else {
    // Original scrolling behavior using base positions
    const deltaY = -e.deltaY * 0.5;
    
    // Only move nodes vertically, similar to dragging behavior
    const nodes = document.querySelectorAll('.node');
    nodes.forEach(node => {
      const baseTop = parseFloat(node.dataset.baseTop) || 0;
      const newBaseTop = baseTop + (deltaY / currentZoom);
      
      // Update base position
      node.dataset.baseTop = newBaseTop;
      
      // Update visual position
      const newY = newBaseTop * currentZoom;
      
      // Prevent nodes from going to invalid positions
      if (newY > -1000 && newY < 10000) {
        node.style.top = newY + 'px';
      }
    });
    
    // Update current pan position
    currentPanY += deltaY;
    
    // Redraw connections
    drawConnections();
  }
});

      // Track mouse position for paste positioning
canvas.addEventListener('mousemove', (e) => {
  lastMouseX = e.offsetX;
  lastMouseY = e.offsetY;
});

// Clear selection when clicking on canvas
canvas.addEventListener('click', (e) => {
  if (e.target === canvas && !isPanning) {
    clearSelection();
  }
});

// Keyboard shortcuts for copy-paste
document.addEventListener('keydown', (e) => {
  // Don't trigger shortcuts when typing in input fields
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
    return;
  }
  
  if (e.ctrlKey || e.metaKey) {
    switch(e.key) {
      case 'c':
      case 'C':
        e.preventDefault();
        copySelectedNodes();
        break;
      case 'v':
      case 'V':
        e.preventDefault();
        pasteNodes();
        break;
      case 'a':
      case 'A':
        e.preventDefault();
        // Select all nodes
        clearSelection();
        document.querySelectorAll('.node:not(.deleted)').forEach(node => {
          selectedNodes.add(node);
          node.classList.add('multi-selected');
        });
        break;
      case '=':
      case '+':
        e.preventDefault();
        zoomIn();
        break;
      case '-':
      case '_':
        e.preventDefault();
        zoomOut();
        break;
      case '0':
        e.preventDefault();
        resetZoom();
        break;
    }
  }
  
  // Zoom shortcuts without Ctrl/Cmd
  if (!e.ctrlKey && !e.metaKey) {
    switch(e.key) {
      case 'f':
      case 'F':
        e.preventDefault();
        fitToView();
        break;
    }
  }
  
  // Delete selected nodes
  if (e.key === 'Delete' || e.key === 'Backspace') {
    if (selectedNodes.size > 0) {
      e.preventDefault();
      selectedNodes.forEach(node => {
        deleteNode(node);
      });
      clearSelection();
    }
  }
  
  // Escape to clear selection
  if (e.key === 'Escape') {
    clearSelection();
  }
});


// ---------------------------------------------------------------------
// Edge glow helper
// ---------------------------------------------------------------------
const glowTop = document.querySelector('.edge-glow.top');
const glowBottom = document.querySelector('.edge-glow.bottom');
const glowLeft = document.querySelector('.edge-glow.left');
const glowRight = document.querySelector('.edge-glow.right');

function updateEdgeGlow() {
   const canvasRect = canvas.getBoundingClientRect();
   const nodes = document.querySelectorAll('.node:not(.deleted)');
   let showTop = false, showBottom = false, showLeft = false, showRight = false;

   nodes.forEach(node => {
     const nodeLeft = parseInt(node.style.left) || 0;
     const nodeTop = parseInt(node.style.top) || 0;
     const nodeWidth = node.offsetWidth || 150;
     const nodeHeight = node.offsetHeight || 100;
     
     // Nodes are already positioned at their zoomed coordinates
     // Check if node extends beyond visible canvas area
     const buffer = 25;
     if (nodeTop < buffer) showTop = true;
     if (nodeTop + nodeHeight > canvasRect.height - buffer) showBottom = true;
     if (nodeLeft < buffer) showLeft = true;
     if (nodeLeft + nodeWidth > canvasRect.width - buffer) showRight = true;
   });

   glowTop.style.opacity = showTop ? 1 : 0;
   glowBottom.style.opacity = showBottom ? 1 : 0;
   glowLeft.style.opacity = showLeft ? 1 : 0;
   glowRight.style.opacity = showRight ? 1 : 0;
  
 }

// Periodic check (fallback) in case some events missed
setInterval(updateEdgeGlow, 1000);

function optimizeLayout(connectionData) {
  const nodeElements = Array.from(document.querySelectorAll('.node'));
  if (nodeElements.length <= 1) return;
  
  console.log('Applying dagre hierarchical layout');
  
  // Create a new dagre graph
  const g = new dagre.graphlib.Graph();
  
  // Set graph options for a clean hierarchical layout
  g.setGraph({
    rankdir: 'TB',    // Top to bottom layout
    nodesep: 120,     // Horizontal separation between nodes
    ranksep: 180,     // Vertical separation between ranks/layers
    marginx: 80,      // Horizontal margin
    marginy: 80       // Vertical margin
  });
  
  // Default to assigning a new object as a label for each new edge
  g.setDefaultEdgeLabel(() => ({}));
  
  // Add nodes to the graph
  nodeElements.forEach(node => {
    const nodeId = node.dataset.opIndex || node.id;
    g.setNode(nodeId, {
      width: 300,   // Node width for layout calculation
      height: 120   // Node height for layout calculation
    });
  });
  
  // Add edges to the graph
  connectionData.forEach(conn => {
    const sourceId = conn.source_index.toString();
    const targetId = conn.target_index.toString();
    g.setEdge(sourceId, targetId);
  });
  
  // Run the dagre layout algorithm
  dagre.layout(g);
  
  // Apply the computed layout to actual DOM nodes
  g.nodes().forEach(nodeId => {
    const node = nodeElements.find(n => (n.dataset.opIndex || n.id) === nodeId);
    if (node) {
      const nodeLayout = g.node(nodeId);
      
      // dagre gives center position, convert to top-left
      const baseX = nodeLayout.x - nodeLayout.width / 2;
      const baseY = nodeLayout.y - nodeLayout.height / 2;
      
      // Set base positions (unzoomed coordinates)
      node.dataset.baseLeft = baseX;
      node.dataset.baseTop = baseY;
      
      // Set visual positions (zoomed coordinates)
      node.style.left = (baseX * currentZoom) + 'px';
      node.style.top = (baseY * currentZoom) + 'px';
      node.style.transform = `scale(${currentZoom})`;
      node.style.transformOrigin = '0 0';
    }
  });
  
  // Redraw all connections
  drawConnections();
  updateEdgeGlow();
}



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

// Helper function to make authenticated requests
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

// Canvas panning variables
let isPanning = false;
let panStartX = 0;
let panStartY = 0;
let currentPanX = 0;
let currentPanY = 0;

// Search functionality variables
let allOperations = [];
const searchInput = document.getElementById('op-search');
const searchClear = document.getElementById('search-clear');
const noResults = document.getElementById('no-results');

// Simple fuzzy search function
function fuzzyMatch(pattern, text) {
  pattern = pattern.toLowerCase();
  text = text.toLowerCase();
  
  // If exact substring match, give high score
  if (text.includes(pattern)) {
    return 100;
  }
  
  // Fuzzy matching - check if all characters of pattern appear in order in text
  let patternIndex = 0;
  let score = 0;
  
  for (let i = 0; i < text.length && patternIndex < pattern.length; i++) {
    if (text[i] === pattern[patternIndex]) {
      score += (text.length - i); // Earlier matches get higher score
      patternIndex++;
    }
  }
  
  // Return score only if all pattern characters were found
  return patternIndex === pattern.length ? score : 0;
}

function filterOperations(searchTerm) {
  const operations = document.querySelectorAll('.op-template');
  let visibleCount = 0;
  
  if (!searchTerm.trim()) {
    // Show all operations if search is empty
    operations.forEach(op => {
      op.classList.remove('hidden');
      visibleCount++;
    });
    noResults.style.display = 'none';
    searchClear.style.display = 'none';
    return;
  }
  
  // Show clear button when there's search text
  searchClear.style.display = 'flex';
  
  // Score and filter operations
  const scoredOps = [];
  operations.forEach(op => {
    const name = op.textContent;
    const score = fuzzyMatch(searchTerm, name);
    
    if (score > 0) {
      scoredOps.push({ element: op, score: score, name: name });
      op.classList.remove('hidden');
      visibleCount++;
    } else {
      op.classList.add('hidden');
    }
  });
  
  // Sort by score (higher is better) and then alphabetically
  scoredOps.sort((a, b) => {
    if (b.score !== a.score) {
      return b.score - a.score;
    }
    return a.name.localeCompare(b.name);
  });
  
  // Reorder DOM elements based on search relevance
  scoredOps.forEach(item => {
    opList.appendChild(item.element);
  });
  
  // Show "no results" message if nothing found
  noResults.style.display = visibleCount === 0 ? 'block' : 'none';
}

// Set up search event listeners
searchInput.addEventListener('input', (e) => {
  filterOperations(e.target.value);
});

searchInput.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    searchInput.value = '';
    filterOperations('');
    searchInput.blur();
  }
});

searchClear.addEventListener('click', () => {
  searchInput.value = '';
  filterOperations('');
  searchInput.focus();
});

// Copy-paste functionality
function getAllVariableNames() {
  const variableNames = new Set();
  document.querySelectorAll('.node:not(.deleted)').forEach(node => {
    // Check input variables
    node.querySelectorAll('input[data-input], input[data-output]').forEach(input => {
      if (input.value.trim()) {
        variableNames.add(input.value.trim());
      }
    });
  });
  return variableNames;
}

function generateUniqueVariableName(baseName, existingNames) {
  if (!existingNames.has(baseName)) {
    return baseName;
  }
  
  let counter = 1;
  let newName;
  do {
    newName = `${baseName}_${counter}`;
    counter++;
  } while (existingNames.has(newName));
  
  return newName;
}

function copySelectedNodes() {
  if (selectedNodes.size === 0) return;
  
  clipboard = [];
  const existingNames = getAllVariableNames();
  
  selectedNodes.forEach(node => {
    // Get all parameters from the param tile
    const nodeData = {
      fqcn: node.dataset.fqcn,
      args: {},
      metadata: null
    };
    
    // Extract metadata
    const opTemplate = document.querySelector(`[data-fqcn="${node.dataset.fqcn}"]`);
    if (opTemplate && opTemplate.dataset.metadata) {
      nodeData.metadata = JSON.parse(opTemplate.dataset.metadata);
    }
    
    // Get parameters from the param tile
    const paramInputs = document.querySelectorAll(`input[data-node-id="${node.id}"][data-param]`);
    paramInputs.forEach(inp => {
      if (inp.value.trim()) {
        let value = parseValue(inp.value);
        
        // If this is a variable parameter, make it unique
        if (inp.dataset.param.includes('variable') || 
            inp.dataset.param.includes('input') || 
            inp.dataset.param.includes('output')) {
          if (typeof value === 'string') {
            value = generateUniqueVariableName(value, existingNames);
            existingNames.add(value); // Add to set so subsequent copies don't collide
          }
        }
        
        nodeData.args[inp.dataset.param] = value;
      }
    });
    
    clipboard.push(nodeData);
    
    // Visual feedback
    node.classList.add('copying');
    setTimeout(() => node.classList.remove('copying'), 300);
  });
  
  // Show copy indicator
  const indicator = document.getElementById('copy-indicator');
  indicator.textContent = `Copied ${clipboard.length} operation${clipboard.length > 1 ? 's' : ''}!`;
  indicator.style.display = 'block';
  setTimeout(() => {
    indicator.style.display = 'none';
  }, 2000);
}

function pasteNodes() {
  if (clipboard.length === 0) return;
  
  const canvasRect = canvas.getBoundingClientRect();
  const startX = canvasRect.width / 2 + currentPanX;
  const startY = canvasRect.height / 2 + currentPanY;
  
  // Clear current selection
  clearSelection();
  
  clipboard.forEach((nodeData, index) => {
    const offsetX = startX + (index * 50); // Stagger horizontally
    const offsetY = startY + (index * 20); // Slight vertical offset
    
    // Create new node
    const params = nodeData.args || {};
    const node = addNode(nodeData.fqcn, params, offsetX, offsetY, nodeData.metadata);
    
    // Update node index
    const existingNodes = document.querySelectorAll('.node');
    node.dataset.opIndex = existingNodes.length - 1;
    
    // Select the pasted node for visual feedback
    selectedNodes.add(node);
    node.classList.add('multi-selected');
  });
  
  // Update selected node for param tile display
  if (selectedNodes.size === 1) {
    selectedNode = Array.from(selectedNodes)[0];
    selectNode(selectedNode);
  }
  updateEdgeGlow();
}

     function clearSelection() {
   selectedNodes.forEach(node => {
     node.classList.remove('selected', 'multi-selected');
   });
   selectedNodes.clear();
   selectedNode = null;
   
   // Clear param tile selection
   document.querySelectorAll('.param-tile.selected').forEach(t => t.classList.remove('selected'));
   
 }

function toggleNodeSelection(node, multiSelect = false) {
  if (!multiSelect) {
    clearSelection();
  }
  
  if (selectedNodes.has(node)) {
    // Deselect
    selectedNodes.delete(node);
    node.classList.remove('selected', 'multi-selected');
    if (selectedNode === node) {
      selectedNode = null;
    }
  } else {
    // Select
    selectedNodes.add(node);
    node.classList.add('multi-selected');
    selectedNode = node; // For param tile display
  }
  
         // Update param tile display for single selection
   if (selectedNodes.size === 1) {
     selectNode(Array.from(selectedNodes)[0]);
   } else if (selectedNodes.size === 0) {
     // Clear param tile selection
     document.querySelectorAll('.param-tile.selected').forEach(t => t.classList.remove('selected'));
   }
   
 }
 
 
 // Prefab modal handlers
 loadPrefabModalBtn.addEventListener('click', () => {
   prefabModal.style.display = 'block';
 });
 
 document.getElementById('cancel-prefab').addEventListener('click', () => {
   prefabModal.style.display = 'none';
 });
 
 document.getElementById('prefab-select').addEventListener('change', (e) => {
   document.getElementById('load-prefab').disabled = !e.target.value;
 });

  async function loadOps() {
  try {
    const res = await authenticatedFetch('/pipeline_ops', {
      method: 'GET'
    });
    const payload = await res.json();
    const ops = Array.isArray(payload) ? payload : (payload.ops || []);
    const warnings = payload && Array.isArray(payload.warnings) ? payload.warnings : [];
    
    // Clear loading indicator
    opList.innerHTML = '';

    if (warnings.length > 0) {
      const warningDiv = document.createElement('div');
      warningDiv.className = 'loading-text';
      warningDiv.style.color = '#b58900';
      warningDiv.textContent = `Loaded with ${warnings.length} warning(s). Some operations may be unavailable.`;
      opList.appendChild(warningDiv);
    }
    
    ops.forEach(op => {
      const div = document.createElement('div');
      div.className = 'op-template';
      div.draggable = true;
      div.dataset.fqcn = op.fqcn;
      div.dataset.params = JSON.stringify(op.parameters);

      // Store full metadata including docstring and parameter types
      const metadataObj = {
        input_params: op.input_params || [],
        output_params: op.output_params || [],
        required_params: op.required_params || [],
        param_types: op.param_types || {},
        docstring: op.docstring || ''
      };
      div.dataset.metadata = JSON.stringify(metadataObj);

      const nameSpan = document.createElement('span');
      nameSpan.className = 'op-name';
      nameSpan.textContent = op.name;
      div.appendChild(nameSpan);

      // Add info button for docstring
      const infoBtn = document.createElement('span');
      infoBtn.className = 'op-info-btn';
      infoBtn.textContent = 'i';
      infoBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        showDocModal(op.docstring || '', op.name);
      });
      // Prevent dragging when clicking the info button
      infoBtn.addEventListener('mousedown', (e) => e.stopPropagation());
      infoBtn.addEventListener('dragstart', (e) => e.preventDefault());

      div.appendChild(infoBtn);

      div.addEventListener('dragstart', e => {
        e.dataTransfer.setData('text/plain', div.dataset.fqcn);
      });
      opList.appendChild(div);
    });
    
    // After operations are loaded, add missing info buttons to existing nodes
    addMissingInfoButtons();
    
    // After operations are loaded, apply dict/list functionality to existing parameter tiles
    applyDictListFunctionality();
    
    console.log('Pipeline operations loaded successfully');
  } catch (error) {
    opList.innerHTML = '<div class="loading-text" style="color: #e74c3c;">Failed to load pipeline operations</div>';
    console.error('Error loading pipeline ops:', error);
    throw error; // Re-throw to handle in initialization
  }
  // Reset pan position
  currentPanX = 0;
  currentPanY = 0;
  // Clear any pending undos
  lastDeletedNode = null;
  document.getElementById('undo-banner').style.display = 'none';
}

function addMissingInfoButtons() {
  // Walk through all existing nodes and add info buttons if they don't have them
  // but the corresponding operation template now has docstring metadata
  const existingNodes = document.querySelectorAll('.node:not(.deleted)');
  
  existingNodes.forEach(node => {
    // Check if this node already has an info button
    const existingInfoBtn = node.querySelector('.info-btn');
    if (existingInfoBtn) {
      return; // Already has an info button, skip
    }
    
    // Get the operation template for this node
    const fqcn = node.dataset.fqcn;
    if (!fqcn) {
      return; // No FQCN, can't find operation template
    }
    
    const opTemplate = document.querySelector(`[data-fqcn="${fqcn}"]`);
    if (!opTemplate || !opTemplate.dataset.metadata) {
      return; // No operation template or metadata found
    }
    
    let metadata;
    try {
      metadata = JSON.parse(opTemplate.dataset.metadata);
    } catch (e) {
      console.warn('Failed to parse metadata for operation:', fqcn, e);
      return;
    }
    
    // Check if the metadata has a docstring
    if (!metadata.docstring) {
      return; // No docstring available
    }
    
    // Create and add the info button
    const infoBtnNode = document.createElement('div');
    infoBtnNode.className = 'info-btn';
    infoBtnNode.textContent = 'i';
    infoBtnNode.addEventListener('click', (e) => {
      e.stopPropagation();
      showDocModal(metadata.docstring || '', fqcn.split('.').pop());
    });
    node.appendChild(infoBtnNode);
  });
}

function applyDictListFunctionality() {
  // Walk through all existing parameter tiles and apply dict/list functionality
  // if the operation metadata is now available
  const existingTiles = document.querySelectorAll('.param-tile');
  
  existingTiles.forEach(tile => {
    const nodeId = tile.dataset.nodeId;
    if (!nodeId) return;
    
    const node = document.getElementById(nodeId);
    if (!node) return;
    
    // Get the operation template for this node
    const fqcn = node.dataset.fqcn;
    if (!fqcn) return;
    
    const opTemplate = document.querySelector(`[data-fqcn="${fqcn}"]`);
    if (!opTemplate || !opTemplate.dataset.metadata) return;
    
    let metadata;
    try {
      metadata = JSON.parse(opTemplate.dataset.metadata);
    } catch (e) {
      console.warn('Failed to parse metadata for dict/list functionality:', fqcn, e);
      return;
    }
    
    // Check each parameter input in this tile
    const paramInputs = tile.querySelectorAll('input[data-param]');
    paramInputs.forEach(input => {
      const paramName = input.dataset.param;
      
      // Skip if this input already has dict/list functionality applied
      if (input.classList.contains('dict-param-input') || 
          input.classList.contains('list-param-input')) {
        return;
      }
      
      // Check if this parameter should be a dict or list
      const paramType = metadata.param_types ? metadata.param_types[paramName] : null;
      const isDict = isDictParameter(paramName, paramType);
      const isList = isListParameter(paramName, paramType);
      
      if (isDict) {
        // Apply dictionary functionality
        input.classList.add('dict-param-input');
        input.readOnly = true;
        input.title = 'Click to edit dictionary';
        
        // Remove any existing change listeners to avoid duplicates
        input.onchange = null;
        
        // Add click listener for dict editor
        input.addEventListener('click', (e) => {
          e.preventDefault();
          showDictEditor(input, paramName, input.value);
        });
        
        // Add change listener for updates
        input.addEventListener('change', (e) => {
          updateNodeAndTileInputs(nodeId, paramName, e.target.value);
          if (paramName === 'name') {
            updateNodeTitle(nodeId, e.target.value);
          }
          debounce(updateConnections, 500)();
        });
      } else if (isList) {
        // Apply list functionality
        input.classList.add('list-param-input');
        input.readOnly = true;
        input.title = 'Click to edit list';
        
        // Remove any existing change listeners to avoid duplicates
        input.onchange = null;
        
        // Add click listener for list editor
        input.addEventListener('click', (e) => {
          e.preventDefault();
          showListEditor(input, paramName, input.value);
        });
        
        // Add change listener for updates
        input.addEventListener('change', (e) => {
          updateNodeAndTileInputs(nodeId, paramName, e.target.value);
          if (paramName === 'name') {
            updateNodeTitle(nodeId, e.target.value);
          }
          debounce(updateConnections, 500)();
        });
      }
    });
  });
}

async function loadPrefabs() {
  const res = await authenticatedFetch('/prefab_names', {
    method: 'GET'
  });
  if (!res.ok) return;
  const names = await res.json();
  names.forEach(n => {
    const opt = document.createElement('option');
    opt.value = n;
    opt.textContent = n;
    prefabSelect.appendChild(opt);
  });
}

function clearCanvas() {
  // Clear D3 connections
  connections = [];
  nodes = [];
  connectionsGroup.selectAll('*').remove();
  
  // Clear DOM nodes
  canvas.querySelectorAll('.node').forEach(node => node.remove());
  paramTiles.innerHTML = '';
  counter = 0;
  selectedNode = null;
  
  // Reset pan position and zoom
  currentPanX = 0;
  currentPanY = 0;
  currentZoom = 1.0;
  applyZoom(currentZoom);
}

function loadPipeline(data) {
  isLoadingPipeline = true; // Disable connection dialog during loading
  clearCanvas();
  
  // Handle both old format (array) and new format (object with ops and connections)
  const ops = Array.isArray(data) ? data : data.ops;
  const connectionData = data.connections || [];
  
  let x = 50;
  let y = 50;
  const nodeData = [];
  const nodesByOutputVar = new Map();
  
        ops.forEach((op, index) => {

    if (!op.name) {
      op.name = op.class;
    }
    // Try to find metadata for this operation class
    const opTemplate = document.querySelector(`[data-fqcn="${op.class}"]`);
    let metadata = null;
    if (opTemplate && opTemplate.dataset.metadata) {
      try {
        metadata = JSON.parse(opTemplate.dataset.metadata);
      } catch (e) {
        console.warn('Failed to parse metadata for operation:', op.class, e);
        metadata = null;
      }
    }
    
    // If no metadata available, infer input/output parameters from operation args
    if (!metadata && op.args) {
      metadata = inferMetadataFromArgs(op.args);
    }
    
    const node = addNode(op.name || op.class, op.args || {}, x, y, metadata);
    
    // Store the operation index on the node for lookup
    node.dataset.opIndex = index;
    
    // Ensure base positions are set correctly when loading
    if (!node.dataset.baseLeft) {
      node.dataset.baseLeft = x;
      node.dataset.baseTop = y;
    }
    
    if (op.args) {
      // Update parameters in the param tile
      const paramInputs = document.querySelectorAll(`input[data-node-id="${node.id}"][data-param]`);
      paramInputs.forEach(inp => {
        const val = op.args[inp.dataset.param];
        if (val !== undefined) {
          if (typeof val === 'object') {
            inp.value = JSON.stringify(val);
          } else {
            inp.value = val;
          }
        }
      });
      
      // Handle output variables dynamically
      if (metadata && metadata.output_params) {
        metadata.output_params.forEach(paramName => {
          const outputInput = node.querySelector(`input[data-output="${paramName}"]`);
          if (op.args[paramName] && outputInput) {
            outputInput.value = op.args[paramName];
            // Map output variables to nodes for connection lookup
            nodesByOutputVar.set(op.args[paramName], node);
          }
        });
      } else if (op.args.output_variable) {
        const outputInput = node.querySelector('input[data-output="output_variable"]');
        if (outputInput) {
          outputInput.value = op.args.output_variable;
          nodesByOutputVar.set(op.args.output_variable, node);
        }
      }
      
      // Handle input variables dynamically
      if (metadata && metadata.input_params) {
        metadata.input_params.forEach(paramName => {
          const inputInput = node.querySelector(`input[data-input="${paramName}"]`);
          if (op.args[paramName] && inputInput) {
            inputInput.value = op.args[paramName];
          }
        });
      } else if (op.args.input_variable) {
        const inputInput = node.querySelector('input[data-input="input_variable"]');
        if (inputInput) {
          inputInput.value = op.args.input_variable;
        }
      }
    }
    nodeData.push({node: node, op: op, index: index});
    y += 150;
  });

        // Connect nodes using the backend-provided connection information
  isProgrammaticConnection = true;
  connectionData.forEach(conn => {
    const sourceNode = nodeData.find(n => n.index === conn.source_index);
    const targetNode = nodeData.find(n => n.index === conn.target_index);
    
    if (sourceNode && targetNode) {
      createConnection(sourceNode.node, targetNode.node);
    }
  });
  isProgrammaticConnection = false;
  
  // Draw all connections
  drawConnections();
  
  // Optimize layout after loading pipeline
  if (connectionData.length > 0) {
    optimizeLayout(connectionData);
  }
  
  isLoadingPipeline = false; // Re-enable connection dialog
  updateEdgeGlow();
}

async function loadCurrentPipeline() {
  const res = await authenticatedFetch('/current_pipeline', { method: 'GET' });
  if (!res.ok) return;

  const data = await res.json();

  if (Array.isArray(data)) {
    // Legacy format without connection info
    if (data.length === 0) return;
    let connections = [];
    try {
      const opsParam = encodeURIComponent(JSON.stringify(data));
      const connRes = await authenticatedFetch(`/analyze_pipeline?ops=${opsParam}`, { method: 'GET' });
      if (connRes.ok) {
        const connData = await connRes.json();
        
        // Check for pipeline analysis errors
        if (connData.status === 'error' && connData.errors && connData.errors.length > 0) {
          showPipelineErrors(connData.errors, connData.message);
          connections = []; // Clear connections if there are errors
        } else {
          connections = connData.connections || [];
        }
      }
    } catch (e) {
      console.warn('Failed to analyze pipeline for connections:', e);
    }
    loadPipeline({ ops: data, connections });
  } else if (data && Array.isArray(data.ops) && data.ops.length) {
    loadPipeline(data);
  }
}

function selectNode(node) {
  // Remove selection from all nodes and tiles
  document.querySelectorAll('.node.selected').forEach(n => n.classList.remove('selected'));
  document.querySelectorAll('.param-tile.selected').forEach(t => t.classList.remove('selected'));
  document.querySelectorAll('.node.multi-selected').forEach(n => n.classList.remove('multi-selected'));

  // Select the new node
  selectedNode = node;
  node.classList.add('selected');
  node.classList.add('multi-selected');
  
  // Find and select corresponding param tile
  const tile = document.querySelector(`.param-tile[data-node-id="${node.id}"]`);
  if (tile) {
    tile.classList.add('selected');
    tile.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }
}

function selectNodeFromTile(nodeId) {
  const node = document.getElementById(nodeId);
  if (node) {
    // Center the node in the canvas viewport
    const canvasRect = canvas.getBoundingClientRect();
    const nodeLeft = parseInt(node.style.left) || 0;
    const nodeTop = parseInt(node.style.top) || 0;
    const nodeWidth = node.offsetWidth || 150;
    const nodeHeight = node.offsetHeight || 100;
    
    // Calculate where the node center should be to center it in viewport
    const targetCenterX = canvasRect.width / 2;
    const targetCenterY = canvasRect.height / 2;
    
    // Calculate current node center position
    const nodeCenterX = nodeLeft + nodeWidth / 2;
    const nodeCenterY = nodeTop + nodeHeight / 2;
    
    // Calculate the pan offset needed to center the node
    const deltaX = targetCenterX - nodeCenterX;
    const deltaY = targetCenterY - nodeCenterY;
    
    // Apply the pan to all nodes
    animatedPanAllNodes(deltaX, deltaY);
    
    selectNode(node);
  }
}

function createParamTile(node, fqcn, params, opMetadata) {
  const tile = document.createElement('div');
  tile.className = 'param-tile';
  tile.dataset.nodeId = node.id;
  
  const title = document.createElement('div');
  title.className = 'param-tile-title';
  title.textContent = fqcn.split('.').pop();
  tile.appendChild(title);

  // Helper function to create a parameter input group
  function createParamGroup(key, val, isRequired = false) {
    const group = document.createElement('div');
    group.className = 'param-group';
    
    const label = document.createElement('label');
    label.textContent = isRequired ? `${key} *` : key;
    if (isRequired) {
      label.style.fontWeight = 'bold';
    }
    
    const input = document.createElement('input');
    input.type = 'text';
    input.dataset.param = key;
    input.dataset.nodeId = node.id;
    
    // Function to update required field styling based on content
    function updateRequiredStyling() {
      if (isRequired) {
        if (!input.value || input.value.trim() === '') {
          input.style.borderColor = '#d32f2f';
          input.style.borderWidth = '2px';
          label.style.color = '#d32f2f';
        } else {
          input.style.borderColor = '';
          input.style.borderWidth = '';
          label.style.color = '';
        }
      }
    }
    
    // Mark input/output parameters for special handling
    if (opMetadata && opMetadata.input_params && opMetadata.input_params.includes(key)) {
      input.dataset.isInputParam = 'true';
    }
    if (opMetadata && opMetadata.output_params && opMetadata.output_params.includes(key)) {
      input.dataset.isOutputParam = 'true';
    }
    
    // Check if this is a dictionary or list parameter
    const paramType = opMetadata && opMetadata.param_types ? opMetadata.param_types[key] : null;
    const isDict = isDictParameter(key, paramType);
    const isList = isListParameter(key, paramType);
    
    if (val !== null && val !== undefined) {
      if (typeof val === 'object') {
        input.value = JSON.stringify(val);
      } else {
        input.value = val;
      }
    }
    
    // Set initial styling for required fields
    updateRequiredStyling();
    
    // Special styling and behavior for dictionary parameters
    if (isDict) {
      input.classList.add('dict-param-input');
      input.readOnly = true;
      input.title = 'Click to edit dictionary';
      
      input.addEventListener('click', (e) => {
        e.preventDefault();
        showDictEditor(input, key, input.value);
      });
    }
    // Special styling and behavior for list parameters
    else if (isList) {
      input.classList.add('list-param-input');
      input.readOnly = true;
      input.title = 'Click to edit list';
      
      input.addEventListener('click', (e) => {
        e.preventDefault();
        showListEditor(input, key, input.value);
      });
    }
    
    // Add change listener to update connections when parameters change
    input.addEventListener('change', (e) => {
      updateNodeAndTileInputs(node.id, key, e.target.value);
      
      // Update required field styling when value changes
      updateRequiredStyling();
      
      // Special handling for 'name' parameter to update node title
      if (key === 'name') {
        updateNodeTitle(node.id, e.target.value);
      }
      
      // debounce(updateConnections, 500)();
    });
    
    // Add input listener for real-time styling updates on required fields
    if (isRequired) {
      input.addEventListener('input', updateRequiredStyling);
    }
    
    group.appendChild(label);
    group.appendChild(input);
    return group;
  }

  // Get required parameters from metadata
  const requiredParams = opMetadata && opMetadata.required_params ? opMetadata.required_params : [];
  
  // Always create 'name' field first
  const nameValue = params.name || '';
  const nameGroup = createParamGroup('name', nameValue, false); // Name is not marked as required even if it technically is
  tile.appendChild(nameGroup);

  // Separate parameters into required and optional (excluding 'name' to avoid duplication)
  const paramEntries = Object.entries(params).filter(([key, val]) => key !== 'name');
  const requiredParamEntries = paramEntries.filter(([key, val]) => requiredParams.includes(key));
  const optionalParamEntries = paramEntries.filter(([key, val]) => !requiredParams.includes(key));

  // Add required parameters next (with asterisk)
  requiredParamEntries.forEach(([key, val]) => {
    const group = createParamGroup(key, val, true);
    tile.appendChild(group);
  });

  // Add optional parameters last
  optionalParamEntries.forEach(([key, val]) => {
    const group = createParamGroup(key, val, false);
    tile.appendChild(group);
  });

  tile.addEventListener('click', () => selectNodeFromTile(node.id));
  paramTiles.appendChild(tile);
  return tile;
}

// Function to infer input/output parameters from operation arguments
function inferMetadataFromArgs(args) {
  const input_params = [];
  const output_params = [];
  
  Object.keys(args).forEach(paramName => {
    // Parameters that represent input variables
    if ((paramName.endsWith('_variable') && paramName.includes('input')) || 
        paramName === 'input_variable' || paramName === 'input_variables' ||
        paramName.endsWith('_input_variable') || paramName === 'sas_variable' ||
        paramName === 'sas_err_variable' || paramName === 'q_variable' ||
        paramName === 'feature_input_variable' || paramName === 'predictor_input_variable' ||
        paramName === 'grid_variable' || paramName.endsWith('_var')) {
      input_params.push(paramName);
    }
    // Parameters that represent output variables or prefixes  
    else if ((paramName.endsWith('_variable') && paramName.includes('output')) || 
             paramName === 'output_variable' || paramName === 'output_variables' ||
             paramName === 'output_prefix' || paramName.endsWith('_output_variable')) {
      output_params.push(paramName);
    }
  });
  
  return {
    input_params: input_params,
    output_params: output_params
  };
}

// Get input/output variables from a node
function getNodeVariables(node) {
  const inputs = [];
  const outputs = [];
  
  // Get input variables
  const inputElements = node.querySelectorAll('input[data-input]');
  inputElements.forEach(input => {
    const paramName = input.getAttribute('data-input');
    const value = input.value || '';
    inputs.push({ paramName, value, element: input });
  });
  
  // Get output variables (exclude output_prefix as it's not a connectable variable)
  const outputElements = node.querySelectorAll('input[data-output]');
  outputElements.forEach(output => {
    const paramName = output.getAttribute('data-output');
    const value = output.value || '';
    
    // Skip output_prefix as it's a configuration parameter, not a connectable variable
    if (paramName !== 'output_prefix') {
      outputs.push({ paramName, value, element: output });
    }
  });
  
  return { inputs, outputs };
}

// Handle new connections between nodes
function handleNewConnection(sourceNode, targetNode) {
  const sourceVars = getNodeVariables(sourceNode);
  const targetVars = getNodeVariables(targetNode);
  
  // If source has no outputs, we can't make any connections
  if (sourceVars.outputs.length === 0) {
    alert('The source node has no output variables to connect.');
    return;
  }
  
  // If target has no inputs, we can't make any connections
  if (targetVars.inputs.length === 0) {
    alert('The target node has no input variables to connect to.');
    return;
  }
  
  // Check if we need to show modal or can auto-connect
  const needsModal = sourceVars.outputs.length > 1 || targetVars.inputs.length > 1;
  
  if (needsModal) {
    showConnectionModal(sourceNode, targetNode, sourceVars, targetVars);
  } else if (sourceVars.outputs.length === 1 && targetVars.inputs.length === 1) {
    // Auto-connect single input to single output
    const sourceOutput = sourceVars.outputs[0];
    const targetInput = targetVars.inputs[0];
    makeConnection(sourceOutput, targetInput);
  }
}

// Show the connection modal
function showConnectionModal(sourceNode, targetNode, sourceVars, targetVars) {
  const modal = document.getElementById('connection-modal');
  const sourceOutputsDiv = document.getElementById('source-outputs');
  const targetInputsDiv = document.getElementById('target-inputs');
  const connectionsPreview = document.getElementById('connections-preview');
  const connectionsList = document.getElementById('connections-list');
  
  // Clear previous content
  sourceOutputsDiv.innerHTML = '';
  targetInputsDiv.innerHTML = '';
  connectionsList.innerHTML = '';
  connectionsPreview.style.display = 'none';
  
  // Track pending connections
  const pendingConnections = [];
  let selectedOutput = null;
  
  // Create source output buttons
  sourceVars.outputs.forEach(output => {
    const button = document.createElement('div');
    button.className = 'variable-button';
    button.style.cssText = 'padding: 8px 12px; margin: 4px 0; border: 2px solid #ddd; border-radius: 4px; cursor: pointer; background: white;';
    button.innerHTML = `<strong>${output.paramName}</strong><br><small style="color: #666;">${output.value || '(not set)'}</small>`;
    
    button.addEventListener('click', () => {
      // Clear previous selection
      sourceOutputsDiv.querySelectorAll('.variable-button').forEach(b => {
        b.style.borderColor = '#ddd';
        b.style.background = 'white';
      });
      
      // Select this output
      selectedOutput = output;
      button.style.borderColor = '#007acc';
      button.style.background = '#e8f4fd';
    });
    
    sourceOutputsDiv.appendChild(button);
  });
  
  // Create target input buttons
  targetVars.inputs.forEach(input => {
    const button = document.createElement('div');
    button.className = 'variable-button';
    button.style.cssText = 'padding: 8px 12px; margin: 4px 0; border: 2px solid #ddd; border-radius: 4px; cursor: pointer; background: white;';
    button.innerHTML = `<strong>${input.paramName}</strong><br><small style="color: #666;">${input.value || '(not set)'}</small>`;
    
    button.addEventListener('click', () => {
      if (!selectedOutput) {
        alert('Please select an output variable first');
        return;
      }
      
      // Add to pending connections
      const connection = {
        source: selectedOutput,
        target: input
      };
      
      // Check if connection already exists
      const exists = pendingConnections.some(c => 
        c.source.paramName === connection.source.paramName && 
        c.target.paramName === connection.target.paramName
      );
      
      if (!exists) {
        pendingConnections.push(connection);
        updateConnectionsPreview();
      }
      
      // Clear output selection
      selectedOutput = null;
      sourceOutputsDiv.querySelectorAll('.variable-button').forEach(b => {
        b.style.borderColor = '#ddd';
        b.style.background = 'white';
      });
    });
    
    targetInputsDiv.appendChild(button);
  });
  
  function updateConnectionsPreview() {
    if (pendingConnections.length > 0) {
      connectionsPreview.style.display = 'block';
      connectionsList.innerHTML = pendingConnections.map((conn, index) => 
        `<div style="padding: 4px 0; display: flex; justify-content: space-between; align-items: center;">
          <span>${conn.source.paramName} → ${conn.target.paramName}</span>
          <button onclick="removePendingConnection(${index})" style="background: #ff5c5c; color: white; border: none; border-radius: 3px; padding: 2px 6px; cursor: pointer;">×</button>
        </div>`
      ).join('');
    } else {
      connectionsPreview.style.display = 'none';
    }
  }
  
  // Store pending connections globally for removal function
  window.pendingConnections = pendingConnections;
  window.updateConnectionsPreview = updateConnectionsPreview;
  
  // Show modal
  modal.style.display = 'block';
  
  // Handle modal buttons
  document.getElementById('cancel-connection').onclick = () => {
    modal.style.display = 'none';
  };
  
  document.getElementById('apply-connections').onclick = () => {
    pendingConnections.forEach(conn => {
      makeConnection(conn.source, conn.target);
    });
    modal.style.display = 'none';
    // Trigger connection update
    debounce(updateConnections, 500)();
  };
}

// Remove a pending connection
function removePendingConnection(index) {
  window.pendingConnections.splice(index, 1);
  window.updateConnectionsPreview();
}

// Make a connection between an output and input variable
function makeConnection(sourceOutput, targetInput) {
  if (!sourceOutput.value) {
    alert('Source output variable is not set');
    return;
  }
  
  const isPlural = targetInput.paramName.endsWith('s');
  let newValue;
  
  if (isPlural) {
    // Append to existing array
    let currentValue = targetInput.value || '[]';
    try {
      const currentArray = JSON.parse(currentValue);
      if (Array.isArray(currentArray)) {
        if (!currentArray.includes(sourceOutput.value)) {
          currentArray.push(sourceOutput.value);
        }
        newValue = JSON.stringify(currentArray);
      } else {
        // Not an array, create new array
        newValue = JSON.stringify([sourceOutput.value]);
      }
    } catch (e) {
      // Invalid JSON, create new array
      newValue = JSON.stringify([sourceOutput.value]);
    }
  } else {
    // Replace existing value
    newValue = sourceOutput.value;
  }
  
  // Update the target input element
  targetInput.element.value = newValue;
  
  // Update the corresponding parameter tile
  updateNodeAndTileInputs(targetInput.element.closest('.node').id, targetInput.paramName, newValue);
}

// Debounce function to limit API calls
function debounce(func, wait) {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}

async function updateConnections() {
  const ops = buildOps();
  if (ops.length === 0) return;
  
  try {
    const opsParam = encodeURIComponent(JSON.stringify(ops));
    const res = await authenticatedFetch(`/analyze_pipeline?ops=${opsParam}`, {
      method: 'GET'
    });
    
    if (res.ok) {
      const data = await res.json();
      
      // Check for pipeline analysis errors
      if (data.status === 'error' && data.errors && data.errors.length > 0) {
        showPipelineErrors(data.errors, data.message);
        return; // Stop processing if there are errors
      }
      
      // Set flag to prevent connection dialog during programmatic updates
      isProgrammaticConnection = true;
      
      // Clear existing connections
      connections = [];
      
      // Recreate connections based on analysis
      const nodeElements = Array.from(document.querySelectorAll('.node'));
      
      data.connections.forEach(conn => {
        const sourceNode = nodeElements.find(node => {
          return parseInt(node.dataset.opIndex) === conn.source_index;
        });
        const targetNode = nodeElements.find(node => {
          return parseInt(node.dataset.opIndex) === conn.target_index;
        });
        
        if (sourceNode && targetNode) {
          createConnection(sourceNode, targetNode);
        }
      });
      
      // Draw all connections
      drawConnections();
      
      // Optimize layout after connections are established
      optimizeLayout(data.connections);
      
      // Reset flag after programmatic updates are complete
      isProgrammaticConnection = false;
    } else {
      const errorText = await res.text();
      throw new Error('Server error: ' + errorText);
    }
  } catch (error) {
    console.error('Error updating connections:', error);
    alert('Error updating connections: ' + error.message);
  }
}
document.getElementById('submit-overlay').onclick = async () => {
  // Before submitting, permanently remove any node pending deletion
  if (lastDeletedNode) {
    permanentlyRemoveNode(lastDeletedNode);
    lastDeletedNode = null;
    document.getElementById('undo-banner').style.display = 'none';
  }
  
  const ops = buildOps();
  if (ops.length === 0) {
    alert('No operations to submit');
    return;
  }
  
  try {
    // Show validation feedback to user
    const submitButton = document.getElementById('submit-overlay');
    const originalText = submitButton.textContent;
    submitButton.textContent = 'Validating Pipeline...';
    submitButton.disabled = true;
    
    try {
      // First, validate the pipeline by analyzing it
      console.log('Validating pipeline before submission...');
      const opsParam = encodeURIComponent(JSON.stringify(ops));
      const validationRes = await authenticatedFetch(`/analyze_pipeline?ops=${opsParam}`, {
        method: 'GET'
      });
      
      if (validationRes.ok) {
        const validationData = await validationRes.json();
        
        // Check for pipeline analysis errors
        if (validationData.status === 'error' && validationData.errors && validationData.errors.length > 0) {
          console.log('Pipeline validation failed:', validationData.errors);
          showPipelineErrors(validationData.errors, 'Pipeline validation failed. Please fix the following issues before submitting:');
          return; // Stop submission if there are validation errors
        }
        
        console.log('Pipeline validation passed, proceeding with submission...');
      } else {
        const errorText = await validationRes.text();
        throw new Error('Pipeline validation failed: ' + errorText);
      }
      
      // Update button text for submission phase
      submitButton.textContent = 'Submitting Pipeline...';
    } finally {
      // Always restore button state in case of early return or error
      if (submitButton.textContent !== 'Submitting Pipeline...') {
        submitButton.textContent = originalText;
        submitButton.disabled = false;
      }
    }
    
    // If validation passes, proceed with submission
    const res = await authenticatedFetch('/enqueue', {
      method: 'POST',
      body: JSON.stringify({
        task_name: 'initialize_pipeline', 
        pipeline: ops,
        name: 'PipelineBuilder Pipeline'
      })
    });
    
    if (res.ok) {
      alert('Pipeline submitted successfully');
    } else {
      const errorText = await res.text();
      throw new Error('Failed to submit pipeline: ' + errorText);
    }
  } catch (error) {
    console.error('Pipeline submission error:', error);
    alert('Pipeline submission failed: ' + error.message);
  } finally {
    // Always restore button state after submission attempt
    const submitButton = document.getElementById('submit-overlay');
    submitButton.textContent = 'Submit Pipeline';
    submitButton.disabled = false;
  }
};

document.getElementById('load-prefab').onclick = async () => {
  const prefabSelect = document.getElementById('prefab-select');
  const name = prefabSelect.value;
  if (!name) return;
  
  // Close modal
  prefabModal.style.display = 'none';
  
  const res = await authenticatedFetch(`/load_prefab?name=${encodeURIComponent(name)}`, {
    method: 'GET'
  });
  if (res.ok) {
    const data = await res.json();
    loadPipeline(data);
  }
};

document.getElementById('optimize-layout').onclick = async () => {
  // Before optimizing, permanently remove any node pending deletion
  if (lastDeletedNode) {
    permanentlyRemoveNode(lastDeletedNode);
    lastDeletedNode = null;
    document.getElementById('undo-banner').style.display = 'none';
  }
  const ops = buildOps();
  if (ops.length === 0) return;
  
  try {
    const opsParam = encodeURIComponent(JSON.stringify(ops));
    const res = await authenticatedFetch(`/analyze_pipeline?ops=${opsParam}`, {
      method: 'GET'
    });
    
    if (res.ok) {
      const data = await res.json();
      
      // Check for pipeline analysis errors
      if (data.status === 'error' && data.errors && data.errors.length > 0) {
        showPipelineErrors(data.errors, data.message);
        return; // Stop layout optimization if there are errors
      }
      
      optimizeLayout(data.connections);
    } else {
      const errorText = await res.text();
      throw new Error('Server error: ' + errorText);
    }
  } catch (error) {
    console.error('Error optimizing layout:', error);
    alert('Error optimizing layout: ' + error.message);
  }
};

document.getElementById('pipeline-info-btn').onclick = () => {
  showPipelineInfo();
};

function deleteNode(node) {
  // If there's a previously deleted node, remove it for good.
  if (lastDeletedNode) {
    permanentlyRemoveNode(lastDeletedNode);
  }
  
  const paramTile = document.querySelector(`.param-tile[data-node-id="${node.id}"]`);
  
  // Store node and tile for potential undo
  lastDeletedNode = { node, tile: paramTile };
  
  // Hide the node and mark as deleted
  node.classList.add('deleted');
  node.style.display = 'none';
  if (paramTile) {
      paramTile.classList.add('deleted');
      paramTile.style.display = 'none';
  }
  
  // Hide connections associated with the node
  const nodeConnections = connections.filter(conn => 
    conn.source === node || conn.target === node
  );
  
  // Store connections for restoration
  node.dataset.hiddenConnections = JSON.stringify(nodeConnections.map(conn => ({
    sourceId: conn.source.id,
    targetId: conn.target.id
  })));
  
  // Remove connections from array
  nodeConnections.forEach(conn => removeConnection(conn));
  
  // Redraw connections
  drawConnections();
  
  // Show undo banner
  document.getElementById('undo-banner').style.display = 'block';
}

function permanentlyRemoveNode({ node, tile }) {
    if (tile) tile.remove();
    
    // Remove from nodes array
    const nodeIndex = nodes.indexOf(node);
    if (nodeIndex > -1) {
      nodes.splice(nodeIndex, 1);
    }
    
    // Remove the element from the DOM
    node.remove();
}

function undoDelete() {
  if (!lastDeletedNode) return;
  
  const { node, tile } = lastDeletedNode;
  
  // Un-hide the node and tile
  node.classList.remove('deleted');
  node.style.display = '';
  if (tile) {
    tile.classList.remove('deleted');
    tile.style.display = '';
  }

  // Restore connections
  if (node.dataset.hiddenConnections) {
    try {
      const connectionData = JSON.parse(node.dataset.hiddenConnections);
      isProgrammaticConnection = true;
      connectionData.forEach(connData => {
        const sourceNode = document.getElementById(connData.sourceId);
        const targetNode = document.getElementById(connData.targetId);
        
        if (sourceNode && targetNode) {
          createConnection(sourceNode, targetNode);
        }
      });
      isProgrammaticConnection = false;
      
      // Redraw connections
      drawConnections();
      
      // Clear the stored connection data
      delete node.dataset.hiddenConnections;
    } catch (e) {
      console.warn('Could not restore connections:', e);
    }
  }
  
  // Clear undo state
  lastDeletedNode = null;
  document.getElementById('undo-banner').style.display = 'none';
}

function updateNodeAndTileInputs(nodeId, paramName, value) {
  // Update param tile
  const tileInput = document.querySelector(`#param-tiles input[data-node-id="${nodeId}"][data-param="${paramName}"]`);
  if (tileInput && tileInput.value !== value) {
    tileInput.value = value;
    // Update required field styling if this is a required field
    updateRequiredFieldStyling(tileInput);
  }
  
  // Update node input if applicable (check for input parameter)
  const nodeInput = document.querySelector(`#${nodeId} input[data-input="${paramName}"]`);
  if (nodeInput && nodeInput.value !== value) {
    nodeInput.value = value;
  }
  
  // Update node output if applicable (check for output parameter)
  const nodeOutput = document.querySelector(`#${nodeId} input[data-output="${paramName}"]`);
  if (nodeOutput && nodeOutput.value !== value) {
    nodeOutput.value = value;
  }
  
  // Special handling for 'name' parameter to update node title
  if (paramName === 'name') {
    updateNodeTitle(nodeId, value);
  }
}

function updateNodeTitle(nodeId, newName) {
  const node = document.getElementById(nodeId);
  if (node) {
    const titleElement = node.querySelector('.node-title');
    if (titleElement) {
      // Use the new name if provided, otherwise fall back to the class name
      const fqcn = node.dataset.fqcn;
      titleElement.textContent = newName || fqcn.split('.').pop();
    }
  }
}

// Connection handling is now done in the finishConnectionDrag function

// Initialize the application
async function initializeApp() {
  // Start operation loading without blocking initial pipeline render.
  const opsPromise = loadOps();

  const [prefabsResult, currentPipelineResult] = await Promise.allSettled([
    loadPrefabs(),
    loadCurrentPipeline()
  ]);

  if (prefabsResult.status === 'rejected') {
    console.error('Failed to load prefabs:', prefabsResult.reason);
  }
  if (currentPipelineResult.status === 'rejected') {
    console.error('Failed to load current pipeline:', currentPipelineResult.reason);
  }

  // Hide canvas loading as soon as core pipeline state is ready.
  hideCanvasLoading();

  const [opsResult] = await Promise.allSettled([opsPromise]);
  if (opsResult.status === 'rejected') {
    console.error('Pipeline operations did not finish loading:', opsResult.reason);
  }

  console.log('Application initialized successfully');
}

function hideCanvasLoading() {
  const canvasLoading = document.getElementById('canvas-loading');
  if (canvasLoading) {
    canvasLoading.style.display = 'none';
  }
}

// Initialize zoom controls
function initializeZoom() {
  applyZoom(currentZoom);
}

// Start the initialization
initializeApp().then(() => {
  // Initialize zoom after everything else is loaded
  initializeZoom();
}).catch(error => {
  console.error('Initialization failed:', error);
  // Still initialize zoom even if other things failed
  initializeZoom();
});

// Wire up undo button
document.getElementById('undo-delete').onclick = undoDelete;

// Wire up zoom control buttons
document.getElementById('zoom-in').onclick = zoomIn;
document.getElementById('zoom-out').onclick = zoomOut;
document.getElementById('fit-to-view').onclick = fitToView;
document.getElementById('reset-zoom').onclick = resetZoom;

// Docstring modal elements
const docModal = document.getElementById('doc-modal');
const docModalContent = document.getElementById('doc-modal-content');
const docModalTitle = document.getElementById('doc-modal-title');
const docModalClose = document.getElementById('doc-modal-close');

function showDocModal(docstring = '', title = 'Documentation') {
  docModalTitle.textContent = title;
  docModalContent.innerHTML = docstring || 'No documentation available.';
  docModal.style.display = 'block';
}

function showPipelineErrors(errors, message) {
  let errorHtml = `<div style="margin-bottom: 15px; font-weight: bold; color: #d32f2f;">${message}</div>`;
  
  errorHtml += '<div style="max-height: 400px; overflow-y: auto;">';
  errors.forEach((error, index) => {
    errorHtml += `
      <div style="margin-bottom: 15px; padding: 10px; border: 1px solid #f44336; border-radius: 4px; background-color: #ffebee;">
        <div style="font-weight: bold; margin-bottom: 5px;">
          Operation ${error.operation_index + 1}: ${error.operation_name}
        </div>
        <div style="margin-bottom: 5px; font-family: monospace; font-size: 12px; color: #666;">
          Class: ${error.operation_class}
        </div>
        <div style="color: #d32f2f; font-family: monospace; font-size: 12px; background-color: #fff; padding: 5px; border-radius: 2px;">
          ${error.error}
        </div>
      </div>
    `;
  });
  errorHtml += '</div>';
  
  showDocModal(errorHtml, 'Pipeline Analysis Errors');
}

function generatePipelineInfo() {
  const nodes = Array.from(document.querySelectorAll('.node:not(.deleted)'));
  
  if (nodes.length === 0) {
    return 'No pipeline operations found.';
  }
  
  // Sort nodes by their visual order (top to bottom)
  nodes.sort((a, b) => {
    const aTop = parseInt(a.style.top);
    const bTop = parseInt(b.style.top);
    return aTop - bTop;
  });
  
  let info = '';
  
  // Header
  info += `${'PipelineOp'.padEnd(40)} ${'input_variable'.padEnd(20)} ---> ${'output_variable'}\n`;
  info += `${'-'.repeat(10).padEnd(40)} ${'-'.repeat(20).padEnd(20)} ${'-'.repeat(35)}\n`;
  
  // Operations list
  nodes.forEach((node, index) => {
    const fqcn = node.dataset.fqcn || 'Unknown';
    const opName = fqcn.split('.').pop();
    
    // Get input and output variables from the node
    const inputElements = node.querySelectorAll('input[data-input]');
    const outputElements = node.querySelectorAll('input[data-output]');
    
    let inputVars = [];
    let outputVars = [];
    
    inputElements.forEach(input => {
      if (input.value.trim()) {
        inputVars.push(input.value.trim());
      }
    });
    
    outputElements.forEach(output => {
      if (output.value.trim()) {
        outputVars.push(output.value.trim());
      }
    });
    
    const inputStr = inputVars.length > 0 ? inputVars.join(', ') : 'None';
    const outputStr = outputVars.length > 0 ? outputVars.join(', ') : 'None';
    
    info += `${(index + ')').padStart(3)} ${'<' + opName + '>'.padEnd(35)} ${inputStr.padEnd(20)} ---> ${outputStr}\n`;
  });
  
  // Input Variables section
  const allInputVars = new Set();
  const allOutputVars = new Set();
  
  // Helper function to extract variables from a value (handles both strings and JSON arrays)
  function extractVariables(value) {
    const variables = [];
    if (!value || !value.trim()) return variables;
    
    const trimmed = value.trim();
    
    // Try to parse as JSON array first
    try {
      const parsed = JSON.parse(trimmed);
      if (Array.isArray(parsed)) {
        // If it's an array, add all string elements
        parsed.forEach(item => {
          if (typeof item === 'string' && item.trim()) {
            variables.push(item.trim());
          }
        });
        return variables;
      }
    } catch (e) {
      // Not valid JSON, continue with string processing
    }
    
    // If not a JSON array, treat as a single variable or comma-separated list
    if (trimmed.includes(',')) {
      // Handle comma-separated values
      trimmed.split(',').forEach(item => {
        const cleaned = item.trim();
        if (cleaned) {
          variables.push(cleaned);
        }
      });
    } else {
      // Single variable
      variables.push(trimmed);
    }
    
    return variables;
  }
  
  nodes.forEach(node => {
    const inputElements = node.querySelectorAll('input[data-input]');
    const outputElements = node.querySelectorAll('input[data-output]');
    
    inputElements.forEach(input => {
      const variables = extractVariables(input.value);
      variables.forEach(variable => allInputVars.add(variable));
    });
    
    outputElements.forEach(output => {
      const variables = extractVariables(output.value);
      variables.forEach(variable => allOutputVars.add(variable));
    });
  });
  
  // Find variables that are inputs but not outputs (pipeline inputs)
  const pipelineInputs = Array.from(allInputVars).filter(v => !allOutputVars.has(v));
  
  info += '\n';
  info += 'Input Variables\n';
  info += '---------------\n';
  if (pipelineInputs.length > 0) {
    pipelineInputs.forEach((inputVar, index) => {
      info += `${index}) ${inputVar}\n`;
    });
  } else {
    info += 'No external input variables found.\n';
  }
  
  // Output Variables section
  const pipelineOutputs = Array.from(allOutputVars).filter(v => !allInputVars.has(v));
  
  info += '\n';
  info += 'Output Variables\n';
  info += '----------------\n';
  if (pipelineOutputs.length > 0) {
    pipelineOutputs.forEach((outputVar, index) => {
      info += `${index}) ${outputVar}\n`;
    });
  } else {
    info += 'No external output variables found.\n';
  }
  
  // Pipeline statistics
  info += '\n';
  info += 'Pipeline Statistics\n';
  info += '-------------------\n';
  info += `Total Operations: ${nodes.length}\n`;
  info += `Total Connections: ${connections.length}\n`;
  info += `Input Variables: ${pipelineInputs.length}\n`;
  info += `Output Variables: ${pipelineOutputs.length}\n`;
  
  return info;
}

function showPipelineInfo() {
  const infoContent = generatePipelineInfo();
  document.getElementById('pipeline-info-content').textContent = infoContent;
  document.getElementById('pipeline-info-modal').style.display = 'block';
}

docModalClose.addEventListener('click', () => {
  docModal.style.display = 'none';
});

// Close modal when clicking outside content
docModal.addEventListener('click', (e) => {
  if (e.target === docModal) {
    docModal.style.display = 'none';
  }
});

// Pipeline info modal elements and event listeners
const pipelineInfoModal = document.getElementById('pipeline-info-modal');
const pipelineInfoClose = document.getElementById('pipeline-info-close');

pipelineInfoClose.addEventListener('click', () => {
  pipelineInfoModal.style.display = 'none';
});

// Close pipeline info modal when clicking outside content
pipelineInfoModal.addEventListener('click', (e) => {
  if (e.target === pipelineInfoModal) {
    pipelineInfoModal.style.display = 'none';
  }
});

// Dictionary editor event listeners
document.getElementById('dict-add-key').addEventListener('click', addTopLevelKey);
document.getElementById('dict-format-json').addEventListener('click', () => {
  if (dictEditorState.isRawView) {
    try {
      const formatted = JSON.stringify(JSON.parse(document.getElementById('dict-raw-editor').value), null, 2);
      document.getElementById('dict-raw-editor').value = formatted;
    } catch (e) {
      alert('Invalid JSON: ' + e.message);
    }
  }
});
document.getElementById('dict-validate').addEventListener('click', validateDictData);
document.getElementById('dict-view-toggle').addEventListener('click', toggleDictView);
document.getElementById('dict-editor-save').addEventListener('click', saveDictEditor);
document.getElementById('dict-editor-cancel').addEventListener('click', closeDictEditor);

// Close dict editor when clicking outside
document.getElementById('dict-editor-modal').addEventListener('click', (e) => {
  if (e.target.id === 'dict-editor-modal') {
    closeDictEditor();
  }
});

// List editor event listeners
document.getElementById('list-add-item').addEventListener('click', addListItem);
document.getElementById('list-format-json').addEventListener('click', () => {
  if (listEditorState.isRawView) {
    try {
      const formatted = JSON.stringify(JSON.parse(document.getElementById('list-raw-editor').value), null, 2);
      document.getElementById('list-raw-editor').value = formatted;
    } catch (e) {
      alert('Invalid JSON: ' + e.message);
    }
  }
});
document.getElementById('list-validate').addEventListener('click', validateListData);
document.getElementById('list-view-toggle').addEventListener('click', toggleListView);
document.getElementById('list-editor-save').addEventListener('click', saveListEditor);
document.getElementById('list-editor-cancel').addEventListener('click', closeListEditor);

// Close list editor when clicking outside
document.getElementById('list-editor-modal').addEventListener('click', (e) => {
  if (e.target.id === 'list-editor-modal') {
    closeListEditor();
  }
});

// Group drag support
const groupDragData = {
  active: false,
  originals: new Map(), // node -> {left, top}
  draggingNode: null
};

// ---------------------------------------------------------------------
// Save current pipeline as prefab
// ---------------------------------------------------------------------
document.getElementById('save-prefab-btn').onclick = async () => {
  const ops = buildOps();
  if (ops.length === 0) {
    alert('No operations to save as a prefab');
    return;
  }

  // Ask user for the prefab name
  const prefabName = prompt('Enter a name for the prefab:', 'my_prefab');
  if (!prefabName || !prefabName.trim()) {
    return; // User cancelled or empty
  }

  try {
    const opsParam = encodeURIComponent(JSON.stringify(ops));
    const url = `/save_prefab?name=${encodeURIComponent(prefabName.trim())}&pipeline=${opsParam}`;
    const res = await authenticatedFetch(url, { method: 'GET' });
    const data = await res.json();

    if (data.status === 'success') {
      alert(`Prefab saved successfully as '${prefabName}'.`);
      // Refresh the prefab dropdown so the new prefab appears
      prefabSelect.innerHTML = "<option value=''>-- Select a prefab --</option>";
      await loadPrefabs();
    } else {
      alert('Failed to save prefab: ' + (data.message || 'Unknown error'));
    }
  } catch (error) {
    console.error('Error saving prefab:', error);
    alert('Error saving prefab: ' + error.message);
  }
};
