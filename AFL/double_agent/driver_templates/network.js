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
       
       // Update button states
       updateButtonStates();
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
       
       // Update button states
       updateButtonStates();
     }
     
     // Update button states based on selection
     function updateButtonStates() {
       const copyPasteButtons = document.getElementById('copy-paste-buttons');
       
       // Show/hide copy-paste buttons based on selection
       if (selectedNodes.size > 0) {
         copyPasteButtons.classList.add('show');
       } else {
         copyPasteButtons.classList.remove('show');
       }
       
       // Update button enabled/disabled states
       copyBtn.disabled = selectedNodes.size === 0;
       pasteBtn.disabled = clipboard.length === 0;
     }
     
     // Button event handlers
     copyBtn.addEventListener('click', () => {
       copySelectedNodes();
       updateButtonStates();
     });
     
     pasteBtn.addEventListener('click', () => {
       pasteNodes();
       updateButtonStates();
     });
     
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
        const ops = await res.json();
        
        // Clear loading indicator
        opList.innerHTML = '';
        
        ops.forEach(op => {
          const div = document.createElement('div');
          div.className = 'op-template';
          div.textContent = op.name;
          div.draggable = true;
          div.dataset.fqcn = op.fqcn;
          div.dataset.params = JSON.stringify(op.parameters);

          // Store full metadata including docstring
          const metadataObj = {
            input_params: op.input_params || [],
            output_params: op.output_params || [],
            docstring: op.docstring || ''
          };
          div.dataset.metadata = JSON.stringify(metadataObj);

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
      } catch (error) {
        opList.innerHTML = '<div class="loading-text" style="color: #e74c3c;">Failed to load pipeline operations</div>';
        console.error('Error loading pipeline ops:', error);
      }
      // Reset pan position
      currentPanX = 0;
      currentPanY = 0;
      // Clear any pending undos
      lastDeletedNode = null;
      document.getElementById('undo-banner').style.display = 'none';
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
      instance.deleteEveryConnection();
      instance.deleteEveryEndpoint();
      canvas.innerHTML = '';
      paramTiles.innerHTML = '';
      counter = 0;
      selectedNode = null;
      
      // Reset pan position
      currentPanX = 0;
      currentPanY = 0;
    }

    function loadPipeline(data) {
      isLoadingPipeline = true; // Disable connection dialog during loading
      clearCanvas();
      
      // Handle both old format (array) and new format (object with ops and connections)
      const ops = Array.isArray(data) ? data : data.ops;
      const connections = data.connections || [];
      
      let x = 50;
      let y = 50;
      const nodes = [];
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
        nodes.push({node: node, op: op, index: index});
        y += 150;
      });

            // Connect nodes using the backend-provided connection information
      isProgrammaticConnection = true;
      connections.forEach(conn => {
        const sourceNode = nodes.find(n => n.index === conn.source_index);
        const targetNode = nodes.find(n => n.index === conn.target_index);
        
        if (sourceNode && targetNode) {
          instance.connect({
            source: sourceNode.node.querySelector('[data-role="out"]'),
            target: targetNode.node.querySelector('[data-role="in"]'),
            connector: ['Bezier', { curviness: 30 }],
            anchors: ['Center', 'Center']
          });
        }
      });
      isProgrammaticConnection = false;
      
      // Optimize layout after loading pipeline
      if (connections.length > 0) {
        optimizeLayout(connections);
      }
      
      isLoadingPipeline = false; // Re-enable connection dialog
      updateEdgeGlow();
    }

    async function loadCurrentPipeline() {
      const res = await authenticatedFetch('/current_pipeline', {
        method: 'GET'
      });
      if (res.ok) {
        const data = await res.json();
        // Handle both legacy (array) and new (object with ops/connections) formats
        if ((Array.isArray(data) && data.length) || (data && Array.isArray(data.ops) && data.ops.length)) {
          loadPipeline(data);
        }
      }
    }

    function selectNode(node) {
      // Remove selection from all nodes and tiles
      document.querySelectorAll('.node.selected').forEach(n => n.classList.remove('selected'));
      document.querySelectorAll('.param-tile.selected').forEach(t => t.classList.remove('selected'));
      
      // Select the new node
      selectedNode = node;
      node.classList.add('selected');
      
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
      function createParamGroup(key, val) {
        const group = document.createElement('div');
        group.className = 'param-group';
        
        const label = document.createElement('label');
        label.textContent = key;
        
        const input = document.createElement('input');
        input.type = 'text';
        input.dataset.param = key;
        input.dataset.nodeId = node.id;
        
        // Mark input/output parameters for special handling
        if (opMetadata && opMetadata.input_params && opMetadata.input_params.includes(key)) {
          input.dataset.isInputParam = 'true';
        }
        if (opMetadata && opMetadata.output_params && opMetadata.output_params.includes(key)) {
          input.dataset.isOutputParam = 'true';
        }
        
        if (val !== null && val !== undefined) {
          if (typeof val === 'object') {
            input.value = JSON.stringify(val);
          } else {
            input.value = val;
          }
        }
        
        // Add change listener to update connections when parameters change
        input.addEventListener('change', (e) => {
          updateNodeAndTileInputs(node.id, key, e.target.value);
          
          // Special handling for 'name' parameter to update node title
          if (key === 'name') {
            updateNodeTitle(node.id, e.target.value);
          }
          
          debounce(updateConnections, 500)();
        });
        
        group.appendChild(label);
        group.appendChild(input);
        return group;
      }

      // Always create 'name' field first
      const nameValue = params.name || '';
      const nameGroup = createParamGroup('name', nameValue);
      tile.appendChild(nameGroup);

      // Create all other parameters (excluding 'name' to avoid duplication)
      Object.entries(params).forEach(([key, val]) => {
        if (key !== 'name') {
          const group = createParamGroup(key, val);
          tile.appendChild(group);
        }
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
          
          // Set flag to prevent connection dialog during programmatic updates
          isProgrammaticConnection = true;
          
          // Clear existing connections
          instance.deleteEveryConnection();
          
          // Recreate connections based on analysis
          const nodesByOutputVar = new Map();
          const nodes = Array.from(document.querySelectorAll('.node'));
          
          nodes.forEach(node => {
            // Check all output parameter inputs for this node
            const outputInputs = node.querySelectorAll('input[data-output]');
            outputInputs.forEach(outInput => {
              if (outInput && outInput.value) {
                nodesByOutputVar.set(outInput.value, node);
              }
            });
          });
          
          data.connections.forEach(conn => {
            const sourceNode = nodes.find(node => {
              return parseInt(node.dataset.opIndex) === conn.source_index;
            });
            const targetNode = nodes.find(node => {
              return parseInt(node.dataset.opIndex) === conn.target_index;
            });
            
            if (sourceNode && targetNode) {
              instance.connect({
                source: sourceNode.querySelector('[data-role="out"]'),
                target: targetNode.querySelector('[data-role="in"]'),
                connector: ['Bezier', { curviness: 30 }],
                anchors: ['Center', 'Center']
              });
            }
          });
          
          // Optimize layout after connections are established
          optimizeLayout(data.connections);
          
          // Reset flag after programmatic updates are complete
          isProgrammaticConnection = false;
        }
      } catch (error) {
        console.error('Error updating connections:', error);
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
          optimizeLayout(data.connections);
        }
      } catch (error) {
        console.error('Error optimizing layout:', error);
      }
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
      const allConnections = instance.getAllConnections();
      const nodeConnections = allConnections.filter(conn => 
        conn.source === node || conn.target === node
      );
      
      // Store connections for restoration
      node.dataset.hiddenConnections = JSON.stringify(nodeConnections.map(conn => ({
        sourceId: conn.source.id,
        targetId: conn.target.id,
        sourceEndpoint: conn.endpoints[0].anchor.type,
        targetEndpoint: conn.endpoints[1].anchor.type
      })));
      
      // Remove connections
      nodeConnections.forEach(conn => instance.deleteConnection(conn));
      
      // Show undo banner
      document.getElementById('undo-banner').style.display = 'block';
    }

    function permanentlyRemoveNode({ node, tile }) {
        if (tile) tile.remove();
        instance.remove(node); // This also removes the element from the DOM
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
              instance.connect({
                source: sourceNode.querySelector('[data-role="out"]'),
                target: targetNode.querySelector('[data-role="in"]'),
                connector: ['Bezier', { curviness: 30 }],
                anchors: ['Center', 'Center']
              });
            }
          });
          isProgrammaticConnection = false;
          
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

    // Handle connection events
    instance.bind('connection', function(info) {
      // Only show connection dialog for user-initiated connections
      if (!isLoadingPipeline && !isProgrammaticConnection) {
        // Get the actual node elements (the connection endpoints are inside the nodes)
        const sourceNode = info.source.closest('.node');
        const targetNode = info.target.closest('.node');
        

        
        if (sourceNode && targetNode) {
          handleNewConnection(sourceNode, targetNode);
        }
      }
    });

    loadOps();
    loadPrefabs();
    loadCurrentPipeline();

    // Wire up undo button
    document.getElementById('undo-delete').onclick = undoDelete;
    
    // Initialize button states
    updateButtonStates();

    // Docstring modal elements
    const docModal = document.getElementById('doc-modal');
    const docModalContent = document.getElementById('doc-modal-content');
    const docModalTitle = document.getElementById('doc-modal-title');
    const docModalClose = document.getElementById('doc-modal-close');

    function showDocModal(docstring = '', title = 'Documentation') {
      docModalTitle.textContent = title;
      docModalContent.textContent = docstring || 'No documentation available.';
      docModal.style.display = 'block';
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
