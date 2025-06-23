    const instance = jsPlumb.getInstance({Container: 'canvas'});
    const opList = document.getElementById('op-list');
    const canvas = document.getElementById('canvas');
    const prefabSelect = document.getElementById('prefab-select');
    const paramTiles = document.getElementById('param-tiles');
    let counter = 0;
    let selectedNode = null;
    let lastDeletedNode = null;
    let isLoadingPipeline = false;
    let isProgrammaticConnection = false;
    
    // Copy-paste functionality variables
    let selectedNodes = new Set();
    let clipboard = [];
    let lastMouseX = 0;
    let lastMouseY = 0;
    
    // UI elements
    const copyBtn = document.getElementById('copy-btn');
    const pasteBtn = document.getElementById('paste-btn');
    const prefabModal = document.getElementById('prefab-modal');
    const loadPrefabModalBtn = document.getElementById('load-prefab-btn');


    canvas.addEventListener('dragover', e => e.preventDefault());
    canvas.addEventListener('drop', e => {
      e.preventDefault();
      const fqcn = e.dataTransfer.getData('text/plain');
      const opTemplate = document.querySelector(`[data-fqcn="${fqcn}"]`);
      const params = JSON.parse(opTemplate.dataset.params);
      const metadata = JSON.parse(opTemplate.dataset.metadata);
      const node = addNode(fqcn, params, e.offsetX, e.offsetY, metadata);
      
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
      node.style.left = (x - currentPanX) + 'px';
      node.style.top = (y - currentPanY) + 'px';
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
      // Make node draggable with group support
      instance.draggable(node, {
        start: function(params) {
          if (selectedNodes.size > 1 && selectedNodes.has(node)) {
            // Capture original positions for each selected node
            groupDragData.active = true;
            groupDragData.draggingNode = node;
            groupDragData.originals.clear();
            selectedNodes.forEach(n => {
              groupDragData.originals.set(n, {
                left: parseInt(n.style.left) || 0,
                top: parseInt(n.style.top) || 0
              });
            });
          } else {
            groupDragData.active = false;
          }
        },
        drag: function(params) {
          if (groupDragData.active && groupDragData.draggingNode === node) {
            const orig = groupDragData.originals.get(node);
            if (!orig) return;
            const currentLeft = parseInt(node.style.left) || 0;
            const currentTop = parseInt(node.style.top) || 0;
            const deltaX = currentLeft - orig.left;
            const deltaY = currentTop - orig.top;
            groupDragData.originals.forEach((pos, n) => {
              if (n !== node) {
                n.style.left = (pos.left + deltaX) + 'px';
                n.style.top = (pos.top + deltaY) + 'px';
                instance.revalidate(n);
              }
            });
          }
        },
        stop: function(params) {
          groupDragData.active = false;
          groupDragData.draggingNode = null;
          updateEdgeGlow();
        }
      });
      instance.makeSource(outAnchor, {
        anchor: 'Center',
        connector: ['Bezier', { curviness: 30 }],
        maxConnections: -1
      });
      instance.makeTarget(inAnchor, {
        anchor: 'Center',
        connector: ['Bezier', { curviness: 30 }],
        maxConnections: -1
      });
      
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
        if ((trimmed.startsWith('{') && trimmed.endsWith('}')) || (trimmed.startsWith('[') && trimmed.endsWith(']'))) {
            try {
                // Attempt to parse as JSON
                return JSON.parse(trimmed);
            } catch (e) {
                // Fallback to string if not valid JSON
                return value;
            }
        }
        return value;
    }

    // Canvas panning functionality
    function animatedPanAllNodes(deltaX, deltaY, duration = 500) {
      const nodes = document.querySelectorAll('.node');
      const startTime = performance.now();
      
      // Temporarily enable transitions
      nodes.forEach(node => {
        node.style.transition = `left ${duration}ms ease-out, top ${duration}ms ease-out`;
      });
      
      // Apply the movement
      nodes.forEach(node => {
        const currentX = parseInt(node.style.left) || 0;
        const currentY = parseInt(node.style.top) || 0;
        node.style.left = (currentX + deltaX) + 'px';
        node.style.top = (currentY + deltaY) + 'px';
      });
      
      // Update current pan position
      currentPanX += deltaX;
      currentPanY += deltaY;
      
      // Continuously update connections during animation
      function updateConnections() {
        const elapsed = performance.now() - startTime;
        if (elapsed < duration) {
          // Revalidate all nodes and repaint connections
          nodes.forEach(node => {
            instance.revalidate(node);
          });
          instance.repaintEverything();
          
          // Continue animation
          requestAnimationFrame(updateConnections);
        } else {
          // Animation complete - final cleanup
          nodes.forEach(node => {
            instance.revalidate(node);
            node.style.transition = '';
          });
          instance.repaintEverything();
          updateEdgeGlow();
        }
      }
      
      // Start the connection update loop
      requestAnimationFrame(updateConnections);
    }
    
    function panAllNodes(deltaX, deltaY) {
      const nodes = document.querySelectorAll('.node');
      nodes.forEach(node => {
        const currentX = parseInt(node.style.left) || 0;
        const currentY = parseInt(node.style.top) || 0;
        node.style.left = (currentX + deltaX) + 'px';
        node.style.top = (currentY + deltaY) + 'px';
        
        // Notify jsPlumb that the node has moved
        instance.revalidate(node);
      });
      
      // Update current pan position
      currentPanX += deltaX;
      currentPanY += deltaY;
      
      // Repaint connections
      instance.repaintEverything();

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

    // Add vertical scrolling support - works like dragging but only vertically
    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      
      // Calculate scroll delta with reduced sensitivity
      const deltaY = -e.deltaY * 0.5;
      
      // Only move nodes vertically, similar to dragging behavior
      const nodes = document.querySelectorAll('.node');
      nodes.forEach(node => {
        const currentY = parseInt(node.style.top) || 0;
        const newY = currentY + deltaY;
        
        // Prevent nodes from going to invalid positions
        if (newY > -1000 && newY < 10000) {
          node.style.top = newY + 'px';
        }
      });
      
      // Update current pan position
      currentPanY += deltaY;
      
      // Revalidate connections without full repaint to avoid SVG errors
      nodes.forEach(node => {
        instance.revalidate(node);
      });
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
            updateButtonStates();
            break;
          case 'v':
          case 'V':
            e.preventDefault();
            pasteNodes();
            updateButtonStates();
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
            updateButtonStates();
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
         
         // Check if node extends beyond visible canvas area
         // Nodes are positioned absolutely, so we check against 0,0 to canvas width/height
         if (nodeTop < 25) showTop = true;
         if (nodeTop + nodeHeight > canvasRect.height - 25) showBottom = true;
         if (nodeLeft < 25) showLeft = true;
         if (nodeLeft + nodeWidth > canvasRect.width - 25) showRight = true;
       });

       glowTop.style.opacity = showTop ? 1 : 0;
       glowBottom.style.opacity = showBottom ? 1 : 0;
       glowLeft.style.opacity = showLeft ? 1 : 0;
       glowRight.style.opacity = showRight ? 1 : 0;
      
      // Debug logging
      if (showTop || showBottom || showLeft || showRight) {
        console.log('Edge glow:', {showTop, showBottom, showLeft, showRight, canvasRect});
      }
     }

    // Periodic check (fallback) in case some events missed
    setInterval(updateEdgeGlow, 1000);

