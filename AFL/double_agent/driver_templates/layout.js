    function optimizeLayout(connections) {
      const nodes = Array.from(document.querySelectorAll('.node'));
      if (nodes.length <= 1) return;
      
      console.log('Applying Sugiyama-style hierarchy layout');
      
      const graph = buildDirectedGraph(nodes, connections);
      let layers = assignLayers(graph);
      
      // Add dummy nodes for edges spanning multiple layers
      const { graph: augmentedGraph, layers: augmentedLayers } = addDummyNodes(graph, layers);
      
      const crossingReducedLayers = reduceCrossings(augmentedLayers, augmentedGraph);
      positionNodes(crossingReducedLayers, augmentedGraph);
      
      // Repaint all connections
      instance.repaintEverything();
      updateEdgeGlow();
    }
    
    function addDummyNodes(graph, layers) {
      let dummyNodeCounter = 0;
      const newNodes = new Map(graph.nodes);
      const newEdges = new Map(JSON.parse(JSON.stringify(Array.from(graph.edges))));
      const newLayers = layers.map(layer => [...layer]);

      for (let i = 0; i < layers.length; i++) {
        for (const u of layers[i]) {
          const uNode = newNodes.get(u);
          if (!uNode) continue;

          // Make a copy of outEdges to iterate over, as we might modify the original
          const outEdges = [...uNode.outEdges];
          for (const v of outEdges) {
            const vNode = newNodes.get(v);
            if (!vNode) continue;
            
            const uLayer = uNode.layer;
            const vLayer = vNode.layer;
            
            if (vLayer > uLayer + 1) {
              // Remove original edge
              uNode.outEdges = uNode.outEdges.filter(id => id !== v);
              vNode.inEdges = vNode.inEdges.filter(id => id !== u);
              const uEdges = newEdges.get(u) || [];
              newEdges.set(u, uEdges.filter(id => id !== v));

              let parent = u;
              for (let j = uLayer + 1; j < vLayer; j++) {
                const dummyId = `dummy_${dummyNodeCounter++}`;
                const dummyNode = {
                  id: dummyId,
                  isDummy: true,
                  inEdges: [parent],
                  outEdges: [],
                  layer: j
                };

                // Add dummy node to graph and layers
                newNodes.set(dummyId, dummyNode);
                if (!newLayers[j]) newLayers[j] = [];
                newLayers[j].push(dummyId);
                
                // Rewire previous node to point to dummy
                newNodes.get(parent).outEdges.push(dummyId);

                // Update edges map
                const parentEdges = newEdges.get(parent) || [];
                if (!parentEdges.includes(dummyId)) {
                    parentEdges.push(dummyId);
                    newEdges.set(parent, parentEdges);
                }
                
                parent = dummyId;
              }
              
              // Connect last dummy node to original target
              newNodes.get(parent).outEdges.push(v);
              vNode.inEdges.push(parent);

              // Update edges map
              const parentEdges = newEdges.get(parent) || [];
              if (!parentEdges.includes(v)) {
                parentEdges.push(v);
                newEdges.set(parent, parentEdges);
              }
            }
          }
        }
      }
      
      const finalGraph = { nodes: newNodes, edges: newEdges };
      return { graph: finalGraph, layers: newLayers };
    }

    function buildDirectedGraph(nodes, connections) {
      const graph = {
        nodes: new Map(),
        edges: new Map()
      };
      
      // Initialize nodes
      nodes.forEach(node => {
        const index = parseInt(node.dataset.opIndex);
        graph.nodes.set(index, {
          id: index,
          element: node,
          inEdges: [],
          outEdges: [],
          layer: -1
        });
        graph.edges.set(index, []);
      });
      
      // Add edges
      connections.forEach(conn => {
        if (graph.nodes.has(conn.source_index) && graph.nodes.has(conn.target_index)) {
          const sourceNode = graph.nodes.get(conn.source_index);
          const targetNode = graph.nodes.get(conn.target_index);
          
          // Avoid duplicate edges
          if (!sourceNode.outEdges.includes(conn.target_index)) {
            sourceNode.outEdges.push(conn.target_index);
          }
          if (!targetNode.inEdges.includes(conn.source_index)) {
            targetNode.inEdges.push(conn.source_index);
          }
          
          const edgeList = graph.edges.get(conn.source_index) || [];
          if (!edgeList.includes(conn.target_index)) {
            edgeList.push(conn.target_index);
            graph.edges.set(conn.source_index, edgeList);
          }
        }
      });
      
      return graph;
    }
    
    function assignLayers(graph) {
      const layers = [];
      const visited = new Set();
      const visiting = new Set();
      
      // Assign layers using longest path from sources
      function assignLayer(nodeId) {
        if (visiting.has(nodeId)) return 0; // Cycle detection
        if (visited.has(nodeId)) return graph.nodes.get(nodeId).layer;
        
        visiting.add(nodeId);
        const node = graph.nodes.get(nodeId);
        
        let maxParentLayer = -1;
        node.inEdges.forEach(parentId => {
          maxParentLayer = Math.max(maxParentLayer, assignLayer(parentId));
        });
        
        node.layer = maxParentLayer + 1;
        visited.add(nodeId);
        visiting.delete(nodeId);
        
        // Add to layers array
        while (layers.length <= node.layer) {
          layers.push([]);
        }
        layers[node.layer].push(nodeId);
        
        return node.layer;
      }
      
      // Process all nodes
      graph.nodes.forEach((node, nodeId) => {
        if (!visited.has(nodeId)) {
          assignLayer(nodeId);
        }
      });
      
      return layers;
    }
    
    function reduceCrossings(layers, graph) {
      // Simple crossing reduction using barycenter method
      const maxIterations = 4;
      
      for (let i = 0; i < maxIterations; i++) {
        // Forward pass
        for (let l = 1; l < layers.length; l++) {
          layers[l].sort((a, b) => {
            const barycenterA = calculateBarycenter(a, graph.nodes.get(a).inEdges, layers[l-1], graph);
            const barycenterB = calculateBarycenter(b, graph.nodes.get(b).inEdges, layers[l-1], graph);
            return barycenterA - barycenterB;
          });
        }
        
        // Backward pass
        for (let l = layers.length - 2; l >= 0; l--) {
          layers[l].sort((a, b) => {
            const barycenterA = calculateBarycenter(a, graph.nodes.get(a).outEdges, layers[l+1], graph);
            const barycenterB = calculateBarycenter(b, graph.nodes.get(b).outEdges, layers[l+1], graph);
            return barycenterA - barycenterB;
          });
        }
      }
      
      return layers;
    }
    
    function calculateBarycenter(nodeId, connectedNodes, targetLayer, graph) {
      if (connectedNodes.length === 0) {
          // If a node has no connections in the current direction, 
          // return a default position
          return 0;
      }
      
      let sum = 0;
      let count = 0;
      
      connectedNodes.forEach(connectedId => {
        const index = targetLayer.indexOf(connectedId);
        if (index !== -1) {
          sum += index;
          count++;
        }
      });
      
      return count > 0 ? sum / count : 0;
    }
    
    function positionNodes(layers, graph) {
      const canvasRect = canvas.getBoundingClientRect();
      const layerHeight = 180;
      const nodeSpacing = 350; // Increased spacing for wider nodes
      const startY = 80;
      
      const positions = new Map();

      layers.forEach((layer, layerIndex) => {
        const y = startY + layerIndex * layerHeight;
        const layerWidth = layer.length * nodeSpacing;
        const startX = Math.max(50, (canvasRect.width - layerWidth) / 2);
        
        layer.forEach((nodeId, position) => {
          const x = startX + position * nodeSpacing;
          positions.set(nodeId, { x, y });
        });
      });
      
      graph.nodes.forEach((node, nodeId) => {
        if (!node.isDummy) {
          const element = node.element;
          const pos = positions.get(nodeId);
          if (element && pos) {
            element.style.left = (pos.x + currentPanX) + 'px';
            element.style.top = (pos.y + currentPanY) + 'px';
            instance.revalidate(element);
          }
        }
      });
    }

