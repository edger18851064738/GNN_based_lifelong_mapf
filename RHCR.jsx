import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, RotateCcw, Settings, Info, ChevronDown, ChevronUp } from 'lucide-react';

const RHCRMAPFDemo = () => {
  const [gridSize] = useState({ width: 25, height: 20 });
  const [agents, setAgents] = useState([]);
  const [obstacles, setObstacles] = useState(new Set());
  const [isRunning, setIsRunning] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [paths, setPaths] = useState({});
  const [conflicts, setConflicts] = useState([]);
  const [showSettings, setShowSettings] = useState(false);
  const [algorithmSteps, setAlgorithmSteps] = useState([]);
  const [expandedInfo, setExpandedInfo] = useState(false);
  
  // RHCR Parameters
  const [rhcrParams, setRhcrParams] = useState({
    simulationWindow: 5,    // h: replanning period
    planningWindow: 10,     // w: planning window
    suboptimalBound: 1.5,   // ECBS suboptimality bound
    maxAgents: 8
  });

  const animationRef = useRef(null);

  // Initialize demo
  useEffect(() => {
    initializeDemo();
  }, []);

  const initializeDemo = () => {
    const newAgents = [];
    const numAgents = Math.min(rhcrParams.maxAgents, 8);
    
    // Generate agents with start and goal positions
    for (let i = 0; i < numAgents; i++) {
      const start = getRandomFreePosition();
      let goal = getRandomFreePosition();
      while (goal.x === start.x && goal.y === start.y) {
        goal = getRandomFreePosition();
      }
      
      newAgents.push({
        id: i,
        start,
        goal,
        currentPos: {...start},
        color: getAgentColor(i),
        tasks: [goal], // Lifelong MAPF: continuous goal assignment
        currentTask: 0,
        path: []
      });
    }
    
    setAgents(newAgents);
    generateObstacles();
    setCurrentTime(0);
    setPaths({});
    setConflicts([]);
    setAlgorithmSteps([]);
  };

  const getRandomFreePosition = () => {
    let pos;
    do {
      pos = {
        x: Math.floor(Math.random() * gridSize.width),
        y: Math.floor(Math.random() * gridSize.height)
      };
    } while (obstacles.has(`${pos.x},${pos.y}`));
    return pos;
  };

  const generateObstacles = () => {
    const newObstacles = new Set();
    const numObstacles = Math.floor(gridSize.width * gridSize.height * 0.15);
    
    for (let i = 0; i < numObstacles; i++) {
      const x = Math.floor(Math.random() * gridSize.width);
      const y = Math.floor(Math.random() * gridSize.height);
      newObstacles.add(`${x},${y}`);
    }
    
    setObstacles(newObstacles);
  };

  const getAgentColor = (id) => {
    const colors = [
      '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', 
      '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F'
    ];
    return colors[id % colors.length];
  };

  // A* Algorithm Implementation
  const aStar = (start, goal, timeConstraints = new Set()) => {
    const openSet = [{
      pos: start,
      g: 0,
      h: heuristic(start, goal),
      f: heuristic(start, goal),
      parent: null,
      time: 0
    }];
    
    const closedSet = new Set();
    const gScore = {};
    gScore[`${start.x},${start.y},0`] = 0;

    while (openSet.length > 0) {
      openSet.sort((a, b) => a.f - b.f);
      const current = openSet.shift();
      
      if (current.pos.x === goal.x && current.pos.y === goal.y) {
        return reconstructPath(current);
      }
      
      const key = `${current.pos.x},${current.pos.y},${current.time}`;
      if (closedSet.has(key)) continue;
      closedSet.add(key);
      
      const neighbors = getNeighbors(current.pos);
      neighbors.push(current.pos); // Wait action
      
      for (const neighbor of neighbors) {
        const newTime = current.time + 1;
        const timeKey = `${neighbor.x},${neighbor.y},${newTime}`;
        
        if (timeConstraints.has(timeKey)) continue;
        if (obstacles.has(`${neighbor.x},${neighbor.y}`)) continue;
        
        const tentativeG = current.g + 1;
        const scoreKey = `${neighbor.x},${neighbor.y},${newTime}`;
        
        if (gScore[scoreKey] === undefined || tentativeG < gScore[scoreKey]) {
          gScore[scoreKey] = tentativeG;
          const h = heuristic(neighbor, goal);
          
          openSet.push({
            pos: neighbor,
            g: tentativeG,
            h: h,
            f: tentativeG + h,
            parent: current,
            time: newTime
          });
        }
      }
    }
    
    return [];
  };

  const heuristic = (pos1, pos2) => {
    return Math.abs(pos1.x - pos2.x) + Math.abs(pos1.y - pos2.y);
  };

  const getNeighbors = (pos) => {
    const neighbors = [];
    const directions = [{x: 0, y: 1}, {x: 1, y: 0}, {x: 0, y: -1}, {x: -1, y: 0}];
    
    for (const dir of directions) {
      const newX = pos.x + dir.x;
      const newY = pos.y + dir.y;
      
      if (newX >= 0 && newX < gridSize.width && newY >= 0 && newY < gridSize.height) {
        neighbors.push({x: newX, y: newY});
      }
    }
    
    return neighbors;
  };

  const reconstructPath = (node) => {
    const path = [];
    let current = node;
    
    while (current) {
      path.unshift({...current.pos, time: current.time});
      current = current.parent;
    }
    
    return path;
  };

  // Enhanced CBS (ECBS) Implementation - Simplified
  const ecbs = (agentSubset, windowStart, windowEnd) => {
    const ct = [{
      constraints: {},
      cost: 0,
      paths: {}
    }];

    let bestNode = null;
    let focal = [];
    
    while (ct.length > 0 || focal.length > 0) {
      let node;
      
      if (focal.length > 0) {
        // Use focal search for suboptimal solutions
        focal.sort((a, b) => getConflictCount(a) - getConflictCount(b));
        node = focal.shift();
      } else {
        ct.sort((a, b) => a.cost - b.cost);
        node = ct.shift();
        
        if (!bestNode || node.cost <= bestNode.cost * rhcrParams.suboptimalBound) {
          focal.push(node);
          continue;
        }
      }

      // Plan paths for all agents with current constraints
      let allPathsValid = true;
      const newPaths = {};
      
      for (const agent of agentSubset) {
        const constraints = node.constraints[agent.id] || new Set();
        const path = aStar(agent.currentPos, agent.tasks[agent.currentTask], constraints);
        
        if (path.length === 0) {
          allPathsValid = false;
          break;
        }
        
        newPaths[agent.id] = path.filter(p => p.time >= windowStart && p.time < windowEnd);
      }
      
      if (!allPathsValid) continue;
      
      node.paths = newPaths;
      const conflicts = findConflicts(newPaths);
      
      if (conflicts.length === 0) {
        return node.paths;
      }
      
      // Create child nodes with new constraints
      const conflict = conflicts[0];
      for (const agentId of [conflict.agent1, conflict.agent2]) {
        const childNode = {
          constraints: {...node.constraints},
          cost: node.cost + 1,
          paths: {}
        };
        
        if (!childNode.constraints[agentId]) {
          childNode.constraints[agentId] = new Set();
        }
        
        const constraintKey = `${conflict.pos.x},${conflict.pos.y},${conflict.time}`;
        childNode.constraints[agentId].add(constraintKey);
        
        if (childNode.cost <= (bestNode?.cost || Infinity) * rhcrParams.suboptimalBound) {
          focal.push(childNode);
        } else {
          ct.push(childNode);
        }
      }
    }
    
    return {};
  };

  const findConflicts = (paths) => {
    const conflicts = [];
    const agentIds = Object.keys(paths);
    
    for (let i = 0; i < agentIds.length; i++) {
      for (let j = i + 1; j < agentIds.length; j++) {
        const path1 = paths[agentIds[i]];
        const path2 = paths[agentIds[j]];
        
        const maxTime = Math.min(path1.length, path2.length);
        for (let t = 0; t < maxTime; t++) {
          if (path1[t] && path2[t] && 
              path1[t].x === path2[t].x && path1[t].y === path2[t].y) {
            conflicts.push({
              agent1: agentIds[i],
              agent2: agentIds[j],
              pos: {x: path1[t].x, y: path1[t].y},
              time: t
            });
          }
        }
      }
    }
    
    return conflicts;
  };

  const getConflictCount = (node) => {
    return findConflicts(node.paths).length;
  };

  // RHCR Main Algorithm
  const runRHCR = () => {
    const newPaths = {};
    const newConflicts = [];
    const steps = [];
    
    const windowStart = currentTime;
    const windowEnd = currentTime + rhcrParams.planningWindow;
    
    steps.push(`RHCR Step: Planning window [${windowStart}, ${windowEnd})`);
    
    // Windowed MAPF solving using ECBS
    const windowPaths = ecbs(agents, windowStart, windowEnd);
    
    for (const agent of agents) {
      if (windowPaths[agent.id]) {
        newPaths[agent.id] = windowPaths[agent.id];
        steps.push(`Agent ${agent.id}: Path planned with ${windowPaths[agent.id].length} steps`);
      } else {
        steps.push(`Agent ${agent.id}: No valid path found`);
      }
    }
    
    const allConflicts = findConflicts(windowPaths);
    newConflicts.push(...allConflicts);
    
    if (allConflicts.length > 0) {
      steps.push(`Found ${allConflicts.length} conflicts in current window`);
    } else {
      steps.push('No conflicts detected in current planning window');
    }
    
    setPaths(newPaths);
    setConflicts(newConflicts);
    setAlgorithmSteps(steps);
  };

  // Animation and simulation
  const animate = () => {
    if (!isRunning) return;
    
    // Move agents along their paths
    setAgents(prevAgents => {
      return prevAgents.map(agent => {
        const agentPath = paths[agent.id];
        if (agentPath && currentTime < agentPath.length) {
          const nextPos = agentPath[currentTime];
          return {
            ...agent,
            currentPos: nextPos ? {x: nextPos.x, y: nextPos.y} : agent.currentPos
          };
        }
        return agent;
      });
    });
    
    // Replan every simulation window
    if (currentTime % rhcrParams.simulationWindow === 0) {
      runRHCR();
    }
    
    setCurrentTime(prev => prev + 1);
    
    animationRef.current = setTimeout(animate, 500);
  };

  const toggleSimulation = () => {
    if (isRunning) {
      clearTimeout(animationRef.current);
      setIsRunning(false);
    } else {
      setIsRunning(true);
      animate();
    }
  };

  const resetSimulation = () => {
    clearTimeout(animationRef.current);
    setIsRunning(false);
    initializeDemo();
  };

  const renderGrid = () => {
    const cells = [];
    
    for (let y = 0; y < gridSize.height; y++) {
      for (let x = 0; x < gridSize.width; x++) {
        const isObstacle = obstacles.has(`${x},${y}`);
        const agent = agents.find(a => a.currentPos.x === x && a.currentPos.y === y);
        const goal = agents.find(a => a.tasks[a.currentTask] && 
                                  a.tasks[a.currentTask].x === x && 
                                  a.tasks[a.currentTask].y === y);
        
        let cellContent = null;
        let cellStyle = 'w-6 h-6 border border-gray-300 flex items-center justify-center text-xs font-bold ';
        
        if (isObstacle) {
          cellStyle += 'bg-gray-800 text-white';
          cellContent = '▓';
        } else if (agent) {
          cellStyle += 'text-white font-bold';
          cellContent = agent.id;
        } else if (goal) {
          cellStyle += 'bg-gray-200 text-black';
          cellContent = `G${goal.id}`;
        } else {
          cellStyle += 'bg-white hover:bg-gray-100';
        }
        
        cells.push(
          <div
            key={`${x}-${y}`}
            className={cellStyle}
            style={{
              backgroundColor: agent ? agent.color : undefined,
              gridColumn: x + 1,
              gridRow: y + 1
            }}
          >
            {cellContent}
          </div>
        );
      }
    }
    
    return cells;
  };

  return (
    <div className="p-6 max-w-7xl mx-auto bg-white">
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          RHCR + A* + ECBS Lifelong Multi-Agent Path Finding Demo
        </h1>
        <div className="flex items-center gap-2 text-sm text-gray-600 mb-4">
          <Info className="w-4 h-4" />
          <span>Rolling Horizon Collision Resolution with Enhanced Conflict-Based Search</span>
          <button 
            onClick={() => setExpandedInfo(!expandedInfo)}
            className="flex items-center gap-1 text-blue-600 hover:text-blue-800"
          >
            {expandedInfo ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            {expandedInfo ? 'Less Info' : 'More Info'}
          </button>
        </div>
        
        {expandedInfo && (
          <div className="bg-blue-50 p-4 rounded-lg mb-4 text-sm">
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <h3 className="font-semibold text-blue-900 mb-2">Algorithm Overview</h3>
                <ul className="space-y-1 text-blue-800">
                  <li><strong>RHCR:</strong> Plans in rolling time windows for lifelong scenarios</li>
                  <li><strong>ECBS:</strong> Bounded-suboptimal conflict resolution</li>
                  <li><strong>A*:</strong> Single-agent pathfinding with space-time constraints</li>
                </ul>
              </div>
              <div>
                <h3 className="font-semibold text-blue-900 mb-2">Key Features</h3>
                <ul className="space-y-1 text-blue-800">
                  <li>• Continuous goal assignment (lifelong MAPF)</li>
                  <li>• Real-time replanning every simulation window</li>
                  <li>• Conflict detection and resolution</li>
                  <li>• Scalable to multiple agents</li>
                </ul>
              </div>
            </div>
          </div>
        )}

        <div className="flex flex-wrap gap-4 items-center mb-4">
          <button
            onClick={toggleSimulation}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            {isRunning ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            {isRunning ? 'Pause' : 'Start'}
          </button>
          
          <button
            onClick={resetSimulation}
            className="flex items-center gap-2 px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
          >
            <RotateCcw className="w-4 h-4" />
            Reset
          </button>
          
          <button
            onClick={() => setShowSettings(!showSettings)}
            className="flex items-center gap-2 px-4 py-2 bg-gray-300 text-gray-700 rounded-lg hover:bg-gray-400 transition-colors"
          >
            <Settings className="w-4 h-4" />
            Settings
          </button>
          
          <div className="flex items-center gap-4 text-sm">
            <span>Time: {currentTime}</span>
            <span>Agents: {agents.length}</span>
            <span>Conflicts: {conflicts.length}</span>
          </div>
        </div>

        {showSettings && (
          <div className="bg-gray-50 p-4 rounded-lg mb-4">
            <h3 className="font-semibold mb-3">RHCR Parameters</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <label className="block text-sm font-medium mb-1">Simulation Window (h)</label>
                <input
                  type="number"
                  min="1"
                  max="10"
                  value={rhcrParams.simulationWindow}
                  onChange={(e) => setRhcrParams(prev => ({
                    ...prev,
                    simulationWindow: parseInt(e.target.value)
                  }))}
                  className="w-full px-2 py-1 border rounded"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Planning Window (w)</label>
                <input
                  type="number"
                  min="5"
                  max="20"
                  value={rhcrParams.planningWindow}
                  onChange={(e) => setRhcrParams(prev => ({
                    ...prev,
                    planningWindow: parseInt(e.target.value)
                  }))}
                  className="w-full px-2 py-1 border rounded"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">ECBS Bound</label>
                <input
                  type="number"
                  min="1.0"
                  max="3.0"
                  step="0.1"
                  value={rhcrParams.suboptimalBound}
                  onChange={(e) => setRhcrParams(prev => ({
                    ...prev,
                    suboptimalBound: parseFloat(e.target.value)
                  }))}
                  className="w-full px-2 py-1 border rounded"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Max Agents</label>
                <input
                  type="number"
                  min="2"
                  max="12"
                  value={rhcrParams.maxAgents}
                  onChange={(e) => setRhcrParams(prev => ({
                    ...prev,
                    maxAgents: parseInt(e.target.value)
                  }))}
                  className="w-full px-2 py-1 border rounded"
                />
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="grid lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <div className="border-2 border-gray-300 rounded-lg p-4 bg-gray-50">
            <h3 className="font-semibold mb-3">Environment Grid</h3>
            <div 
              className="grid gap-0 mx-auto"
              style={{
                gridTemplateColumns: `repeat(${gridSize.width}, 1fr)`,
                gridTemplateRows: `repeat(${gridSize.height}, 1fr)`,
                maxWidth: '600px'
              }}
            >
              {renderGrid()}
            </div>
            <div className="mt-4 text-xs text-gray-600">
              <div className="flex flex-wrap gap-4">
                <span><span className="w-3 h-3 bg-gray-800 inline-block mr-1"></span>Obstacles</span>
                <span><span className="w-3 h-3 bg-gray-200 inline-block mr-1"></span>Goals (Gn)</span>
                <span>Colored circles: Agents (numbered)</span>
              </div>
            </div>
          </div>
        </div>

        <div className="space-y-4">
          <div className="border rounded-lg p-4">
            <h3 className="font-semibold mb-3">Algorithm Status</h3>
            <div className="space-y-2 text-sm">
              <div>
                <span className="font-medium">Current Phase:</span>
                <span className="ml-2 text-blue-600">
                  {currentTime % rhcrParams.simulationWindow === 0 ? 'Planning' : 'Executing'}
                </span>
              </div>
              <div>
                <span className="font-medium">Next Replan:</span>
                <span className="ml-2">
                  {rhcrParams.simulationWindow - (currentTime % rhcrParams.simulationWindow)} steps
                </span>
              </div>
              <div>
                <span className="font-medium">Window:</span>
                <span className="ml-2">
                  [{Math.floor(currentTime / rhcrParams.simulationWindow) * rhcrParams.simulationWindow}, 
                   {Math.floor(currentTime / rhcrParams.simulationWindow) * rhcrParams.simulationWindow + rhcrParams.planningWindow}]
                </span>
              </div>
            </div>
          </div>

          <div className="border rounded-lg p-4">
            <h3 className="font-semibold mb-3">Algorithm Steps</h3>
            <div className="max-h-40 overflow-y-auto">
              {algorithmSteps.length > 0 ? (
                algorithmSteps.map((step, idx) => (
                  <div key={idx} className="text-xs text-gray-700 mb-1 p-1 bg-gray-50 rounded">
                    {step}
                  </div>
                ))
              ) : (
                <div className="text-sm text-gray-500">Click Start to see algorithm steps</div>
              )}
            </div>
          </div>

          <div className="border rounded-lg p-4">
            <h3 className="font-semibold mb-3">Agent Details</h3>
            <div className="max-h-40 overflow-y-auto space-y-1">
              {agents.map(agent => (
                <div key={agent.id} className="text-xs p-2 rounded" style={{backgroundColor: agent.color + '20'}}>
                  <div className="font-medium">Agent {agent.id}</div>
                  <div>Pos: ({agent.currentPos.x}, {agent.currentPos.y})</div>
                  <div>Goal: ({agent.tasks[agent.currentTask]?.x}, {agent.tasks[agent.currentTask]?.y})</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RHCRMAPFDemo;