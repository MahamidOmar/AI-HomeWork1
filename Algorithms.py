import numpy as np

from CampusEnv import CampusEnv
from typing import List, Tuple
import heapdict


class DFSGNode:
  def __init__(self,state=None,action=None,cost=None,parent=None,isHole=None):
    self.state = state
    self.action = action
    self.cost = cost
    self.parent = parent
    self.isHole = isHole

class DFSGAgent:

  def __init__(self):
    self.expanded = 0

  def path(self, node) -> Tuple[List[int], float, int]:
    sol = []
    total_cost = 0
    while node.parent is not None:
        sol.append(node.action)
        total_cost += node.cost
        node = node.parent
    return sol[::-1], float(total_cost), self.expanded

  def dfs_search_rec(self, env: CampusEnv, OPEN_NODES: List[DFSGNode], OPEN_STATES: set, CLOSE_NODES: List[DFSGNode], CLOSE_STATES: set) -> Tuple[List[int], float, int]:
        if not OPEN_NODES:
            return [], 0, 0
        node = OPEN_NODES.pop()
        node_state = node.state
        CLOSE_NODES.append(node)
        CLOSE_STATES.add(node_state)
        if env.is_final_state(node.state):
            return self.path(node)
        self.expanded += 1
        if node.isHole:
            return [], 0, 0
        for action, successor in env.succ(node.state).items():
            child = DFSGNode(successor[0], action, successor[1], node, successor[2])
            if child.state not in OPEN_STATES and child.state not in CLOSE_STATES:
                OPEN_NODES.append(child)
                OPEN_STATES.add(child.state)
                res = self.dfs_search_rec(env, OPEN_NODES, OPEN_STATES, CLOSE_NODES, CLOSE_STATES)
                if res != ([], 0, 0):
                    return res
        return [], 0, 0

  def search(self, env: CampusEnv) -> Tuple[List[int], float, int]:
        self.expanded = 0
        OPEN_NODES = [DFSGNode(env.get_initial_state(), None, 0, None, False)]
        OPEN_STATES = {env.get_initial_state()}
        CLOSE_NODES = []
        CLOSE_STATES = set()
        res = self.dfs_search_rec(env, OPEN_NODES, OPEN_STATES, CLOSE_NODES, CLOSE_STATES)
        if res == ([], 0, 0):
            res = [], 0, self.expanded
        return res
  
class UCSNode:
  def __init__(self,state=None,action=None,cost=None,parent=None,isHole=None, g=0):
    self.state = state
    self.action = action
    self.cost = cost
    self.parent = parent
    self.isHole = isHole
    self.g = g

class UCSAgent:

    def __init__(self):
      self.expanded = 0

    def path(self, node) -> Tuple[List[int], float, int]:
      sol, total_cost = [], 0
      while node.parent:
          total_cost += node.cost
          sol.append(node.action)
          node = node.parent
      return sol[::-1], float(total_cost), self.expanded
    
    def inOpen(self, open_set: heapdict.heapdict, state: int) -> bool:
        return any(node.state == state for node in open_set.keys())

    def getNodeByState(self, open_set, state):
      return next((node for node in open_set.keys() if node.state == state), None)

    def search(self, env: CampusEnv) -> Tuple[List[int], float ,int]:
      self.expanded = 0
      OPEN = heapdict.heapdict()
      CLOSE_NODES = []
      CLOSE_STATES = []
      node = UCSNode(env.get_initial_state(),None,0,None,False, 0)
      OPEN[node] = (node.g, node.state)
      while len(OPEN) > 0:
        node_tuple = OPEN.popitem()
        node_ucs = node_tuple[0]
        node_state = node_ucs.state
        CLOSE_NODES.append(node_ucs)
        CLOSE_STATES.append(node_state)
        if env.is_final_state(node_state) == True:
          return self.path(node_ucs)
        self.expanded+=1
        if node_ucs.isHole == True:
          continue
        for action, successor in env.succ(node_ucs.state).items():
          new_cost = node_ucs.g + successor[1]
          child = UCSNode(successor[0],action,successor[1],node_ucs,successor[2],new_cost)
          if (not (self.inOpen(OPEN,child.state) == True)) and (not (child.state in CLOSE_STATES)):
            OPEN[child]=(child.g,child.state)
          else:
            if (self.inOpen(OPEN,child.state) == True):
              nodeWithSameState = self.getNodeByState(OPEN,child.state)
              if nodeWithSameState.g > new_cost:
                del OPEN[nodeWithSameState]
                OPEN[child] = (child.g, child.state)
      return ([],0,self.expanded)

    def incorrect_search(self, env: 'CampusEnv') -> Tuple[List[int], float, int]:
      self.expanded = 0
      open_set = heapdict.heapdict()
      close_nodes = []
      close_states = set()
      
      initial_state = env.get_initial_state()
      root_node = UCSNode(state=initial_state, cost=0, g=0)
      open_set[root_node] = root_node.g

      while open_set:
          node_ucs, _ = open_set.popitem()
          if env.is_final_state(node_ucs.state):
              return self.path(node_ucs)
          
          close_nodes.append(node_ucs)
          close_states.add(node_ucs.state)
          self.expanded += 1

          if node_ucs.isHole:
              continue

          for action, (successor_state, cost, is_hole) in env.succ(node_ucs.state).items():
              new_cost = node_ucs.g + cost
              child_node = UCSNode(successor_state, action, cost, node_ucs, is_hole, new_cost)

              if successor_state not in close_states and not any(n.state == successor_state for n in open_set):
                  open_set[child_node] = child_node.g
              elif any(n.state == successor_state for n in open_set):
                  existing_node = next(n for n in open_set if n.state == successor_state)
                  if existing_node.g > new_cost:
                      del open_set[existing_node]
                      open_set[child_node] = child_node.g

      return [], 0, self.expanded

class ASTARNode:
  def __init__(self,state=None,action=None,cost=None,parent=None,isHole=None, h=0, g=0, f=0):
    self.state = state
    self.action = action
    self.cost = cost
    self.parent = parent
    self.isHole = isHole
    self.h = h
    self.g = g
    self.f = f

class WeightedAStarAgent:

    def __init__(self):
      self.expanded=0

    def path(self, node) -> Tuple[List[int], float, int]:
      sol = []
      total_cost = 0
      while node.parent is not None:
          sol.append(node.action)
          total_cost += node.cost
          node = node.parent
      return sol[::-1], float(total_cost), self.expanded

    def inOpen(self, open_set, state):
      return any(node.state == state for node in open_set.keys())

    def inClose(self, close_nodes, state):
      return next((node for node in close_nodes if node.state == state), None)

    def getNodeByState(self, open_set, state):
      return next((node for node in open_set.keys() if node.state == state), None)

    def get_h_value(self, state, env: 'CampusEnv') -> int:
      row, col = env.to_row_col(state)
      min_distance = min(
          abs(row - env.to_row_col(g)[0]) + abs(col - env.to_row_col(g)[1])
          for g in env.goals
      )
      return min(min_distance, 100)

    def search(self, env: CampusEnv, h_weight: float) -> Tuple[List[int], float ,int]:
      self.expanded = 0
      OPEN = heapdict.heapdict()
      CLOSE_NODES = []
      CLOSE_STATES = []
      startManhatan = self.get_h_value(env.get_initial_state(),env)

      node = ASTARNode(env.get_initial_state(),None,0,None,False,startManhatan ,0, h_weight*startManhatan+0 )
      OPEN[node] = (node.f, node.state)
      while len(OPEN) > 0:
        node_tuple = OPEN.popitem()
        node_astar = node_tuple[0]
        node_state = node_astar.state
        CLOSE_NODES.append(node_astar)
        CLOSE_STATES.append(node_state)
        if env.is_final_state(node_state) == True:
          return self.path(node_astar)
        self.expanded += 1
        if node_astar.isHole == True:
          continue
        for action, successor in env.succ(node_astar.state).items():
          new_g = node_astar.g + successor[1]
          new_h = self.get_h_value(successor[0],env)
          new_f = new_g*(1-h_weight) + new_h*h_weight
          child = ASTARNode(successor[0],action,successor[1],node_astar,successor[2],new_h,new_g,new_f)
          if (not (self.inOpen(OPEN,child.state) == True)) and (not (child.state in CLOSE_STATES)):
            OPEN[child]=(child.f,child.state)
          elif (self.inOpen(OPEN,child.state) == True):
              nodeWithSameState = self.getNodeByState(OPEN,child.state)
              if nodeWithSameState.f > new_f:
                del OPEN[nodeWithSameState]
                OPEN[child] = (child.f, child.state)

      return ([],0,self.expanded)

class AStarAgent:

    def __init__(self):
      self.expanded=0

    def path(self, node) -> Tuple[List[int], float, int]:
      sol = []
      total_cost = 0
      while node.parent is not None:
          sol.append(node.action)
          total_cost += node.cost
          node = node.parent
      return sol[::-1], float(total_cost), self.expanded

    def inOpen(self, open_set, state):
      return any(node.state == state for node in open_set.keys())

    def inClose(self, close_nodes, state):
      return next((node for node in close_nodes if node.state == state), None)

    def getNodeByState(self, open_set, state):
      return next((node for node in open_set.keys() if node.state == state), None)

    def get_h_value(self, state, env: 'CampusEnv') -> int:
      row, col = env.to_row_col(state)
      min_distance = min(
          abs(row - env.to_row_col(g)[0]) + abs(col - env.to_row_col(g)[1])
          for g in env.goals
      )
      return min(min_distance, 100)

    def search(self, env: CampusEnv) -> Tuple[List[int], float ,int]:
      self.expanded = 0
      OPEN = heapdict.heapdict()
      CLOSE_NODES = []
      CLOSE_STATES = []
      startManhatan = self.get_h_value(env.get_initial_state(),env)

      node = ASTARNode(env.get_initial_state(),None,0,None,False,startManhatan ,0, (1/2)*startManhatan+0 )
      OPEN[node] = (node.f, node.state)
      while len(OPEN) > 0:
        node_tuple = OPEN.popitem()
        node_astar = node_tuple[0]
        node_state = node_astar.state
        CLOSE_NODES.append(node_astar)
        CLOSE_STATES.append(node_state)
        if env.is_final_state(node_state) == True:
          return self.path(node_astar)
        self.expanded += 1
        if node_astar.isHole == True:
          continue
        for action, successor in env.succ(node_astar.state).items():
          new_g = node_astar.g + successor[1]
          new_h = self.get_h_value(successor[0],env)
          new_f = new_g*((1/2)) + new_h*(1/2)
          child = ASTARNode(successor[0],action,successor[1],node_astar,successor[2],new_h,new_g,new_f)
          if (not (self.inOpen(OPEN,child.state) == True)) and (not (child.state in CLOSE_STATES)):
            OPEN[child]=(child.f,child.state)
          elif (self.inOpen(OPEN,child.state) == True):
              nodeWithSameState = self.getNodeByState(OPEN,child.state)
              if nodeWithSameState.f > new_f:
                del OPEN[nodeWithSameState]
                OPEN[child] = (child.f, child.state)

      return ([],0,self.expanded)

