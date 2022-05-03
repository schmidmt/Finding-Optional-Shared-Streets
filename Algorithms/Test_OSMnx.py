# -*- coding: utf-8 -*-
"""
Created on Mon May  2 18:30:45 2022

@author: Evan Shapiro
"""
import numpy as np
import copy as copy
import pandas as pd
import geopandas as gpd
import osmnx as ox
import networkx as nx
from shapely.ops import unary_union
import util
import seaborn as sns
import numpy as np
from matplotlib import cm, colors
import matplotlib.pyplot as plt

# Enable OSMNX cache
ox.config(use_cache=True)

def Adj_f_Res(G):            
    for s, nbrs in G.adjacency():
            #print(nbrs,':')
            G.nodes[s]['Adj'] = []
            for t, data in nbrs.items():
                G.nodes[s]['Adj'].append(t)
                #print(G.nodes[s]['Adj'])
    return G

def Res(G):
    R = nx.DiGraph()
    #Keeping track of edges from the original graph
    R.add_nodes_from(G.nodes)
    R.add_edges_from(  
        (u, v, {"capacity": c}) for (u, v, c) in G.edges(data="capacity"))

                     
    for u,v in G.edges:
        #print(G[u][v])
        if not G.has_edge(v,u):
            G.add_edge(v,u)
            G.edges[v,u]['capacity'] = 0
            
    G = Adj_f_Res(G)
    
    for u in G.nodes:
        G.nodes[u]['excess'] = 0
    for (u,v) in G.edges:
        G.edges[(u,v)]['flow'] = 0
    for (u,v) in R.edges:
        print((u,v))
    return G, R 

def BFS_res(R, s):
    
    if s not in R.nodes:
        print(s, 'is not a valid node')
        return   
    R.nodes[s]['D'] = 0
    pred = dict()
    pred[s] = 0
    visited_nodes = [s] #Keep track of visited nodes to not make incorrect distance labels
    current_node = [s]
    while current_node:
        n= current_node.pop(0)
        if R.nodes[n]['Adj']: #Check if i is a key value in the adjacency list
            for i in range(len(R.nodes[n]['Adj'])):
                j = R.nodes[n]['Adj'][i]
                if j not in visited_nodes:
                    visited_nodes.append(j)                  
                    pred[j] = n
                    current_node.append(j)
                    R.nodes[j]['D'] = R.nodes[n]['D'] +1
    return R

def preprocess_res(R, s):
    
    if s not in R.nodes:
        print(s, 'is not a valid node')
        return
    
    R.nodes[s]['D'] = len(R.nodes) #Set distance label of sink to |N|
    
    adj_s = R.nodes[s]['Adj'] 
    active = []
    for j in adj_s:
        
        R.nodes[j]['Adj'].append(s)
        
        delta = R[s][j]['capacity'] 
        R.nodes[j]['excess'] = delta
        print(R.nodes[j]['excess'])
        R[s][j]['capacity'] = 0
        R[j][s]['capacity'] = delta
        if R.nodes[j]['excess'] >0:
            active.append(j)
    
    return R, active

def push_relabel_res(R, i, active_nodes,s,t): #Adj_L,D,e,
    #adj_l= copy.deepcopy(R.nodes[i]['Adj'])
    cur_dis = []
    rec_node = []
    #flows = {}
    #for (u,v) in R.edges:
        #flows[(u,v)]=0
       
    while R.nodes[i]['excess'] > 0:   
        f_nodes = []
        b_nodes = []
        for j in R.nodes[i]['Adj']:
            # if R.nodes[i]['D'] == R.nodes[j]['D'] +1 and R[i][j]['capacity'] >0:
            #     f_nodes.append(j)
            # else R.nodes[i]['D'] != R.nodes[j]['D'] +1 and R[i][j]['capacity'] >0:
            #     b_nodes.append(j)
            #print(R.nodes[i]['excess'])
            cur_dis.append(R.nodes[j]['D'])
        #Given a node (i), we check to see if there is an admissible arc (i,j) in the residual network
        #If there is, push flow across the edge.
            if R.nodes[i]['D'] == R.nodes[j]['D'] +1 and R[i][j]['capacity'] >0:
                if j != s and j !=t:
                    active_nodes.append(j)
                R.nodes[j]['Adj'].append(i)
                #Check to see if (i,j) is in the residual network
                #if (i,j,_) in G.edges and G.nodes[i]['D'] == G.nodes[j]['D'] +1:
                delta = min(R.nodes[i]['excess'],R[i][j]['capacity']) #G.edges[(i,j)]['Capacity']) #choose a
                
                #Update residual network
                #Check if G.nodes[i]['Excess'] exists. If it does ad delta flow to
                #the excess
                R.nodes[i]['excess'] -= delta
                    
                if R.nodes[j]['excess']:
                    R.nodes[j]['excess'] += delta
                else:
                    R.nodes[j]['excess'] = delta
                    
                # if flows[(i,j)]:
                #     flows[(i,j)] += delta
                # else:
                #     flows[(i,j)] = delta
                    
              
                R[i][j]['capacity'] -= delta
                R[j][i]['capacity'] += delta
                #if i not in G[j]['Adj_l']:
                    #G[j]['Adj_L'].append(i)
                
                #Do we need to explicitly remove arcs whose residual capacity is 0
            #If there is no admissible arc (i,j) 
            elif R.nodes[i]['D'] < R.nodes[j]['D'] and  R[i][j]['capacity'] >0:
                rec_node.append(j)
        #Checking to see if there is still excess on the node and there are no admissible arcs.   
        if  R.nodes[i]['excess'] > 0 and rec_node:
            #print('yes')
            #print(R.nodes[i]['excess'])
                    #Construct a set of all distance labels for nodes in the adjacency 
                    #list of node i
                    #G.nodes[i]['D'] = [G.nodes[j]['D'] +1: (i,j) in G.edges and G.edges[(i,j)]['R_cap'] >0 ]
            dis = []
            for j in rec_node:
                    dis.append(R.nodes[j]['D'])
            d_min = min(dis)
            index = dis.index(d_min)
            j = rec_node[index]
            if j != s and j !=t:
                active_nodes.append(j)
                
            R.nodes[j]['Adj'].append(i)
            R.nodes[i]['D'] = d_min + 1
            delta = min(R.nodes[i]['excess'],R[i][j]['capacity']) #G.edges[(i,j)]['Capacity']) #choose a
                
                #Update residual network
                #Check if G.nodes[i]['Excess'] exists. If it does ad delta flow to
                #the excess
            
            R.nodes[i]['excess'] -= delta
            if R.nodes[j]['excess']:
                R.nodes[j]['excess'] += delta
            else:
                R.nodes[j]['excess'] = delta
                
            # if flows[(i,j)]:
            #         flows[(i,j)]+= delta
            # else:
            #         flows = {}
            #         flows[(i,j)] = delta  
          
            R[i][j]['capacity'] -= delta
            R[j][i]['capacity'] += delta
            rec_node.pop(0)
            #print(rec_node)
            # else:
            #     cur_dis.pop(index)
            #     adj_l.pop(index)
    return active_nodes, R

def pre_flow_push(G,s,t):
    flow = {}
    G, R = Res(G)
    G = Adj_f_Res(G)
    G = BFS_res(G,t)
    G, active_nodes = preprocess_res(G,s)
    while active_nodes:
        i = active_nodes.pop(0)
        #print(i)
        if G.nodes[i]['excess'] > 0:
            active_nodes, G  = push_relabel_res(G, i, active_nodes,s,t)
            #print(active_nodes)
            print(G.nodes[i]['excess'] )
    for (u,v) in G.edges:
        if (u,v) not in R.edges:
            flow[(v,u)] = G.edges[(u,v)]['capacity']
    #print(G.nodes[t]['excess'])
    flow_value = G.nodes[t]['excess']
    print(flow_value)
    return flow_value, flow
    
    #Need to add nodes to active node list in push relabel algorithm
    
from unittest import TestCase
class TestPreflowPush(TestCase):
    """Tests for preflow_push"""

    def test_simple(self):
        """Test a simple graph for max-flow"""
        graph = nx.DiGraph()

        graph.add_edges_from(
            [
                (0, 1, {"capacity": 15}),
                (0, 3, {"capacity": 4}),
                (1, 2, {"capacity": 12}),
                (2, 3, {"capacity": 3}),
                (2, 5, {"capacity": 7}),
                (3, 4, {"capacity": 10}),
                (4, 1, {"capacity": 5}),
                (4, 5, {"capacity": 10}),
            ]
        )

        flow_value, flows = pre_flow_push(graph, 0, 5)

        self.assertEqual(flow_value, 14)
        self.assertEqual(
            flows,
            {
                (0, 1): 10,
                (0, 3): 4,
                (2, 3): 3,
                (1, 2): 10,
                (2, 5): 7,
                (4, 1): 0,
                (4, 5): 7,
                (3, 4): 7,
            },
        )
import unittest
unittest.main(argv=[''], verbosity=2, exit=False)