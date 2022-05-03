# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 16:59:21 2022

@author: Evan Shapiro
"""
import numpy as np
import geopandas as gpd
import networkx as nx
import osmnx as ox
import copy as copy
#Defining function to create an adjacency list for each node
#G:= Graph dictionary with keys for each node and edge. Each node and edge has
# a dictionary with a key for each property.

#Node i (integer) properties: 
# 'e':= Excess on each node
# 'Adj':= Adjacency list of each node
# 'D' := Distance from sink
#Edge (i,j)-tuple Properties:
# 'cap':= Arc capacity
# 'res':= Residual Capacity of Arc
def graph_con(N, E, Cap, D):
    for i in E:
        G[i] = dict()
        G[(i,j)]['Cap'] = Cap[(i,j)]
        G[(i,j)]['res_cap'] = G[(i,j)]['Cap']
        
    for i in N:
        G[i] = dict()
        G[i]['D'] = D[i]
        G[i]['Adj_L'] = Adj_L[i]
        G[i]['e'] = e[i]   
    return G

def Adj_F(G):
    G.nodes[s]['Adj'] = []
    for t, data in nbrs.items():
        G.nodes[s]['Adj'].append(t)
    return G

# def Adj_f(G):
#     Adj_L = dict()
#     temp_adj = []
#     x =len( G['E'])
#     temp_adj = []
    
#     for i in range(x):
#         temp_adj.append(G['E'][i][0][0])
           
#     for i in range(x):
#         temp = []
#         for j in range(len( G['E'][i])):
#             temp.append(G['E'][i][j][1])
#         Adj_L[temp_adj[i]] = temp
#         G[i]
#     return Adj_L


    

def BFS(G, Adj_L, s):
    
    if 'N' not in G:
        print('Node key N is not defined')
        return
    if 'E' not in G:
        print('Edge key E is not defined')
        return
    
    if s not in G['N']:
        print(s, 'is not a valid node')
        return
    
    
    
    D = dict() #dictionary of distance values
    D[s] = 0
    pred = dict()
    pred[s] = 0
    nxt = 1
    visited_nodes = [s] #Keep track of visited nodes to not make incorrect distance labels
    current_node = [s]
    while current_node:
        n= current_node.pop(0)
        if n in Adj_L: #Check if i is a key value in the adjacency list
            for i in range(len(Adj_L[n])):
                j = Adj_L[n][i]
                if j not in visited_nodes:
                    visited_nodes.append(j)                  
                    pred[j] = n
                    nxt +=1
                    current_node.append(j)
                    D[j] =D[n] +1 # Recording distance labels
    return D

def BFS(G, s):
    
    if s not in G.nodes:
        print(s, 'is not a valid node')
        return   
    G.nodes[s]['D'] = 0
    pred = dict()
    pred[s] = 0
    visited_nodes = [s] #Keep track of visited nodes to not make incorrect distance labels
    current_node = [s]
    while current_node:
        n= current_node.pop(0)
        if G.nodes[n]['Adj']: #Check if i is a key value in the adjacency list
            for i in range(len(G.nodes[n]['Adj'])):
                j = G.nodes[n]['Adj'][i]
                if j not in visited_nodes:
                    visited_nodes.append(j)                  
                    pred[j] = n
                    current_node.append(j)
                    G.nodes[j]['D'] = G.nodes[n]['D'] +1
    return G


# def preprocess(G, Adj_L, Cap,D, s):
    
#     if s not in G['N']:
#         print(s, 'is not a valid node')
#         return
    
#     x = dict()
#     e = dict()
#     adj_s = Adj_L[s]
#     for j in adj_s:
#         x[(s,j)] = Cap[(s,j)]
#         e[j] = Cap[(s,j)]
#     D[s] = len(G['N'])
#     return D, x 





def preprocess(G, s):
    
    if s not in G.nodes:
        print(s, 'is not a valid node')
        return
    
    G.nodes[s]['D'] = len(G.nodes) #Set distance label of sink to |N|
    
    adj_s = G.nodes[s]['Adj'] 
    active = []
    for j in adj_s:
        G.edges[(s,j,_)]['Flow'] = G.edges[(s,j,_)]['Capacity'] 
        G.nodes[j]['Excess'] = G.edges[(s,j,_)]['Capacity'] 
        if G.nodes[j]['Excess'] >0:
            active.append(j)
    
    return G, active

# def push_relabel(G, i): #Adj_L,D,e,
#     adj_l= G[i]['Adj_L']
#     for j in adj_l[i]:
#         if (i,j) in G and G[i][D] == G[j][D] +1:
#             delta = min(G[i]['e'], G[(i,j)]['cap']) #choose a
#             #Update residual network
#             G[j]['e'] -= delta
#             G[j]['e'] += delta 
#             G[(i,j)]['res_cap'] -= delta
#             G[(j,i)]['res_cap'] += delta
#             if i not in G[j]['Adj_l']:
#                 G[j]['Adj_L'].append(i)
            
#             #Do we need to explicitly remove arcs whose residual capacity is 0
#         else:
#             #Construct a set of all distance labels for nodes in the adjacency 
#             #list of node i
#             adj_dis = dis(G[i]['Adj_L'])
#             G[i]['D'] = min
                
                
#         s = j
#         adj_s = Adj_L[s]
 
#Need to define G.edges[(i,j)]['R_Cap'] for all (i,j) in residual network
def push_relabel(G, i): #Adj_L,D,e,
    adj_l= copy.deepcopy(G.nodes[i]['Adj'])
    cur_dis = []
    for j in adj_l:
        cur_dis.append(G.nodes[j]['D'])
    if G.nodes[i]['D'] +1 in cur_dis:
        #Check to see if (i,j) is in the residual network
        #if (i,j,_) in G.edges and G.nodes[i]['D'] == G.nodes[j]['D'] +1:
        delta = min(G.nodes[i]['Excess'],G.edges[(i,j)]['R_Cap']) #G.edges[(i,j)]['Capacity']) #choose a
        
        #Update residual network
        #Check if G.nodes[i]['Excess'] exists. If it does ad delta flow to
        #the excess
        if G.nodes[i]['Excess']:
            G.nodes[i]['Excess'] -= delta
        else:
            G.nodes[i]['Excess'] = -delta
            
        if G.nodes[j]['Excess']:
            G.nodes[j]['Excess'] += delta
        else:
            G.nodes[j]['Excess'] = delta
            
      
        G[(i,j)]['R_Cap'] -= delta
        G[(j,i)]['R_cap'] += delta
        #if i not in G[j]['Adj_l']:
            #G[j]['Adj_L'].append(i)
        
        #Do we need to explicitly remove arcs whose residual capacity is 0
    else:
        while cur_dis:
            #Construct a set of all distance labels for nodes in the adjacency 
            #list of node i
            #G.nodes[i]['D'] = [G.nodes[j]['D'] +1: (i,j) in G.edges and G.edges[(i,j)]['R_cap'] >0 ]
            d_min = min(cur_dis)
            index = cur_dis.index(d_min)
            upd_node = adj_l[index] #G.nodes[i]['Adj'][index]
            if G.nodes[upd_node]['R_cap'] > 0:
                G.nodes[i]['D'] = G.nodes[upd_node]['D'] + 1
            else:
                cur_dis.pop(index)
                adj_l.pop(index)
        
        
        
                



def pre_flow_push(G,s):
    G, active_nodes = preprocess(G)
    while active_nodes:
        n= active_nodes.pop(0)
        push_relabel(G, n)
    #Need to add nodes to active node list in push relabel algorithm
        

#Retrieve a list of distance labels for nodes in adjacency list of node i 
#Input is Adjacency List of a node i  
def adj_dis(G,i):
    

#Notice that x is actually the excess     


        

    
    
    

G = {'N':[1,2,3,4,5],
     'E':[[(1,2),(1,4)],[(2,3)], [(4,2),(4,5)], [(5,3),(5,1)]]} 

Cap ={(1,2): 4,
      (1,4): 10,
      (2,3): 3,
      (4,2): 9,
      (4,5): 4,
      (5,3): 1,
      (5,1): 10
      } 

Adj_L = Adj_f(G)
D = BFS(G,Adj_L,4)
       
        

        