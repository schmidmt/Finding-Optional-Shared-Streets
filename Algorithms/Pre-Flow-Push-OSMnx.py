# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 12:11:48 2022

@author: Evan Shapiro
"""
from copy import deepcopy
# def Adj_f(G):
#     for s, nbrs in G.adjacency():
#         #print(s,':')
#         G.nodes[s]['Adj'] = []
#         for t, data in nbrs.items():
#             G.nodes[s]['Adj'].append(t)

def Adj_f_Res(G,R):            
    for s, nbrs in G.adjacency():
            #print(nbrs,':')
            R.nodes[s]['Adj'] = []
            for t, data in nbrs.items():
                R.nodes[s]['Adj'].append(t)
    return R

def Res(G):
    R = nx.DiGraph()
    R.add_nodes_from(G)

    #inf = float("inf")
    # Extract edges with positive capacities. Self loops excluded.
    edge_list = [
        (u, v, attr)
        for u, v, attr in G.edges(data=True)
        if u != v and attr.get(capacity, inf) > 0
    ]
    for u,v, attr in edge_list:
        r = attr.get(capacity)
        if not R.has_edge(u,v):
            R.add_edge(u,v, capacity = r)
            R.add_edge(v,u, capacity = 0)
        else:
            R[u][v]["capacity"]=r
    R = Adj_f_Res(G,R)
    for u,v in edge_list:
        R[u][v]['flow'] = 0
        R[u][v]['excess'] = 0
    return R        
    



            


# def BFS(G, s):
    
#     if s not in G.nodes:
#         print(s, 'is not a valid node')
#         return   
#     G.nodes[s]['D'] = 0
#     pred = dict()
#     pred[s] = 0
#     visited_nodes = [s] #Keep track of visited nodes to not make incorrect distance labels
#     current_node = [s]
#     while current_node:
#         n= current_node.pop(0)
#         if G.nodes[n]['Adj']: #Check if i is a key value in the adjacency list
#             for i in range(len(G.nodes[n]['Adj'])):
#                 j = G.nodes[n]['Adj'][i]
#                 if j not in visited_nodes:
#                     visited_nodes.append(j)                  
#                     pred[j] = n
#                     current_node.append(j)
#                     G.nodes[j]['D'] = G.nodes[n]['D'] +1
#     return G

def BFS_res(R, s):
    
    if s not in R.nodes:
        print(s, 'is not a valid node')
        return   
    R[s]['D'] = 0
    pred = dict()
    pred[s] = 0
    visited_nodes = [s] #Keep track of visited nodes to not make incorrect distance labels
    current_node = [s]
    while current_node:
        n= current_node.pop(0)
        if R[n]['Adj']: #Check if i is a key value in the adjacency list
            for i in range(len(R[n]['Adj'])):
                j = R[n]['Adj'][i]
                if j not in visited_nodes:
                    visited_nodes.append(j)                  
                    pred[j] = n
                    current_node.append(j)
                    R[j]['D'] = R[n]['D'] +1
    return R

# def preprocess(G, s):
    
#     if s not in G.nodes:
#         print(s, 'is not a valid node')
#         return
    
#     G.nodes[s]['D'] = len(G.nodes) #Set distance label of sink to |N|
    
#     adj_s = G.nodes[s]['Adj'] 
#     active = []
#     for j in adj_s:
#         G.edges[(s,j,_)]['Flow'] = G.edges[(s,j,_)]['Capacity'] 
#         G.nodes[j]['Excess'] = G.edges[(s,j,_)]['Capacity'] 
#         if G.nodes[j]['Excess'] >0:
#             active.append(j)
    
#     return G, active



def preprocess_res(R, s):
    
    if s not in G.nodes:
        print(s, 'is not a valid node')
        return
    
    R[s]['D'] = len(R.nodes) #Set distance label of sink to |N|
    
    adj_s = G[s]['Adj'] 
    active = []
    for j in adj_s:
        R[s][j]['flow'] = R[s][j]['capacity'] 
        R[j]['excess'] = R[s][j]['capacity'] 
        if R[j]['excess'] >0:
            active.append(j)
    
    return R, active


# def push_relabel(G, i): #Adj_L,D,e,
#     adj_l= copy.deepcopy(G.nodes[i]['Adj'])
#     cur_dis = []
#     for j in adj_l:
#         cur_dis.append(G.nodes[j]['D'])
#     if G.nodes[i]['D'] +1 in cur_dis:
#         #Check to see if (i,j) is in the residual network
#         #if (i,j,_) in G.edges and G.nodes[i]['D'] == G.nodes[j]['D'] +1:
#         delta = min(G.nodes[i]['Excess'],G.edges[(i,j)]['R_Cap']) #G.edges[(i,j)]['Capacity']) #choose a
        
#         #Update residual network
#         #Check if G.nodes[i]['Excess'] exists. If it does ad delta flow to
#         #the excess
#         if G.nodes[i]['Excess']:
#             G.nodes[i]['Excess'] -= delta
#         else:
#             G.nodes[i]['Excess'] = -delta
            
#         if G.nodes[j]['Excess']:
#             G.nodes[j]['Excess'] += delta
#         else:
#             G.nodes[j]['Excess'] = delta
            
      
#         G[(i,j)]['R_Cap'] -= delta
#         G[(j,i)]['R_cap'] += delta
#         #if i not in G[j]['Adj_l']:
#             #G[j]['Adj_L'].append(i)
        
#         #Do we need to explicitly remove arcs whose residual capacity is 0
#     else:
#         while cur_dis:
#             #Construct a set of all distance labels for nodes in the adjacency 
#             #list of node i
#             #G.nodes[i]['D'] = [G.nodes[j]['D'] +1: (i,j) in G.edges and G.edges[(i,j)]['R_cap'] >0 ]
#             d_min = min(cur_dis)
#             index = cur_dis.index(d_min)
#             upd_node = adj_l[index] #G.nodes[i]['Adj'][index]
#             if G.nodes[upd_node]['R_cap'] > 0:
#                 G.nodes[i]['D'] = G.nodes[upd_node]['D'] + 1
#             else:
#                 cur_dis.pop(index)
#                 adj_l.pop(index)
def push_relabel_res(R, i): #Adj_L,D,e,
    adj_l= copy.deepcopy(R.nodes[i]['Adj'])
    cur_dis = []
    for j in adj_l:
        cur_dis.append(R.nodes[j]['D'])
    if R[i]['D'] +1 in cur_dis:
        #Check to see if (i,j) is in the residual network
        #if (i,j,_) in G.edges and G.nodes[i]['D'] == G.nodes[j]['D'] +1:
        delta = min(R[i]['excess'],R[i][j]['capacity']) #G.edges[(i,j)]['Capacity']) #choose a
        
        #Update residual network
        #Check if G.nodes[i]['Excess'] exists. If it does ad delta flow to
        #the excess
        if G.nodes[i]['excess']:
            R[i]['excess'] -= delta
        else:
            G.nodes[i]['excess'] = -delta
            
        if G.nodes[j]['excess']:
            G.nodes[j]['excess'] += delta
        else:
            G.nodes[j]['excess'] = delta
            
      
        R[i][j]['capacity'] -= delta
        R[j][i]['capacity'] += delta
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
            if R[i][upd_node]['capacity'] > 0:
                R[i]['D'] = R[upd_node]['D'] + 1
            else:
                cur_dis.pop(index)
                adj_l.pop(index)


def push_relabel_res(R, i, active_nodes): #Adj_L,D,e,
    adj_l= copy.deepcopy(R.nodes[i]['Adj'])
    cur_dis = []
    rec_node = []
    while R[i]['excess'] > 0:
        for j in adj_l:
            cur_dis.append(R.nodes[j]['D'])
        #Given a node (i), we check to see if there is an admissible arc (i,j) in the residual network
        #If there is, push flow across the edge.
            if R[i]['D'] == R[j]['D'] +1 and R[i][j]['capacity'] >0:
                active_nodes.append(j)
                #Check to see if (i,j) is in the residual network
                #if (i,j,_) in G.edges and G.nodes[i]['D'] == G.nodes[j]['D'] +1:
                delta = min(R[i]['excess'],R[i][j]['capacity']) #G.edges[(i,j)]['Capacity']) #choose a
                
                #Update residual network
                #Check if G.nodes[i]['Excess'] exists. If it does ad delta flow to
                #the excess
                if R.nodes[i]['excess']:
                    R[i]['excess'] -= delta
                else:
                    R.nodes[i]['excess'] = -delta
                    
                if R.nodes[j]['excess']:
                    R.nodes[j]['excess'] += delta
                else:
                    R.nodes[j]['excess'] = delta
                    
              
                R[i][j]['capacity'] -= delta
                R[j][i]['capacity'] += delta
                #if i not in G[j]['Adj_l']:
                    #G[j]['Adj_L'].append(i)
                
                #Do we need to explicitly remove arcs whose residual capacity is 0
            #If there is no admissible arc (i,j) 
            elif R[i]['D'] <= R[j]['D'] +1 and  R[i][j]['capacity'] >0:
                rec_node.append(j)
                
        while rec_node:
                    #Construct a set of all distance labels for nodes in the adjacency 
                    #list of node i
                    #G.nodes[i]['D'] = [G.nodes[j]['D'] +1: (i,j) in G.edges and G.edges[(i,j)]['R_cap'] >0 ]
                dis = []
                for j in rec_node:
                    dis.append(R[j]['D'])
                d_min = min(dis)
                index = dis.index(d_min)
                j = rec_node.index(index)
                R[i]['D'] = d_min + 1
                delta = min(R[i]['excess'],R[i][j]['capacity']) #G.edges[(i,j)]['Capacity']) #choose a
                
                #Update residual network
                #Check if G.nodes[i]['Excess'] exists. If it does ad delta flow to
                #the excess
                if R.nodes[i]['excess']:
                    R[i]['excess'] -= delta
                else:
                    R.nodes[i]['excess'] = -delta
                    
                if R.nodes[j]['excess']:
                    R.nodes[j]['excess'] += delta
                else:
                    R.nodes[j]['excess'] = delta
                    
              
                R[i][j]['capacity'] -= delta
                R[j][i]['capacity'] += delta
                # else:
                #     cur_dis.pop(index)
                #     adj_l.pop(index)
        return active_nodes, R