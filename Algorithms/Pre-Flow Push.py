# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 16:59:21 2022

@author: Evan Shapiro
"""

#Defining function to create an adjacency list for each node
#G:= Graph dictionary with keys: 'N' for nodes, 'E' for edges, 'C' for capacity
#Adj_L:= Dictionary with nodes as key values
def Adj_f(G):
    Adj_L = dict()
    temp_adj = []
    x = range(len(G['E']))
    for i in x:
        temp_adj = []
        for j in range(len(G['E'][i])):
            temp_adj.append(G['E'][i][j][1])
        Adj_L[i+1]=temp_adj
    return Adj_L


    

def BFS(G, s,t):
    
    if 'N' not in G:
        print('Node key N is not defined')
        return
    if 'E' not in G:
        print('Edge key E is not defined')
        return
    
    if s not in G['N']:
        print(s, 'is not a valid node')
        return
    if t not in G['N']:
        print(t, 'is not a valid node')
        return
    
    Adj_L = Adj_f(G)
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
        
        
G = {'N':[1,2,3,4,5],
     'E':[[(1,2),(1,4)],[(2,3)], [(4,2),(4,5)], [(5,3),(5,1)]]} 

D = BFS(G,1,3)
       
        

        