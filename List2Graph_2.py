import os
import numpy as np
import networkx as nx
import config
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv
import argparse
import os
from config import task_config


def No_QubitsGate(data):
    if isinstance(data,str):
        data = eval(data)
    gatenum = len(data)
    tt = list(zip(*data))
    qubits = max(max(tt[1]), max(tt[2])) + 1

    return qubits,gatenum

def list_to_adj(data):
    
    # Largest_data = data[-1]
    max_qubits, max_gatenum = 6, 39
    
    res_adj = []
    res_graph = []
    res_laplacian = []  
    for i, list_arc in tqdm(enumerate(data), desc='list to adj'):
        
        N = No_QubitsGate(list_arc)[0]
        list_arc = make_it_unique(list_arc, N)
        temp_op = []
        
        graph = nx.DiGraph()
        graph.add_node('start', label = 'start') 
        for j in range(0, len(list_arc)):  
            graph.add_node(j, label = list_arc[j])
        for j in range(len(list_arc),max_gatenum):  
            graph.add_node(j, label = [0,0,0])
        graph.add_node('end', label = 'end')  
        
        
        last = ['start' for _ in range(N)]  

        for k in range(0, len(list_arc)):  
            if list_arc[k][1] == list_arc[k][2]:  
                graph.add_edge(last[list_arc[k][1]], k)
                last[list_arc[k][1]] = k
            else:  
                graph.add_edge(last[list_arc[k][1]], k)
                graph.add_edge(last[list_arc[k][2]], k)
                last[list_arc[k][1]] = k
                last[list_arc[k][2]] = k
            # print(last)
            # print(k)

        for _ in last:  
            graph.add_edge(_, 'end')

        # nx.draw_networkx(graph)
        # plt.show()

        gate_type = 2
        #  encoding
        for node in graph.nodes:  
            if node == 'start':
                t1 = [0 for _ in range(gate_type + 2)]
                # t2 = [1 for _ in range(N)]
                t2 = [1 for _ in range(N)]
                t3 = [0 for _ in range(max_qubits - N)]
                t2.extend(t3)
                t1[0] = 1
                t1.extend(t2)
                temp_op.append(t1)
            elif node == 'end':
                t1 = [0 for _ in range(gate_type + 2)]
                t2 = [1 for _ in range(N)]
                t3 = [0 for _ in range(max_qubits - N)]
                t2.extend(t3)
                t1[-1] = 1
                t1.extend(t2)
                temp_op.append(t1)
            else:
                t1 = [0 for _ in range(gate_type + 2)]
                t2 = [0 for _ in range(max_qubits)]
                if node < len(list_arc):
                    t1[int(graph.nodes[node]['label'][0])] = 1
                    t2[int(graph.nodes[node]['label'][1])] = 1
                    t2[int(graph.nodes[node]['label'][2])] = 1

                t1.extend(t2)
                temp_op.append(t1)
            assert len(t1)== max_qubits + gate_type + 2
        
        temp_adj = nx.adjacency_matrix(graph).todense()
        res_adj.append(temp_adj)
        
        temp_adj = -1 * temp_adj

        #  degree
        for j in range(0, len(temp_adj)):
            temp_adj[j,j] = -1 * np.sum(temp_adj[j])

        res_laplacian.append(temp_adj)
        res_graph.append(temp_op)

    return res_graph,res_adj, res_laplacian

def make_it_unique(arc, num_qubit):
    
    if isinstance(arc,str):
        arc = eval(arc)
        
    lists = []
    final_list = []

    for i in range(0, num_qubit):
        lists.append([])

    for gate in arc:  
        if gate[2] != gate[1]:  
            if len(lists[gate[1]]) >= len(lists[gate[2]]):  
                lists[gate[1]].append(gate)  
                while len(lists[gate[1]]) > len(lists[gate[2]]):  
                    lists[gate[2]].append(0)
            else:  
                while len(lists[gate[1]]) < len(lists[gate[2]]):
                    lists[gate[1]].append(0)  
                lists[gate[1]].append(gate)  
                lists[gate[2]].append(0)  

        else:  
            lists[gate[1]].append(gate)

    depth = []
    for i in range(0, num_qubit):
        depth.append(len(lists[i]))
    max_depth = max(depth)
    for i in range(max_depth):
        for j in range(num_qubit):
            if depth[j] - 1 < i:
                continue
            if lists[j][i] != 0:
                final_list.append(lists[j][i])

    return final_list

def dataset(listParas, current_path,args):

    totalCir= []
    totalExp= []

    for index  in enumerate(listParas):

        qubit = index[1][0]
        gate_number = index[1][1]
        lists = []
        circuit = config.load_pkl(f'{current_path}/qubit-{qubit}/list_cir/gate-number-{gate_number}.pkl')
        for i in range(0,500):
            temps = circuit[i]
            tempList = []
            for j in range(0,len(temps)):
                if temps[j].qubits == 1:
                    tempList.append([1] + 2 * temps[j].act_on )
                else:
                    tempList.append([2] + temps[j].act_on )
            lists.append(tempList)
            
        expType = args.expressibility_type
        if args.noise:
            f_name = "_Noisy"
        else:
            f_name = ""
        exp_cir = config.load_pkl(f'{current_path}/qubit-{qubit}/Expressibility_{expType}{f_name}/gate-number-{gate_number}.pkl')
        totalCir.extend(lists)
        a1 = np.repeat(qubit,(500,))
        a2 = np.repeat(gate_number,(500,))
        if expType == "MMD":
            exp_cir = [1 - x for x in exp_cir]
        if expType == "KL" and args.KL_Rel:
            idel = (2 ** qubit - 1) * np.log(100)
            exp_cir = -np.log(exp_cir / idel)
        express = np.c_[a1,a2,exp_cir]
        totalExp.extend(express)

    return totalCir, totalExp

def main(args):
    
    current_path1 = os.path.join(os.path.abspath('.'), 'GeneratedCircuits')
    current_path1 = current_path1.replace("\\","/")
    qubitInfo = args.qubitInfo
    exp_type = args.expressibility_type
    if args.noise:
        f_name = "_Noisy"
    else:
        f_name = ""
    if exp_type=="KL" and args.KL_Rel:
        r_name = "_R"
    else:
        r_name = ""
        
    if qubitInfo == "Mix":
        listParas = np.c_[np.repeat(4, 20), np.arange(10, 30)]
        listParas = np.r_[listParas, np.c_[np.repeat(5, 20), np.arange(15, 35)]]
        listParas = np.r_[listParas, np.c_[np.repeat(6, 20), np.arange(20, 40)]]
        list_arc, exp_arc = dataset(listParas,current_path1,args)
    else:
        currentQubits = int(qubitInfo)
        gateNumStart = (currentQubits-2)*5
        listParas = np.c_[np.repeat(currentQubits, 20), np.arange(gateNumStart, gateNumStart+20)]
        list_arc, exp_arc = dataset(listParas, current_path1,args)


    res_graph, res_adj, res_laplacian = list_to_adj(list_arc)
    current_path = current_path1 + f'/qubit-{qubitInfo}/Graph_{exp_type}{f_name}{r_name}'
    if not os.path.exists(current_path):
        os.makedirs(current_path)
    config.save_pkl(list_arc, f'{current_path}/list_circuit.pkl')
    config.save_pkl(exp_arc, f'{current_path}/list_exp.pkl')
    print('List is ready!')
    config.save_pkl(res_graph, f'{current_path}/list_graph.pkl')
    config.save_pkl(res_adj, f'{current_path}/list_adj.pkl')
    config.save_pkl(res_laplacian, f'{current_path}/list_laplacian.pkl')
    print('Graph is ready')

if __name__ == '__main__':

    for expType in ["MMD","KL"]:
        for qubitInfo in ["4","5","6","Mix"]:  #
            args = task_config(qubitInfo)
            args.expressibility_type = expType
            args.KL_Rel = False
            args.noise = False
            args.qubitInfo = qubitInfo
            main(args)
            
    expType = "MMD"
    for qubitInfo in ["4","5","6","Mix"]:
        args = task_config(qubitInfo)
        args.expressibility_type = expType
        args.KL_Rel = False
        args.noise = True
        args.qubitInfo = qubitInfo
        main(args)
    
    expType = "KL"
    for qubitInfo in ["4", "5", "6", "Mix"]:
        args = task_config(qubitInfo)
        args.expressibility_type = expType
        args.KL_Rel = True
        args.noise = False
        args.qubitInfo = qubitInfo
        main(args)