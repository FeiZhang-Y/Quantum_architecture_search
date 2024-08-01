"""
Randomly generate candidate circuits from search space
"""
import pickle
import numpy as np
import networkx as nx
import utils
from tqdm import tqdm
import pennylane as qml
import matplotlib.pyplot as plt
import pandas as pd
from quantum_gates import Gate
import argparse
import os

 
class ArchitectureGenerator:

    def __init__(self, gate_pool, max_gate_num, num_layers, num_qubits, max_two_qubit_gates_rate,
                 available_edge, not_first, start_with_u):
        self.mean = 0  
        self.standard_deviation = 1.35  
        self.gate_pool = gate_pool  
        self.nt = max_gate_num  
        if num_layers < 0:
            self.D = max_gate_num  
        else:
            self.D = num_layers  
        self.N = num_qubits 
        self.start_with_u = start_with_u  
        self.start_gate = None  
        self.available_edge = available_edge  
        self.not_first = not_first  
        self.max_two_qubit_gates_rate = max_two_qubit_gates_rate  

    def draw_plot(self, arcs, index):
        dev = qml.device('default.qubit', wires=self.N)

        @qml.qnode(dev)
        def circuit(cir):
            for gate in cir:
                if gate.name == 'Hadamard':
                    qml.Hadamard(wires=gate.act_on[0])
                elif gate.name == 'Rx':
                    qml.RX(0, wires=gate.act_on[0])
                elif gate.name == 'Ry':
                    qml.RY(0, wires=gate.act_on[0])
                elif gate.name == 'Rz':
                    qml.RZ(0, wires=gate.act_on[0])
                elif gate.name == 'XX':
                    qml.IsingXX(0, wires=[gate.act_on[0], gate.act_on[1]])
                elif gate.name == 'YY':
                    qml.IsingYY(0, wires=[gate.act_on[0], gate.act_on[1]])
                elif gate.name == 'ZZ':
                    qml.IsingZZ(0, wires=[gate.act_on[0], gate.act_on[1]])
                elif gate.name == 'SWAP':
                    qml.SWAP(wires=[gate.act_on[0], gate.act_on[1]])
                elif gate.name == 'U3':
                    qml.U3(0, 0, 0, wires=gate.act_on[0])
                elif gate.name == 'CZ':
                    qml.CZ(wires=[gate.act_on[0], gate.act_on[1]])
                else:
                    print('invalid gate')
                    exit(0)
            return [qml.expval(qml.PauliZ(q)) for q in range(0, self.N)]

        fig, ax = qml.draw_mpl(circuit)(arcs)
        # plt.show()
        s_p = f'GeneratedCircuits/qubit-{self.N}/img_cir/gate-number-{self.nt}'
        if not os.path.exists(s_p):
            os.makedirs(s_p)
        plt.savefig(f'{s_p}/{index}.png')
        plt.close()
        return 0

    def check_reasonable(self, gate, last_gate_list):

        reasonable = True
        start_gate = True  
        for q in gate.act_on:
            if last_gate_list[q] != 0:
                start_gate = False

        if start_gate:
            for n_f in self.not_first:  
                if n_f["name"] == gate.name:
                    return False

        
        if last_gate_list[gate.act_on[0]] == 0 and gate.name[0] == 'C':
            return False


        if gate.name == 'CZ':
            if last_gate_list[gate.act_on[0]] == 0 or last_gate_list[gate.act_on[1]] == 0:
                return False

        count = 0
        for act_q in gate.act_on:
            if last_gate_list[act_q] == 0:
                break
            elif last_gate_list[act_q].name == gate.name:  
                if last_gate_list[act_q].act_on == gate.act_on:  
                    count += 1  
        if count == len(gate.act_on):  
            reasonable = False

        return reasonable

    def generate_circuit(self, generation_type):  
        normal = np.random.normal(self.mean, self.standard_deviation, len(self.gate_pool))
        log_it_list = normal 
        log_it_list = np.exp(log_it_list) / sum(np.exp(log_it_list))
        last_gate_on_each_qubit = [0] * self.N  
        ciru = []

        if self.start_with_u:
          
            if self.start_gate["qubits"] == 1:  
                for i in range(0, self.N):
                    h_g = Gate(**self.start_gate)
                    h_g.act_on = [i]
                    ciru.append(h_g)
                    last_gate_on_each_qubit[i] = h_g
            elif self.start_gate["qubits"] == 2:  
                for i in range(0, int(np.floor(self.N / 2))):  
                    h_g = Gate(**self.start_gate)
                    h_g.act_on = [2 * i, 2 * i + 1]
                    last_gate_on_each_qubit[2 * i] = h_g
                    last_gate_on_each_qubit[2 * i + 1] = h_g
                    ciru.append(h_g)
            else:
                print('Start gate supposed to be single qubit gate or two-qubit gate!')
                exit(0)

        if generation_type == 0:
            while len(ciru) < self.nt:  
                g = self.add_gate(log_it_list)  
                res = self.check_reasonable(g, last_gate_on_each_qubit)  
                if res:
                    for q in g.act_on:  
                        last_gate_on_each_qubit[q] = g
                    ciru.append(g)


        else:
            print('invalid generation type,supposed to be gate_wise only')
            exit(0)

        return ciru


    def add_gate(self, log_it_list):  

        gate = np.random.choice(a=self.gate_pool, size=1, p=log_it_list).item()  

        if gate['qubits'] > 1:  
            position = np.random.choice(a=len(self.available_edge), size=1).item()
            act_on = self.available_edge[position]  #
            res = Gate(**gate)
            res.act_on = act_on

        else:  
            position = np.random.choice(a=self.N, size=1).item()
            res = Gate(**gate)
            res.act_on = [position]

        return res

    def check_same(self, cir1, cir2):
        if len(cir1) != len(cir2):
            return False
        same = True
        for i in range(0, len(cir1)):
            if cir1[i].name != cir2[i].name:
                same = False
                break
            if cir1[i].act_on != cir2[i].act_on:
                same = False
                break

        return same

    def check(self, cir):  

        res = [0] * self.N  
        no_para = 0  
        num_two_qubit_gates = 0
        keep = True

        for gate in cir:  
            if not gate.para_gate:  
                no_para += 1

            if gate.qubits > 1:  
                num_two_qubit_gates += 1
                depth_q = []
                for q in gate.act_on:
                    depth_q.append(res[q])
                max_depth = max(depth_q)
                max_depth += 1
                for q in gate.act_on:
                    res[q] = max_depth
            else:
                res[gate.act_on[0]] += 1

        for i in res:
            if i > self.D:
                keep = False
                break

        if no_para >= len(cir):  
            keep = False

        if num_two_qubit_gates > int(len(cir) * self.max_two_qubit_gates_rate):  
            keep = False

        return keep

    def get_architectures(self, num_architecture, generate_type):
        cirs = []  
        num = 0
        pbar = tqdm(total=num_architecture, desc='Randomly generating circuits')
        while num < num_architecture:
            temp = self.generate_circuit(generation_type=generate_type)
            if len(temp) > self.nt:
                del temp[self.nt: len(temp)]

            keep = self.check(temp)
            if keep:
                temp = self.make_it_unique(temp)  
                check_same = False
                for c in cirs: 
                    check_same = self.check_same(c, temp)
                    if check_same:
                        break
                if not check_same:
                    cirs.append(temp)
                    num += 1
                    pbar.update(1)
        return cirs

    def make_it_unique(self, arc):
        lists = []   
        final_list = []  

        for i in range(0, self.N):
            lists.append([])

        for gate in arc:
            if len(gate.act_on) > 1:  
                depth_now = []
                for act_q in gate.act_on:  
                    depth_now.append(len(lists[act_q]))
                max_depth = max(depth_now)
                for act_q in gate.act_on:  
                    while len(lists[act_q]) < max_depth:
                        lists[act_q].append(0)
                min_q = min(gate.act_on) 
                lists[min_q].append(gate)
                max_depth_now = len(lists[min_q])
                for act_q in gate.act_on:  
                    while len(lists[act_q]) < max_depth_now:
                        lists[act_q].append(0)
            else:  
                lists[gate.act_on[0]].append(gate)

       
        depth = []  
        for i in range(0, len(lists)):
            depth.append(len(lists[i]))
        max_depth = max(depth) 
        for q in range(self.N):  
            while len(lists[q]) < max_depth:
                lists[q].append(0)
        for i in range(max_depth):  
            for j in range(0, len(lists)):  
                if lists[j][i] != 0:
                    final_list.append(lists[j][i])

        return final_list


def main(args):
    np.random.seed(args.seed)
    print(f'seed:{args.seed}')
    qubit = args.qubits
    gate_start = (qubit-2)*5
    edge_list=[]
    for i in range(0,qubit-1):
        edge_list.append([i,i+1])
    for gate in range(gate_start,gate_start+20):
        ag = ArchitectureGenerator(gate_pool = [{"name": "U3", "qubits": 1, "para_gate": True},
                                                {"name": "CZ", "qubits": 2, "para_gate": False}],  
                                   max_gate_num = gate,  
                                   num_layers = -1,  
                                   num_qubits = qubit,  
                                   max_two_qubit_gates_rate = 2,  
                                   available_edge = edge_list,  
                                   not_first = [{"name": "CZ", "qubits": 2, "para_gate": False}],  
                                   start_with_u = True,  
                                   )
        ag.start_gate = {"name": "U3", "qubits": 1, "para_gate": True}  
        
        s_p_l = f'GeneratedCircuits/qubit-{qubit}/list_cir'
        if not os.path.exists(s_p_l):
            os.makedirs(s_p_l)
        utils.save_pkl(list_arc, f'{s_p_l}/gate-number-{gate}.pkl')
        for c_index, l_arc in enumerate(list_arc):
            ag.draw_plot(l_arc, c_index)
        # matrix_rep = ag.list_to_adj(list_arc)
        print(f'Qubit: {qubit} GateNumber: {gate} Successful')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--qubit', type=int, default=7, help='qubit')
    parser.add_argument('--seed', type=int, default=0, help='gate number')
    args = parser.parse_args()
    for qubit in range(7,20):
        parser.set_defaults(qubits = qubit)
        args = parser.parse_args()
        main(args)
    
