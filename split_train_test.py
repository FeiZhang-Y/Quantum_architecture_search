import os
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Subset
from config import task_config
import utils

def split_train_index(args):
    
    trainindex, validindex, testindex = [], [], []
    
    for i in range(3):
        indices = list(range(0+10000*i, 10000+10000*i))
        np.random.shuffle(indices)
        split1 = int(np.floor(args.training_set_rate * 10000))
        split2 = int(np.floor(args.validation_set_rate * 10000))
        tt_trainindex, tt_validindex, tt_testindex = indices[:split1], indices[split1:split1 + split2], indices[split1 + split2:]
        trainindex.extend(tt_trainindex)
        validindex.extend(tt_validindex)
        testindex.extend(tt_testindex)
    
    return trainindex, validindex, testindex
    
    
def circuit_deepth(circuit):

    deepth = []
    No_two_qubit = []
    for i in range(0, len(circuit)):
        cir = np.array(circuit[i])
        res = [0] * len(np.unique(cir))
        num_two_qubit_gates = 0
        max_depth = 0
        for j in cir:
            if j[1] != j[2]:
                num_two_qubit_gates += 1
                max_depth = max(res[j[1]], res[j[2]]) + 1
                res[j[1]], res[j[2]] = max_depth, max_depth
            else:
                res[j[1]] += 1
                max_depth = max(max_depth, res[j[1]])

        deepth.append(max_depth)
        No_two_qubit.append(num_two_qubit_gates)

    return No_two_qubit, deepth

def createData(args):

    exp_type = args.expressibility_type
    if args.noise:
        f_name = "_Noisy"
    else:
        f_name = ""
    if exp_type == "KL" and args.KL_Rel:
        r_name = "_R"
    else:
        r_name = ""
    current_path = os.path.join(os.path.abspath('.'), 'GeneratedCircuits')
    current_path = current_path.replace("\\", "/")
    
    list_cir, exp_cir, graph_encoding, laplacian_matrix = [], [], [], []
    for qubitInfo in ["4","5","6"]:
        
        currentPath = current_path + f'/qubit-{qubitInfo}/Graph_{exp_type}{f_name}{r_name}'
    
        list_cir0 = utils.load_pkl(f'{currentPath}/list_circuit.pkl')
        exp_cir0 = utils.load_pkl(f'{currentPath}/list_exp.pkl')
        graph_encoding0 = utils.load_pkl(f'{currentPath}/list_graph.pkl')
        # laplacian_matrix = utils.load_pkl(f'{currentPath}/list_laplacian.pkl')
        laplacian_matrix0 = utils.load_pkl(f'{currentPath}/list_adj.pkl')
        
        list_cir.extend(list_cir0)
        exp_cir.extend(exp_cir0)
        graph_encoding.extend(graph_encoding0)
        laplacian_matrix.extend(laplacian_matrix0)
    
    return list_cir, exp_cir, graph_encoding, laplacian_matrix

def train_test_data(args, trainindex, validindex, testindex):
    
    exp_type = args.expressibility_type
    if args.noise:
        f_name = "_Noisy"
    else:
        f_name = ""
    if exp_type == "KL" and args.KL_Rel:
        r_name = "_R"
    else:
        r_name = ""
    
    list_cir, exp_cir, graph_encoding, laplacian_matrix = createData(args)
    
    No_two_qubit, deepth = circuit_deepth(list_cir)
    exp_cir = np.array(exp_cir)
    exp_cir = np.insert(exp_cir, 2, No_two_qubit, axis=1)
    exp_cir = np.insert(exp_cir, 3, deepth, axis=1)

    g_e_train = Subset(graph_encoding, trainindex)
    l_m_train = Subset(laplacian_matrix, trainindex)
    exp_train = Subset(exp_cir, trainindex)
    g_e_train = torch.tensor(np.array(g_e_train),dtype=torch.float)
    l_m_train = torch.tensor(np.array(l_m_train),dtype=torch.float)
    exp_train = torch.tensor(np.array(exp_train),dtype=torch.float)
    y_train, y_tindex = exp_train[:, -1], exp_train[:, :-1]
    train_data = TensorDataset(g_e_train, l_m_train, y_train, y_tindex,torch.tensor(trainindex))

    if len(validindex) != 0:
        g_e_valid = Subset(graph_encoding, validindex)
        l_m_valid = Subset(laplacian_matrix, validindex)
        exp_valid = Subset(exp_cir, validindex)
        g_e_valid = torch.tensor(np.array(g_e_valid),dtype=torch.float)
        l_m_valid = torch.tensor(np.array(l_m_valid),dtype=torch.float)
        exp_valid = torch.tensor(np.array(exp_valid),dtype=torch.float)
        y_valid, y_vindex = exp_valid[:, -1], exp_valid[:, :-1]
        valid_data = TensorDataset(g_e_valid, l_m_valid, y_valid, y_vindex,torch.tensor(validindex))
        valid_loader = DataLoader(valid_data, batch_size = 64, shuffle = False, drop_last = False)
    else:
        valid_data = []
        valid_loader= []
    
    if len(validindex) != 0:
        g_e_test = Subset(graph_encoding, testindex)
        l_m_test = Subset(laplacian_matrix, testindex)
        exp_test = Subset(exp_cir, testindex)
        g_e_test = torch.tensor(np.array(g_e_test),dtype=torch.float)
        l_m_test = torch.tensor(np.array(l_m_test),dtype=torch.float)
        exp_test = torch.tensor(np.array(exp_test),dtype=torch.float)
        y_test, y_index = exp_test[:, -1], exp_test[:, :-1]
        test_data = TensorDataset(g_e_test, l_m_test, y_test, y_index,torch.tensor(testindex))
        test_loader = DataLoader(test_data, batch_size=64, shuffle=False, drop_last=False)
    else:
        test_data = []
        test_loader = []
    
    # save_name = f"GeneratedCircuits/TrainTest_{exp_type}{f_name}{r_name}.pt"
    save_name = f"GeneratedCircuits/Total_{exp_type}{f_name}{r_name}.pt"
    torch.save({'train': train_data, 'valid': valid_data, 'test':test_data,
                'validloader': valid_loader,'testloader': test_loader}, save_name)
    
    return 0

if __name__ == '__main__':

    args = task_config("Mix")
    trainindex, validindex, testindex = split_train_index(args)
    
    
    for expType in ["MMD", "KL"]:
        args.expressibility_type = expType
        args.KL_Rel = False
        args.noise = False
        train_test_data(args, trainindex, validindex, testindex)
        
    for expType in ["KL"]:
        args.expressibility_type = expType
        args.KL_Rel = True
        args.noise = False
        train_test_data(args, trainindex, validindex, testindex)
        
    for expType in ["MMD"]:
        args.expressibility_type = expType
        args.KL_Rel = False
        args.noise = True
        train_test_data(args, trainindex, validindex, testindex)
    