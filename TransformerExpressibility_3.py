import os
import sys
import networkx as nx
import torch
import numpy as np
import utils
import argparse
from torch.utils.data import TensorDataset, DataLoader, Subset
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:2'  # !(important)It makes rnn_based model deterministic
from TransformerModel import TransformerPredictor
from TransformerModel import EarlyStopping
import csv
import random
import datetime
from scipy import stats
from config import task_config
from sklearn.metrics import r2_score
import pandas as pd
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def analysisResult(trueE,predE):
    loss1 = torch.nn.MSELoss()
    loss2 = torch.nn.SmoothL1Loss(reduction = 'mean')
    trueE = torch.tensor(np.array(trueE))
    predE = torch.tensor(np.array(predE))
    RMSELoss = torch.sqrt(loss1(predE, trueE[:, -1])).tolist()
    huberLoss = loss2(predE, trueE[:, -1]).tolist()
    spearmanr = stats.spearmanr(trueE[:, -1], predE)[0]
    kendall = stats.kendalltau(trueE[:, -1], predE)[0]
    r2 = r2_score(np.array(trueE[:, -1]), np.array(predE))
    result = [RMSELoss, huberLoss, spearmanr,kendall,r2]

    sortd = np.argsort(-trueE[:,-1])
    ind = range(0,int(0.1*len(trueE)))
    top10_trueE = trueE[sortd[ind],-1]
    top10_predE = predE[sortd[ind]]
    RMSELoss = torch.sqrt(loss1(top10_predE, top10_trueE)).tolist()
    huberLoss = loss2(top10_predE, top10_trueE).tolist()
    spearmanr = stats.spearmanr(top10_trueE, top10_predE)[0]
    kendall = stats.kendalltau(top10_trueE, top10_predE)[0]
    r2 = r2_score(np.array(top10_trueE), np.array(top10_predE))
    result.extend([RMSELoss, huberLoss, spearmanr,kendall,r2])

    return result

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

    qubitInfo = args.qubitInfo
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
    currentPath = current_path + f'/qubit-{qubitInfo}/Graph_{exp_type}{f_name}{r_name}'
    file0 = f"Result-head-{args.nhead}-layer-{args.num_layers}-hiddenDim-{args.hidden_dim}/"
    save_path = file0 + f"qubit-{qubitInfo}/Predict_{exp_type}{f_name}{r_name}/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    list_cir = utils.load_pkl(f'{currentPath}/list_circuit.pkl')
    exp_cir = utils.load_pkl(f'{currentPath}/list_exp.pkl')
    graph_encoding = utils.load_pkl(f'{currentPath}/list_graph.pkl')
    # laplacian_matrix = utils.load_pkl(f'{currentPath}/list_laplacian.pkl')
    laplacian_matrix = utils.load_pkl(f'{currentPath}/list_adj.pkl')
    No_two_qubit, deepth = circuit_deepth(list_cir)
    exp_cir = np.array(exp_cir)
    exp_cir = np.insert(exp_cir, 2, No_two_qubit, axis=1)
    exp_cir = np.insert(exp_cir, 3, deepth, axis=1)

    indices = list(range(0,len(list_cir)))
    np.random.shuffle(indices)
    split1 = int(np.floor(args.training_set_rate * len(list_cir)))
    split2 = int(np.floor(args.validation_set_rate * len(list_cir)))

    trainindex, validindex, testindex = indices[:split1], indices[split1:split1+split2], indices[split1+split2:]

    g_e_train = Subset(graph_encoding, trainindex)
    l_m_train = Subset(laplacian_matrix, trainindex)
    exp_train = Subset(exp_cir, trainindex)
    g_e_train = torch.tensor(np.array(g_e_train),dtype=torch.float)
    l_m_train = torch.tensor(np.array(l_m_train),dtype=torch.float)
    exp_train = torch.tensor(np.array(exp_train),dtype=torch.float)
    y_train, y_tindex = exp_train[:, -1], exp_train[:, :-1]
    train_data = TensorDataset(g_e_train, l_m_train, y_train, y_tindex)

    if len(validindex) != 0:
        g_e_valid = Subset(graph_encoding, validindex)
        l_m_valid = Subset(laplacian_matrix, validindex)
        exp_valid = Subset(exp_cir, validindex)
        g_e_valid = torch.tensor(np.array(g_e_valid),dtype=torch.float)
        l_m_valid = torch.tensor(np.array(l_m_valid),dtype=torch.float)
        exp_valid = torch.tensor(np.array(exp_valid),dtype=torch.float)
        y_valid, y_vindex = exp_valid[:, -1], exp_valid[:, :-1]
        valid_data = TensorDataset(g_e_valid, l_m_valid, y_valid, y_vindex)
        valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True, drop_last=False)
    else:
        valid_loader=[]

    g_e_test = Subset(graph_encoding, testindex)
    l_m_test = Subset(laplacian_matrix, testindex)
    exp_test = Subset(exp_cir, testindex)
    g_e_test = torch.tensor(np.array(g_e_test),dtype=torch.float)
    l_m_test = torch.tensor(np.array(l_m_test),dtype=torch.float)
    exp_test = torch.tensor(np.array(exp_test),dtype=torch.float)
    y_test, y_index = exp_test[:, -1], exp_test[:, :-1]
    test_data = TensorDataset(g_e_test, l_m_test, y_test, y_index)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, drop_last=False)

    return train_data, valid_loader, test_loader, save_path

def modelEval(model,dataloader):

    predE = []
    trueE = []
    valid_losses = []
    model.eval()
    loss_func = torch.nn.MSELoss()
    for step, (g, l, b_y, b_index) in enumerate(dataloader):
        g, l= g.to(device), l.to(device)
        trueE.extend(np.c_[b_index, b_y])
        with torch.no_grad():
            p = model(g, l).detach().cpu().tolist()
            predE.extend(p)
            p = torch.tensor(p)
            loss = loss_func(p, b_y)
            valid_losses.append(loss.item())

    valid_loss = np.sqrt(np.mean(valid_losses))

    return trueE, predE, valid_loss

def saveExcel(save_csv,trainEpochLoss,trainPreExp,trainLoss,testPreExp,testLoss):
    trainEpochLoss = pd.DataFrame(trainEpochLoss, columns = ['Trainingloss'])
    trainPreExp = pd.DataFrame(trainPreExp,columns = ["qubit","gateNo","twoQubit","deepth","trueE","predE"])
    trainLoss = pd.DataFrame(np.array(trainLoss).reshape(1,10), columns = ["RMSE", "Huber", "Sperman", "Kendall", "R2"]*2)
    testPreExp = pd.DataFrame(testPreExp,columns = ["qubit","gateNo","twoQubit","deepth","trueE","predE"])
    testLoss = pd.DataFrame(np.array(testLoss).reshape(1,10), columns = ["RMSE", "Huber", "Sperman", "Kendall", "R2"]*2)
    with pd.ExcelWriter(save_csv,engine='xlsxwriter') as writer:
        trainEpochLoss.to_excel(writer, sheet_name = "trainEpochLoss")
        trainPreExp.to_excel(writer, sheet_name = "trainPreExp")
        trainLoss.to_excel(writer, sheet_name = " trainLoss")
        testPreExp.to_excel(writer, sheet_name = "testPreExp")
        testLoss.to_excel(writer, sheet_name = "testLoss")
        
def training(train_data, valid_loader, arg, seed):

    setup_seed(seed)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=False)
    predictor = TransformerPredictor(arg).to(device)
    optimizer = torch.optim.Adam(predictor.parameters(), lr=arg.learning_rate) # weight_decay=5e-4
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda epoch: 1/(epoch+1))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,  T_max=1000)

    loss_func = torch.nn.MSELoss()
    TotalTrainLoss = []
    valid_losses = []
    for j in range(arg.epoch):
        predictor.train()
        train_loss = []

        for step, (g, l, b_y, _) in enumerate(train_loader):
            optimizer.zero_grad()
            g, l, b_y = g.to(device), l.to(device), b_y.to(device)
            forward = predictor(g, l)
            loss = loss_func(forward, b_y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        print(f"Epoch {j}, Training Loss:{np.sqrt(np.mean(train_loss))}")
        TotalTrainLoss.append(np.mean(train_loss))

        if j==args.epoch-1:

            trueE, predE, _ = modelEval(predictor, train_loader)
            print('predicting complete')

            trainResult = np.c_[trueE, predE]
            trainL = analysisResult(trueE,predE)

            break

        scheduler.step()

    return predictor, TotalTrainLoss, trainResult, trainL, train_loader

def testing(predictor, test_loader, arg):

    trueE, predE, _ = modelEval(predictor, test_loader)
    
    testResult = np.c_[trueE, predE]
    testResult = testResult.tolist()

    testL = analysisResult(trueE,predE)
    
    return testResult, testL

def main(arg):


    train_data, valid_loader, test_loader, save_path = createData(args)
    result=[]
    for seed in range(0,15000,1000):

        start_time = datetime.datetime.now()
        nowTime = start_time.strftime('%Y%m')
        save_name = save_path + "hidden-" + str(arg.hidden_dim) + "-epoch-" + str(arg.epoch) +  "-seed-" + str(seed) + "-" + str(nowTime) + ".pth"
        
        predictor, TotalTrainLoss, trainResult, trainLoss, train_loader = training(train_data,valid_loader, arg, seed)
        testResult, testLoss = testing(predictor,test_loader,arg)

        save_dict = {"model": predictor,
                     "trainEpochLoss": TotalTrainLoss,
                     "trainPreExp": trainResult,
                     "trainLoss": trainLoss,
                     "testPreExp": testResult,
                     "testLoss": testLoss,
                     "testdata": test_loader
                     }
        if seed == 0:
            save_dict.update({"traindata": train_loader})
        torch.save(save_dict, save_name)
        
        save_csv = save_path + "result-hidden-" + str(arg.hidden_dim) + "-epoch-" + str(arg.epoch) + "-seed-" + str(seed) + ".xlsx"
        saveExcel(save_csv, TotalTrainLoss, trainResult, trainLoss, testResult, testLoss)
        
        print(f"Training Loss: {trainLoss}, Testing Loss: {testLoss}, Rank Correlation: {testLoss[2]} ")
        end_time = datetime.datetime.now()
        runing_time = (end_time - start_time).seconds
        print(runing_time)
        trainLoss.extend(testLoss)
        trainLoss.append(runing_time)
        result.append(trainLoss)
    
    return result

if __name__ == '__main__':
    
    # for expType in ["MMD","KL"]:  #
    #     for qubitInfo in ["4","5","6","Mix"]:  #
    #         args = task_config(qubitInfo)
    #         args.epoch = 100
    #         args.expressibility_type = expType
    #         args.KL_Rel = False
    #         args.noise = False
    #         args.qubitInfo = qubitInfo
    #         args.dropratio = 0.05
    #         # args.hidden_dim = 16
    #         args.batch_size = 64
    #         args.learning_rate = 0.001
    #         if args.noise:
    #             f_name = "-Noisy"
    #         else:
    #             f_name = ""
    #         if args.expressibility_type == "KL" and args.KL_Rel:
    #             r_name = "-R"
    #         else:
    #             r_name = ""
    #
    #         result = main(args)
    #
    #         nowTime = datetime.datetime.now().strftime('%Y%m%d')
    #         file0 = f"Result-head-{args.nhead}-layer-{args.num_layers}-hiddenDim-{args.hidden_dim}/"
    #         filename = file0 + f"Qubits-{qubitInfo}-hidden-{args.hidden_dim}-{args.expressibility_type}{f_name}{r_name}-{nowTime}.csv"
    #         np.savetxt(filename, result, delimiter = ",",header = 'RMSE,Huber,Spear,Kendall,R2,RMSE,Huber,Spear,Kendall,R2,RMSE,Huber,Spear,Kendall,R2,RMSE,Huber,Spear,Kendall,R2,time')


    for qubitInfo in ["Mix","4", "5", "6"]:
        args = task_config(qubitInfo)
        args.epoch = 100
        args.expressibility_type = "MMD"
        args.KL_Rel = False
        args.noise = True
        args.qubitInfo = qubitInfo
        args.dropratio = 0.05
        # args.hidden_dim = 16
        args.batch_size = 64
        args.learning_rate = 0.001
        if args.noise:
            f_name = "-Noisy"
        else:
            f_name = ""
        if args.expressibility_type == "KL" and args.KL_Rel:
            r_name = "-R"
        else:
            r_name = ""

        result = main(args)


        nowTime = datetime.datetime.now().strftime('%Y%m%d')
        file0 = f"Result-head-{args.nhead}-layer-{args.num_layers}-hiddenDim-{args.hidden_dim}/"
        filename = file0 + f"Qubits-{qubitInfo}-hidden-{args.hidden_dim}-{args.expressibility_type}{f_name}{r_name}-{nowTime}.csv"
        np.savetxt(filename, result, delimiter = ",",
                   header = 'RMSE,Huber,Spear,Kendall,R2,RMSE,Huber,Spear,Kendall,R2,RMSE,Huber,Spear,Kendall,R2,RMSE,Huber,Spear,Kendall,R2,time')
    
    # for qubitInfo in ["4", "5", "6", "Mix"]:
    #     args = task_config(qubitInfo)
    #     args.epoch = 100
    #     args.expressibility_type = "KL"
    #     args.KL_Rel = True
    #     args.noise = False
    #     args.qubitInfo = qubitInfo
    #     args.dropratio = 0.05
    #     # args.hidden_dim = 32
    #     args.batch_size = 64
    #     args.learning_rate = 0.001
    #     if args.noise:
    #         f_name = "-Noisy"
    #     else:
    #         f_name = ""
    #     if args.expressibility_type == "KL" and args.KL_Rel:
    #         r_name = "-R"
    #     else:
    #         r_name = ""
    #
    #     result = main(args)
    #
    #     nowTime = datetime.datetime.now().strftime('%Y%m%d')
    #     file0 = f"Result-head-{args.nhead}-layer-{args.num_layers}-hiddenDim-{args.hidden_dim}/"
    #     filename = file0 + f"Qubits-{qubitInfo}-hidden-{args.hidden_dim}-{args.expressibility_type}{f_name}{r_name}-{nowTime}.csv"
    #     np.savetxt(filename, result, delimiter = ",",
    #                header = 'RMSE,Huber,Spear,Kendall,R2,RMSE,Huber,Spear,Kendall,R2,RMSE,Huber,Spear,Kendall,R2,RMSE,Huber,Spear,Kendall,R2,time')
