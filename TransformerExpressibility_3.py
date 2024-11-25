import os
import sys
import networkx as nx
import torch
import numpy as np
import utils
import argparse
from torch.utils.data import TensorDataset, DataLoader, Subset
import csv
import random
import datetime
from scipy import stats
from config import task_config
from sklearn.metrics import r2_score
import pandas as pd
from TransformerModel import TransformerPredictor

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:2'  # !(important)It makes rnn_based model deterministic
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
    trueE = torch.tensor(np.array(trueE))
    predE = torch.tensor(np.array(predE))
    RMSELoss = torch.sqrt(loss1(predE, trueE[:, -1])).tolist()
    spearmanr = stats.spearmanr(trueE[:, -1], predE)[0]
    kendall = stats.kendalltau(trueE[:, -1], predE)[0]
    r2 = r2_score(np.array(trueE[:, -1]), np.array(predE))
    result = [RMSELoss, spearmanr, kendall, r2]

    return result

def circuit_deepth(circuit):

    deepth = []
    No_two_qubit = []
    for i in range(0, len(circuit)):
        cir = np.array(circuit[i])
        res = [0] * len(np.unique(cir))  # 记录深度
        num_two_qubit_gates = 0
        max_depth = 0
        for j in cir:  # 遍历线路中每一个门
            if j[1] != j[2]:  # 是否双量子门
                num_two_qubit_gates += 1
                max_depth = max(res[j[1]], res[j[2]]) + 1
                res[j[1]], res[j[2]] = max_depth, max_depth
            else:
                res[j[1]] += 1
                max_depth = max(max_depth, res[j[1]])

        deepth.append(max_depth)
        No_two_qubit.append(num_two_qubit_gates)

    return No_two_qubit, deepth

def modelEval(model,dataloader):

    predE = []
    trueE = []
    valid_losses = []
    model.eval()
    loss_func = torch.nn.MSELoss()
    for step, (g, l, b_y, b_info, b_index) in enumerate(dataloader):
        g, l= g.to(device), l.to(device)
        trueE.extend(np.c_[b_index, b_info, b_y])
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
    trainPreExp = pd.DataFrame(trainPreExp,columns = ["cir_id","qubit","gateNo","twoQubit","deepth","trueE","predE"])
    trainLoss = pd.DataFrame(np.array(trainLoss).reshape(1,4), columns = ["RMSE",  "Sperman", "Kendall", "R2"])
    testPreExp = pd.DataFrame(testPreExp,columns = ["cir_id","qubit","gateNo","twoQubit","deepth","trueE","predE"])
    testLoss = pd.DataFrame(np.array(testLoss).reshape(1,4), columns = ["RMSE", "Sperman", "Kendall", "R2"])
    with pd.ExcelWriter(save_csv,engine='xlsxwriter') as writer:
        trainEpochLoss.to_excel(writer, sheet_name = "trainEpochLoss")
        trainPreExp.to_excel(writer, sheet_name = "trainPreExp")
        trainLoss.to_excel(writer, sheet_name = " trainLoss")
        testPreExp.to_excel(writer, sheet_name = "testPreExp")
        testLoss.to_excel(writer, sheet_name = "testLoss")
        
def training(train_data, valid_loader, arg, seed):

    setup_seed(seed)

    train_loader = DataLoader(train_data, batch_size=arg.batch_size, shuffle=True, drop_last=False)
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

        for step, (g, l, b_y, _, _) in enumerate(train_loader):
            optimizer.zero_grad()
            g, l, b_y = g.to(device), l.to(device), b_y.to(device)
            forward = predictor(g, l)
            loss = loss_func(forward, b_y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        print(f"Epoch {j}, Training Loss:{np.sqrt(np.mean(train_loss))}")
        TotalTrainLoss.append(np.mean(train_loss))

        if j==arg.epoch-1:

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

def main(args):
    
    file_name = f"GeneratedCircuits/TrainTest_{args.expressibility_type}.pt"
    data = torch.load(file_name)
    train_data = data['train']
    valid_loader = data['validloader']
    test_loader = data['testloader']
    
    file0 = f"PredictedResult/head-{args.nhead}-layer-{args.num_layers}-hiddenDim-{args.hidden_dim}/"
    save_path = file0 + f"Qubits-{args.qubitInfo}/Predict_{args.expressibility_type}/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    result=[]
    for seed in range(0,10000,1000):

        start_time = datetime.datetime.now()
        nowTime = start_time.strftime('%Y%m')
        save_name = save_path + "epoch-" + str(args.epoch) + "-seed-" + str(seed) + "-" + str(nowTime) + ".pth"
        
        predictor, TotalTrainLoss, trainResult, trainLoss, train_loader = training(train_data,valid_loader, args, seed)
        testResult, testLoss = testing(predictor,test_loader,args)

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
        
        save_csv = save_path + "result-hidden-" + str(args.hidden_dim) + "-epoch-" + str(args.epoch) + "-seed-" + str(seed) + ".xlsx"
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
    
    qubitInfo = "Mix"
    args = task_config(qubitInfo)
    args.epoch = 100
    args.qubitInfo = qubitInfo
    args.batch_size = 64
    args.learning_rate = 0.001
    
    for expType in ["KL", "KL_R", "MMD", "MMD_Noisy"]:  #
        
        args.expressibility_type = expType
        if expType == "KL":
            args.KL_Rel = False
            args.noise = False
            args.nhead = 2
            args.num_layers = 1
            args.hidden_dim = 32
        if expType == "KL_R":
            args.KL_Rel = True
            args.noise = False
            args.nhead = 2
            args.num_layers = 1
            args.hidden_dim = 16
        if expType == "MMD":
            args.KL_Rel = False
            args.noise = False
            args.nhead = 1
            args.num_layers = 2
            args.hidden_dim = 32
        if expType == "MMD_Noisy":
            args.KL_Rel = False
            args.noise = True
            args.nhead = 2
            args.num_layers = 2
            args.hidden_dim = 16
        
        result = main(args)
        
        nowTime = datetime.datetime.now().strftime('%Y%m%d')
        file0 = f"PredictedResult/Result-head-{args.nhead}-layer-{args.num_layers}-hiddenDim-{args.hidden_dim}-"
        filename = file0 + f"{args.expressibility_type}-{nowTime}.csv"
        np.savetxt(filename, result, delimiter = ",", header = 'RMSE,Spear,Kendall,R2,RMSE,Spear,Kendall,R2,time')
    
    for expType in ["KL", "KL_R", "MMD", "MMD_Noisy"]:  #
        
        args.expressibility_type = expType
        if expType == "KL":
            args.KL_Rel = False
            args.noise = False
            args.nhead = 1
            args.num_layers = 2
            args.hidden_dim = 16
        if expType == "KL_R":
            args.KL_Rel = True
            args.noise = False
            args.nhead = 2
            args.num_layers = 1
            args.hidden_dim = 32
        if expType == "MMD":
            args.KL_Rel = False
            args.noise = False
            args.nhead = 2
            args.num_layers = 2
            args.hidden_dim = 32
        if expType == "MMD_Noisy":
            args.KL_Rel = False
            args.noise = True
            args.nhead = 2
            args.num_layers = 2
            args.hidden_dim = 32
        
        result = main(args)
        
        nowTime = datetime.datetime.now().strftime('%Y%m%d')
        file0 = f"PredictedResult/Result-head-{args.nhead}-layer-{args.num_layers}-hiddenDim-{args.hidden_dim}-"
        filename = file0 + f"{args.expressibility_type}-{nowTime}.csv"
        np.savetxt(filename, result, delimiter = ",", header = 'RMSE,Spear,Kendall,R2,RMSE,Spear,Kendall,R2,time')
    
    for expType in ["KL", "KL_R", "MMD", "MMD_Noisy"]:  #
        
        args.expressibility_type = expType
        if expType == "KL":
            args.KL_Rel = False
            args.noise = False
            args.nhead = 2
            args.num_layers = 1
            args.hidden_dim = 16
        if expType == "KL_R":
            args.KL_Rel = True
            args.noise = False
            args.nhead = 1
            args.num_layers = 2
            args.hidden_dim = 16
        if expType == "MMD":
            args.KL_Rel = False
            args.noise = False
            args.nhead = 2
            args.num_layers = 1
            args.hidden_dim = 16
        if expType == "MMD_Noisy":
            args.KL_Rel = False
            args.noise = True
            args.nhead = 1
            args.num_layers = 2
            args.hidden_dim = 32
        
        result = main(args)
        
        nowTime = datetime.datetime.now().strftime('%Y%m%d')
        file0 = f"PredictedResult/Result-head-{args.nhead}-layer-{args.num_layers}-hiddenDim-{args.hidden_dim}-"
        filename = file0 + f"{args.expressibility_type}-{nowTime}.csv"
        np.savetxt(filename, result, delimiter = ",", header = 'RMSE,Spear,Kendall,R2,RMSE,Spear,Kendall,R2,time')
    
    for expType in ["KL", "KL_R", "MMD", "MMD_Noisy"]:  #
        
        args.expressibility_type = expType
        if expType == "KL":
            args.KL_Rel = False
            args.noise = False
            args.nhead = 2
            args.num_layers = 2
            args.hidden_dim = 16
        if expType == "KL_R":
            args.KL_Rel = True
            args.noise = False
            args.nhead = 1
            args.num_layers = 2
            args.hidden_dim = 32
        if expType == "MMD":
            args.KL_Rel = False
            args.noise = False
            args.nhead = 2
            args.num_layers = 2
            args.hidden_dim = 16
        if expType == "MMD_Noisy":
            args.KL_Rel = False
            args.noise = True
            args.nhead = 2
            args.num_layers = 1
            args.hidden_dim = 32
        
        result = main(args)
        
        nowTime = datetime.datetime.now().strftime('%Y%m%d')
        file0 = f"PredictedResult/Result-head-{args.nhead}-layer-{args.num_layers}-hiddenDim-{args.hidden_dim}-"
        filename = file0 + f"{args.expressibility_type}-{nowTime}.csv"
        np.savetxt(filename, result, delimiter = ",", header = 'RMSE,Spear,Kendall,R2,RMSE,Spear,Kendall,R2,time')
