import tensorcircuit.interfaces.tensortrans
import tensorflow
import torch
import matplotlib.pyplot as plt
import utils
import numpy as np
import tensorcircuit as tc
from tqdm import tqdm
import tensorflow as tf
from scipy import stats
import argparse
from scipy.linalg import sqrtm
import os
import random

tc.set_backend("tensorflow")
tc.set_dtype("complex128")
current_path = os.path.join(os.path.abspath('.'),'GeneratedCircuits')
current_path = current_path.replace("\\","/")
# quantum_circuit defines whether consider noise
# output of quantum_circuit is input of fidelity_calculator
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
class ExpressibilityCalculator:

    def __init__(self,arg):
        self.pi_2 = 2 * np.pi
        self.num_random_initial = arg.num_random_initial
        self.qubits = arg.qubits
        self.seed = arg.seed
        self.parallel = arg.parallel
        self.noise = arg.noise
        self.haarPoints = arg.haarPoints
        self.gaussian_kernel_sigma = arg.gaussian_kernel_sigma
        self.expressibility_type = arg.expressibility_type
        self.two_qubit_channel_depolarizing_p = None
        self.single_qubit_channel_depolarizing_p = None
        self.bit_flip_p = None
        self.fidelity_calculation = arg.fidelity_calculation
        if self.noise:
            self.two_qubit_channel_depolarizing_p = arg.two_qubit_channel_depolarizing_p
            self.single_qubit_channel_depolarizing_p = arg.single_qubit_channel_depolarizing_p
            self.bit_flip_p = arg.bit_flip_p
            self.two_qubit_dep_channel = tc.channels.generaldepolarizingchannel(self.two_qubit_channel_depolarizing_p / 15, 2)
            tc.channels.kraus_identity_check(self.two_qubit_dep_channel)
            self.single_qubit_dep_channel = tc.channels.generaldepolarizingchannel(self.single_qubit_channel_depolarizing_p / 3, 1)
            tc.channels.kraus_identity_check(self.single_qubit_dep_channel)
            
    def quantum_circuit(self, structure, param):
        
        if isinstance(structure, str):
            structure = eval(structure)
            
        if self.noise:
            K0 = np.array([[1, 0], [0, 1]]) * np.sqrt(1 - self.bit_flip_p)
            K1 = np.array([[0, 1], [1, 0]]) * np.sqrt(self.bit_flip_p)
            
            c = tc.DMCircuit(self.qubits)
            if self.expressibility_type == "MMD":
                for i in range(self.qubits):
                    c.h(i)  
            for i, gate in enumerate(structure):
                if gate.name == "CZ":
                    c.cz(gate.act_on[0], gate.act_on[1])
                    c.general_kraus(self.two_qubit_dep_channel, gate.act_on[0], gate.act_on[1])
                elif gate.name == "U3":
                    c.u(gate.act_on[0], theta = param[i][0], phi = param[i][1], lbd = param[i][2])
                    c.general_kraus(self.single_qubit_dep_channel, gate.act_on[0])
                else:
                    print("invalid gate!")
                    exit(0)
            for q in range(self.qubits):
                c.general_kraus([K0, K1], q)
            state = c.state()
        else:
            c = tc.Circuit(self.qubits)
            if self.expressibility_type == "MMD":
                for i in range(self.qubits):
                    c.h(i)
            for i, gate in enumerate(structure):
                if gate.name == "CZ":
                    c.cz(gate.act_on[0], gate.act_on[1])
                elif gate.name == "U3":
                    c.u(gate.act_on[0], theta = param[i][0], phi = param[i][1], lbd = param[i][2])
                else:
                    print("invalid gate!")
                    exit(0)
            state = c.state()
        
        return state

    def Fu_distribution(self, circuit):

        setup_seed(self.seed)
        mapped_func = tc.backend.vmap(self.quantum_circuit, vectorized_argnums=(1,))
        para = np.random.uniform(-1, 1, self.num_random_initial * len(circuit) * 3)  
        para = para.reshape((self.num_random_initial, len(circuit), 3)) * self.pi_2
        para = tf.Variable(
            initial_value=tf.convert_to_tensor(para, dtype=getattr(tf, tc.rdtypestr))
        )
        try:
            output_states = mapped_func(circuit, para) 
            output_states0 = output_states.numpy()
        except:
            output_states0 = []
            for p in para:
                out = self.quantum_circuit(circuit,p)
                output_states0.append(out.numpy())
                
        return output_states0
    
    def fidelity_calculator_signal(self, circuit):
        
        output_states = self.Fu_distribution(circuit)
        if self.noise:
            output_states1 = output_states[0:int(self.num_random_initial / 2)]
            output_states2 = output_states[int(self.num_random_initial / 2):]
            fidelity = []
            for k in range(len(output_states1)):
                rho = output_states1[k]
                sigma = output_states2[k]
                fidelity0 = np.trace(sqrtm(sqrtm(rho) @ sigma @ sqrtm(rho)).astype(rho.dtype)).real
                fidelity.append(fidelity0)
        else:
            output_states1 = output_states[0:int(self.num_random_initial / 2)]
            output_states2 = output_states[int(self.num_random_initial / 2):]
            fidelity = (output_states1 * output_states2.conjugate()).sum(-1)  
            fidelity = np.power(np.absolute(fidelity), 2)  
            
        return fidelity
    
    def fidelity_calculator(self, circuits, gateNo, path1):
        
        if self.noise:
            f_name = "_Noisy"
        else:
            f_name = ""
        if path1 is None:
            path1 = f'{current_path}/qubit-{self.qubits}/Fidelity{f_name}/gate-number-{gateNo}.pkl'
        
        if self.fidelity_calculation:
            fidelities = []
            if self.parallel:
                pool = Pool(processes = 20)
                fidelities = pool.map(self.fidelity_calculator_signal, circuits)
                pool.close()
                pool.join()
            else:
                for cir in tqdm(circuits, desc = 'Computing fidelity'):
                    f = self.fidelity_calculator_signal(cir)
                    fidelities.append(f)
            # utils.save_pkl(fidelities, path1)
        else:
            fidelities = utils.load_pkl(path1)
        
        return fidelities
    
    def haar_calculator(self):
        N = self.qubits  
        points = self.haarPoints  
        space = 1 / points  
        x = [space * (i + 1) for i in range(-1, points)]  
        haar_points = []  
        for i in range(1, len(x)):  
            temp1 = -1 * np.power((1 - x[i]), np.power(2, N) - 1)
            temp0 = -1 * np.power((1 - x[i - 1]), np.power(2, N) - 1)
            haar_points.append(temp1 - temp0)
        haar_points = np.array(haar_points)  
        
        return haar_points
    
    def expressibility_KL(self, circuits, gateNo, path):
        
        fidelities = self.fidelity_calculator(circuits, gateNo, path1= 1) # path1= None
        haar_points = self.haar_calculator()
        
        expressivity = []
        points = self.haarPoints  
        for inner in tqdm(fidelities, desc = 'Computing expressivity'):
            bin_index = np.floor(inner * points).astype(int)
            num = []
            for i in range(0, points):
                num.append(len(bin_index[bin_index == i]))
            num = np.array(num) / sum(num)
            output = stats.entropy(num, haar_points)
            expressivity.append(output)
            np.savetxt(path+".csv", expressivity, delimiter = ",")
            utils.save_pkl(expressivity, path+".pkl")
        expressivity = np.array(expressivity)
        
        return expressivity
    
    def Fu_calculator(self,cir):
        output_states0 = self.Fu_distribution(cir)
        if self.noise:
            output_states = np.diagonal(output_states0, axis1 = 1, axis2 = 2)
        else:
            output_states = output_states0 * output_states0.conjugate()
        Fu = output_states.real
        return Fu
    
    def F_uniformSample(self):
        # np.random.seed(self.seed)
        setup_seed(self.seed)
        points = np.power(2,self.qubits)-1
        u0 = np.random.uniform(0,1,(self.num_random_initial,points))
        u1 = np.sort(u0,axis =1)
        s = np.insert(u1,0,np.zeros(self.num_random_initial),axis =1)
        s = np.insert(s,points+1,np.ones(self.num_random_initial),axis =1)
        F_uniform = np.diff(s)
        return F_uniform

    def MMD_claulator(self,Fu,F_uniform):
        sigma = self.gaussian_kernel_sigma
        RandomPoints = int(self.num_random_initial)
        
        Fu = torch.tensor(Fu[:, 0:-1], dtype = torch.float64)
        F_uniform = torch.tensor(F_uniform[:, 0:-1], dtype = torch.float64)
        
        xx = torch.cdist(Fu, Fu, p = 2)
        yy = torch.cdist(F_uniform, F_uniform, p = 2)
        xy = torch.cdist(Fu, F_uniform, p = 2)
        value_xx = torch.exp(- xx * xx / (4 * sigma))
        value_yy = torch.exp(- yy * yy / (4 * sigma))
        value_xy = torch.exp(- xy * xy / (4 * sigma))
        diff = (value_xx + value_yy - 2 * value_xy)
        
        Exp1 = torch.abs(torch.sum(diff)) / RandomPoints ** 2
        return Exp1.numpy()
    
    def expressibility_MMD(self, circuits,gateNo,path):
        
        F_uniform = self.F_uniformSample()
        MMD = []
        for cir in tqdm(circuits, desc = 'Computing expressivity'):
            Fu= self.Fu_calculator(cir)
            Exp1 = self.MMD_claulator(Fu,F_uniform)
            MMD.append(Exp1)
            
            np.savetxt(path+".csv", MMD, delimiter = ",")
            utils.save_pkl(MMD, path+".pkl")
            torch.cuda.empty_cache()
           
        return MMD

def main(args):
    
    fc = ExpressibilityCalculator(args)
    qubit = args.qubits
    if qubit == 4:
        gates = range(10,30)
    elif qubit == 5:
        gates = range(15,35)
    elif qubit == 6:
        gates = range(31,40)
        
    if args.noise:
        f_name = "_Noisy"
    else:
        f_name = ""
        
    expType = args.expressibility_type
    for gateNo in gates:
        path1 = f'{current_path}/qubit-{qubit}/list_cir/gate-number-{gateNo}.pkl'
        cir = utils.load_pkl(path1)
        path = f'{current_path}/qubit-{qubit}/Expressibility_{expType}{f_name}/gate-number-{gateNo}'
        if expType == "MMD":
            express = fc.expressibility_MMD(cir,gateNo,path)
        else:
            express = fc.expressibility_KL(cir,gateNo,path)

       # print(f"qubit: {qubit}, gate number: {gateNo}, successful!")
            
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--parallel", type=int, default=False, help="parallel processing")
    parser.add_argument("--num_random_initial", type=int, default=10000, help="number of random initial for fidelities calcualtion")
    parser.add_argument("--qubits", type=int, default=6, help="qubit")
    parser.add_argument("--noise", type=int, default=False, help="noise")
    parser.add_argument("--two_qubit_channel_depolarizing_p", type=float, default=0.01, help="two_qubit_noise")
    parser.add_argument("--single_qubit_channel_depolarizing_p", type=float, default=0.001, help="single_qubit_noise")
    parser.add_argument("--bit_flip_p", type=float, default=0.01, help="bit_flip_noise")
    parser.add_argument("--haarPoints", type=int, default=100, help="KL_haar Points")
    parser.add_argument("--gaussian_kernel_sigma", type=float, default=0.01, help="MMD_gaussian_kernel_sigma")
    parser.add_argument("--expressibility_type", type = str, default = "MMD", help = "MMD or KL")
    parser.add_argument("--fidelity_calculation", type = int, default = True, help = "whether to calculate fidelity")
    
    
    for qubit in range(6,7):
        parser.set_defaults(qubits = qubit)
        parser.set_defaults(noise = False)
        parser.set_defaults(expressibility_type = "MMD")
        parser.set_defaults(gaussian_kernel_sigma = 0.01)
        args = parser.parse_args()
        main(args)
        