import argparse

def task_config(qubitInfo):
    
    parser = argparse.ArgumentParser()

    # training
    parser.add_argument("--training_set_rate", type = int, default = 0.8, help = "training set")
    parser.add_argument("--validation_set_rate", type=int, default = 0, help="validation set")
    parser.add_argument("--gate_type", type = str, default = ['U3','CZ'], help = "see task_congigs.py")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--epoch", type=int, default=100, help="epoch")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--hidden_dim", type=int, default=32, help="hidden dimendion")
    parser.add_argument("--nhead", type = int, default = 1, help="multi-head")
    parser.add_argument("--num_layers", type = int, default = 2, help="hidden layer")
    parser.add_argument("--dropout", type = int, default = False, help="whether to deep out")
    parser.add_argument("--dropratio", type = float, default = "0.05")
    
    # dataset
    parser.add_argument("--expressibility_type", type = str, default = "MMD", help = "MMD or KL")
    parser.add_argument("--noise", type = int, default = False, help = "noise")
    parser.add_argument("--qubitInfo", type = str, default = "Mix", help = "4, 5, 6, Mix")
    parser.add_argument("--qubit", type = int, default = 6, help = "see task_congigs.py")
    parser.add_argument("--gate_num", type = int, default = 39, help = "see task_congigs.py")
    
    parser.add_argument("--KL_Rel", type = int, default = False, help = "relative")
    parser.add_argument("--seed", type = int, default = 0, help = "seed")
    args = parser.parse_args()
    return args
