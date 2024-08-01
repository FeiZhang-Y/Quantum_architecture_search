## Description

This repository serves as a pipeline for training and evaluating Transformer models on expressibility estimation tasks.

### Usage

The config for each dataset/model can be found in file config.py.

The generated circuits with 4 qubit are provided in folder GeneratedCircuits/qubit-4/list_cir/

Expressibility of the 4 qubit generated circuits can be found in folder qubit-4/Expressibility_KL, qubit-4/Expressibility_MMD and qubit-4/Expressibility_MMD_Noisy

Graph Encoding of the circuits with four expressibility measures can be found in folder Graph_KL, Graph_KL_R，Graph_MMD，Graph_MMD_Noisy

If you want to run the experiments to generate datasets, please refer to the commands below :

```bash
# Circuit Generation
python ArchitectureGenerator_0.py
# Expressibility Calculation
python ExpressibilityCalculation_1.py
# Graph Encoding
python List2Graph_2.py
```
### Transformer

```bash
# Expressibilty Prediction by Transformer
python TransformerExpressibility_3.py
```
