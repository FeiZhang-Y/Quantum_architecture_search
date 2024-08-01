## Description

This repository serves as a pipeline for training and evaluating Transformer models on expressibility estimation tasks.

### Usage

The configuration for each dataset/model can be found in the _config.py_ file.

The generated circuits with 4 qubit are provided in the folder _GeneratedCircuits/qubit-4/list_cir/_

Expressibility of the 4 qubit generated circuits can be found in the folders _qubit-4/Expressibility_KL_, _qubit-4/Expressibility_MMD_ and _qubit-4/Expressibility_MMD_Noisy_

Graph Encoding of the circuits with four expressibility measures can be found in the folders _Graph_KL_, _Graph_KL_R_，_Graph_MMD_，_Graph_MMD_Noisy_

To run the experiments and generate datasets, please refer to the commands below:

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
