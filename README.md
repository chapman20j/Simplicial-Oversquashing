# Simplicial-Oversquashing

Demystifying Topological Message-Passing with Relational Structures: A Case Study on Oversquashing in Simplicial Message-Passing

## Installation

Create the conda environment:

```bash
conda create -n simplicial python=3.10 
```

Install the requirements:

```bash
conda activate simplicial
pip install -r requirements.txt
```

Perform graph lifting as a preprocessing step:

```bash
python pretransform_datasets.py
```

## Experiments

To run graph classification experiments on TU datasets:

```bash
python classification.py --datasets MUTAG ENZYMES PROTEINS NCI1  --lifts none clique --rewiring none fosr afr4 sdrf
python classification.py --datasets IMDB-BINARY  --lifts none --rewiring none fosr afr4 sdrf
python classification.py --datasets IMDB-BINARY  --lifts clique --rewiring none fosr
```

To run graph lifting experiments on example graphs:

```bash
python graph_lift.py --curvature or --graph long_dumbbell
```

To preprocess TU datasets with graph lifting:

```bash
python pretransform_datasets.py
```

To run graph transfer experiments:

```bash
python transfer.py --dataset ringtransfer --model rgcn --exp size --lift both
python transfer.py --dataset ringtransfer --model rgcn --exp rewire --lift both
python transfer.py --dataset ringtransfer --model rgcn --exp hidden_dim --lift both
```

To compute graph statistics and weighted curvatures:

```bash
python weighted_curvature.py --dataset MUTAG --curvature or
```
