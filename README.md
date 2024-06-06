# Graph-similarity-baseline
Baseline methods of MLP and GNN in graph similarity learning.
Including Graph Transformer for graph similarity learning(SAT_GSL).
## Requirements
To install requirements:
```
pip install -r requirements.txt
```

## Training
Train the model by running the following command:
```
python main.py  --dataset <dataset> --model <model_name> --criterion <criterion>
```

\<dataset> can be among \{AIDS700nef, LINUX, IMDBMulti}.

\<model_name> can be among \{mlp_base, gnn_base, SAT_GSL}

\<criterion> can be among \{ged, mcs, eigenvalue, degree}
