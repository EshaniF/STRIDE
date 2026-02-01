

### Process data
First, unpack the data files 

For the three ICEWS datasets 'ICEWS18', 'ICEWS14', go into the dataset folder in the `./data` directory and run the following command to construct the static graph and the query historical subgraph.
```
cd ./data/
python get_his_subg.py
cd ./<dataset>
python ent2word.py
cd .. 

Then for all datasets, run get_his_subg.py file
python get_his_subg.py
```

### Train models
Then the following commands can be used to train the proposed models. By default, dev set evaluation results will be printed when training terminates.

cd src
```

ICEWS14
ppython main.py -d ICEWS14 --train-history-len 7 --test-history-len 7 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --gpu=0 --n-hidden 200 --self-loop --decoder convtranse 
--encoder uvrgcn --layer-norm --weight 0.5  --entity-prediction --angle 10 --discount 1 --pre-weight 0.9  --pre-type all --add-static-graph  --temperature 0.05 --run-statistic --n-epochs 80 --evaluate-every 1 --use-cl --cl_approach original

ICEWS18
python main.py -d ICEWS18 --train-history-len 7 --test-history-len 7 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --gpu=0 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5  --entity-prediction --angle 10 --discount 1 --pre-weight 0.9  --pre-type all --add-static-graph  --temperature 0.05 --run-statistic --n-epochs 70 --evaluate-every 1 --use-cl --cl_approach original

GDELT
python main.py -d GDELT --train-history-len 7 --test-history-len 7 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --gpu=0 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5  --entity-prediction --angle 10 --discount 1 --pre-weight 0.9  --pre-type all --add-static-graph  --temperature 0.03 --run-statistic --n-epochs 100 --evaluate-every 1 --use-cl --cl_approach original
```

*********************************************************************************************
Code partially adapted from authors' implementation of LogCL, RE-GCN, TIRGN models.
W. Chen, H. Wan, Y. Wu, S. Zhao, J. Cheng, Y. Li, and Y. Lin,
“Local-global history-aware contrastive learning for temporal knowledge
graph reasoning,” in 2024 IEEE 40th International Conference on Data
Engineering (ICDE). IEEE, 2024, pp. 733–746.

Z. Li, X. Jin, W. Li, S. Guan, J. Guo, H. Shen, Y. Wang, and
X. Cheng, “Temporal knowledge graph reasoning based on evolutional
representation learning,” in Proceedings of the 44th international ACM
SIGIR conference on research and development in information retrieval,
2021, pp. 408–417.

Y. Li, S. Sun, and J. Zhao, “Tirgn: Time-guided recurrent graph network
with local-global historical patterns for temporal knowledge graph
reasoning.” in IJCAI, 2022, pp. 2152–2158.

