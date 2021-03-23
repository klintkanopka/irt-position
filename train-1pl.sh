#!/bin/bash

python main.py -N big_batch_1 -i Rsim/data1.csv -m 1 -b 256 -c 1e-5 -e 100 -n 1000 -E 0.05 -M 0.05
python main.py -N big_batch_2 -i Rsim/data2.csv -m 1 -b 256 -c 1e-5 -e 100 -n 1000 -E 0.05 -M 0.05
python main.py -N big_batch_3 -i Rsim/data3.csv -m 1 -b 256 -c 1e-5 -e 100 -n 1000 -E 0.05 -M 0.05
python main.py -N bigger_batch_1 -i Rsim/data1.csv -m 1 -b 512 -c 1e-5 -e 100 -n 1000 -E 0.05 -M 0.05
python main.py -N bigger_batch_2 -i Rsim/data2.csv -m 1 -b 512 -c 1e-5 -e 100 -n 1000 -E 0.05 -M 0.05
python main.py -N bigger_batch_3 -i Rsim/data3.csv -m 1 -b 512 -c 1e-5 -e 100 -n 1000 -E 0.05 -M 0.05
