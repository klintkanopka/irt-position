#!/bin/bash

python main.py -i Rsim/data1_1pl.csv -m 1 -b 64 -c 5e-6 -e 500 -n 500 -E 0.05 -M 0.05
python main.py -i Rsim/data2_1pl.csv -m 1 -b 64 -c 5e-6 -e 500 -n 500 -E 0.05 -M 0.05
python main.py -i Rsim/data3_1pl.csv -m 1 -b 64 -c 5e-6 -e 500 -n 500 -E 0.05 -M 0.05
python main.py -i Rsim/data1.csv -m 2 -b 64 -c 5e-6 -e 500 -n 500 -E 0.05 -M 0.05
python main.py -i Rsim/data2.csv -m 2 -b 64 -c 5e-6 -e 500 -n 500 -E 0.05 -M 0.05
python main.py -i Rsim/data3.csv -m 2 -b 64 -c 5e-6 -e 500 -n 500 -E 0.05 -M 0.05
