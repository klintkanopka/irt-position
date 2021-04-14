#!/bin/bash

python main.py -i Rsim-2/data1_1pl.csv -m 1 -b 128 -c 1e-3 -e 500 -n 500 -E 0.1 -M 0.1
python main.py -i Rsim-2/data2_1pl.csv -m 1 -b 128 -c 1e-3 -e 500 -n 500 -E 0.1 -M 0.1
python main.py -i Rsim-2/data3_1pl.csv -m 1 -b 128 -c 1e-3 -e 500 -n 500 -E 0.1 -M 0.1
python main.py -i Rsim-2/data1.csv -m 2 -b 128 -c 1e-3 -e 500 -n 500 -E 0.1 -M 0.1
python main.py -i Rsim-2/data2.csv -m 2 -b 128 -c 1e-3 -e 500 -n 500 -E 0.1 -M 0.1
python main.py -i Rsim-2/data3.csv -m 2 -b 128 -c 1e-3 -e 500 -n 500 -E 0.1 -M 0.1
