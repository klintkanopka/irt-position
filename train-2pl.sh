#!/bin/bash

python main.py -i Rsim-2pl/data1.csv -m 2 -b 512 -c 5e-6 -e 500 -n 500 -E 0.05 -M 0.05
python main.py -i Rsim-2pl/data2.csv -m 2 -b 512 -c 5e-6 -e 500 -n 500 -E 0.05 -M 0.05
python main.py -i Rsim-2pl/data3.csv -m 2 -b 512 -c 5e-6 -e 500 -n 500 -E 0.05 -M 0.05
