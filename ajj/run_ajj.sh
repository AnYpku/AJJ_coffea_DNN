#!/bin/bash
date
export HOME="/afs/cern.ch/user/y/yian/"
python3 -u /afs/cern.ch/user/y/yian/work/DESY_pro/AJJ_coffea_DNN/ajj/ajj.py --year $1 --samples $2 --outfile $3 --nproc $4
date
cp $3 /eos/user/y/yian/AJJ_analysis/

