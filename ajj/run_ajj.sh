#!/bin/bash
date
export HOME="/afs/desy.de/user/y/yian/"
python3 -u /afs/desy.de/user/y/yian/cms/AJJ_coffea_DNN/ajj/ajj.py --year $1 --samples $2 --outfile $3 --nproc $4
date
#cp $3 /afs/desy.de/user/y/yian/cms/AJJ_coffea_DNN/ajj

