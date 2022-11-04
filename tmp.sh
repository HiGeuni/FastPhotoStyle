#!/bin/bash
StartTime=$(date +%s)
python process_stylization_folder.py --fast --content_image=./content/ --theme autumn 2>/dev/null >./results/autumn.txt
python process_stylization_folder.py --fast --content_image=./content/ --theme night 2>/dev/null >./results/night.txt
python process_stylization_folder.py --fast --content_image=./content/ --theme snowy 2>/dev/null >./results/snowy.txt
python process_stylization_folder.py --fast --content_image=./content/ --theme sunset 2>/dev/null >./results/sunset.txt
EndTime=$(date +%s)
echo "It takes $(($EndTime - $StartTime)) seconds to complete this task."




