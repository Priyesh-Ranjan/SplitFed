parallel --bar -a Plant_Fed.txt -j 3 bash -c "{} > /dev/null && echo {} >> finished.txt"
