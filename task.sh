parallel --bar -a Experiments.txt -j 3 bash -c "{} > /dev/null && echo {} >> finished.txt"
