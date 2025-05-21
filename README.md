Label flipping attack in split-fed

https://github.com/chandra2thapa/SplitFed-When-Federated-Learning-Meets-Split-Learning.git



python main.py --experiment_name "split_fed_baseline_inner_epochs=5_epochs=50" --setup "split_fed" --attack "No Attack" --inner_epochs 5 --epochs 50

python main.py --experiment_name "fed_baseline_inner_epochs=5_epochs=50" --setup "fed" --attack "No Attack" --inner_epochs 5 --epochs 50

python main.py --experiment_name "split_baseline_inner_epochs=5_epochs=50" --setup "split" --attack "No Attack" --inner_epochs 5 --epochs 50

python main.py --experiment_name "split_fed_label_flipping_inner_epochs=5_epochs=50_PDR=1.0" --setup "split_fed" --attack "label_flipping 5->3,3->5" --inner_epochs 5 --epochs 50 --PDR 1.0 --label_flipping "bi" --scale 4

python main.py --experiment_name "split_fed_data_poisoning_inner_epochs=5_epochs=50_PDR=1.0" --setup "split_fed" --attack "data_poisoning 0.5,0.25" --inner_epochs 5 --epochs 50 --PDR 1.0 --scale 4
