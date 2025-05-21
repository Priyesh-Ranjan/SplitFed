Label flipping attack in split-fed

https://github.com/chandra2thapa/SplitFed-When-Federated-Learning-Meets-Split-Learning.git

---

# For running the code

---

## No Attack Cases

For normal Split Learning (In split learning there is no attack) :

```
python main.py --experiment_name "split_fed_baseline_inner_epochs=5_epochs=50" --setup "split_fed" --attack "No Attack" --inner_epochs 5 --epochs 50
```

For normal Fed Learning :

```
python main.py --experiment_name "fed_baseline_inner_epochs=5_epochs=50" --setup "fed" --attack "No Attack" --inner_epochs 5 --epochs 50
```

For normal Split-Fed Learning :

```
python main.py --experiment_name "split_baseline_inner_epochs=5_epochs=50" --setup "split" --attack "No Attack" --inner_epochs 5 --epochs 50
```

---

## For Attack Cases

For Split-Fed and Fed Learning we can do these attacks:

- Label Flipping Attack: Currently it is flipping 5->9; changing the numbers will flip the corresponding labels

```
python main.py --experiment_name "split_fed_label_flipping_5_to_9_inner_epochs=5_epochs=50_PDR=1.0" --setup "split_fed" --attack "label_flipping 5->9" --inner_epochs 5 --epochs 50 --PDR 1.0 --scale 4
```
- Sign Flipping Attack: Multiplying all the elements in the models by -1; the loss will be very high for alternate rounds

```
python main.py --experiment_name "split_fed_sign_flipping_inner_epochs=5_epochs=50_PDR=1.0" --setup "split_fed" --attack "sign_flipping" --inner_epochs 5 --epochs 50 --PDR 1.0 --scale 4
```
- Random Flipping Attack: Randomly flip a label to any other label

```
python main.py --experiment_name "split_fed_random_flipping_inner_epochs=5_epochs=50_PDR=1.0" --setup "split_fed" --attack "random_flipping" --inner_epochs 5 --epochs 50 --PDR 1.0 --scale 4
```
- Data Poisoning Attack: Adding gaussian noise in data with mu,sigma

```
python main.py --experiment_name "split_fed_data_poisoning_0_0.25_inner_epochs=5_epochs=50_PDR=1.0" --setup "split_fed" --attack "data_poisoning 0,0.25" --inner_epochs 5 --epochs 50 --PDR 1.0 --scale 4
```
- Model Poisoning Attack: Adding gaussian noise in model with mu,sigma; keep mu=0 or loss will become nan

```
python main.py --experiment_name "split_fed_model_poisoning_0_0.1_inner_epochs=5_epochs=50_PDR=1.0" --setup "split_fed" --attack "data_poisoning 0,0.1" --inner_epochs 5 --epochs 50 --PDR 1.0 --scale 4
```

For all these cases PDR = 1.0
