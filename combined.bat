@echo off

REM Create logs directory
if not exist logs mkdir logs

echo Running experiments...

REM ==================== RUN 1 ====================
set EXP=split_fed_combined_10_5_inner_epochs=5_epochs=10_PDR=1.0_scale=3_fedavg
set LOG=logs\%EXP%.log

echo Running %EXP%
echo ===== START %DATE% %TIME% ===== > "%LOG%"
python main.py ^
 --experiment_name "%EXP%" ^
 --setup "split_fed" ^
 --attack "data_poisoning 10,5; model_poisoning 1,0.5; random_flipping" ^
 --inner_epochs 5 ^
 --epochs 10 ^
 --PDR 1.0 ^
 --scale 3 ^
 --alpha 0.5 ^
 --AR fedavg >> "%LOG%" 2>&1
echo ===== END %DATE% %TIME% ===== >> "%LOG%"


REM ==================== RUN 2 ====================
set EXP=split_fed_combined_10_5_inner_epochs=5_epochs=10_PDR=1.0_scale=6_fedavg
set LOG=logs\%EXP%.log

echo Running %EXP%
echo ===== START %DATE% %TIME% ===== > "%LOG%"
python main.py ^
 --experiment_name "%EXP%" ^
 --setup "split_fed" ^
 --attack "data_poisoning 10,5; model_poisoning 1,0.5; random_flipping" ^
 --inner_epochs 5 ^
 --epochs 10 ^
 --PDR 1.0 ^
 --scale 6 ^
 --alpha 0.5 ^
 --AR fedavg >> "%LOG%" 2>&1
echo ===== END %DATE% %TIME% ===== >> "%LOG%"


REM ==================== RUN 3 ====================
set EXP=split_fed_combined_10_5_inner_epochs=5_epochs=10_PDR=1.0_scale=9_fedavg
set LOG=logs\%EXP%.log

echo Running %EXP%
echo ===== START %DATE% %TIME% ===== > "%LOG%"
python main.py ^
 --experiment_name "%EXP%" ^
 --setup "split_fed" ^
 --attack "data_poisoning 10,5; model_poisoning 1,0.5; random_flipping" ^
 --inner_epochs 5 ^
 --epochs 10 ^
 --PDR 1.0 ^
 --scale 9 ^
 --alpha 0.5 ^
 --AR fedavg >> "%LOG%" 2>&1
echo ===== END %DATE% %TIME% ===== >> "%LOG%"


echo All experiments completed.
pause
