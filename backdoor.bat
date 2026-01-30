@echo off

REM Create logs directory
if not exist logs mkdir logs

echo Running experiments...

REM ==================== RUN 1 ====================
set EXP=split_fed_backdoor_pls_to_14_inner_epochs=5_epochs=10_PDR=1.0_scale=1_fedavg
set LOG=logs\%EXP%.log

echo Running %EXP%
echo ===== START %DATE% %TIME% ===== > "%LOG%"
python main.py ^
 --experiment_name "%EXP%" ^
 --setup "split_fed" ^
 --attack "backdoor pls->14" ^
 --inner_epochs 5 ^
 --epochs 10 ^
 --PDR 1.0 ^
 --scale 1 ^
 --alpha 0.5 ^
 --AR fedavg >> "%LOG%" 2>&1
echo ===== END %DATE% %TIME% ===== >> "%LOG%"


REM ==================== RUN 2 ====================
set EXP=split_fed_backdoor_pls_to_14_inner_epochs=5_epochs=10_PDR=1.0_scale=2_fedavg
set LOG=logs\%EXP%.log

echo Running %EXP%
echo ===== START %DATE% %TIME% ===== > "%LOG%"
python main.py ^
 --experiment_name "%EXP%" ^
 --setup "split_fed" ^
 --attack "backdoor pls->14" ^
 --inner_epochs 5 ^
 --epochs 10 ^
 --PDR 1.0 ^
 --scale 2 ^
 --alpha 0.5 ^
 --AR fedavg >> "%LOG%" 2>&1
echo ===== END %DATE% %TIME% ===== >> "%LOG%"


REM ==================== RUN 3 ====================
set EXP=split_fed_backdoor_pls_to_14_inner_epochs=5_epochs=10_PDR=1.0_scale=3_fedavg
set LOG=logs\%EXP%.log

echo Running %EXP%
echo ===== START %DATE% %TIME% ===== > "%LOG%"
python main.py ^
 --experiment_name "%EXP%" ^
 --setup "split_fed" ^
 --attack "backdoor pls->14" ^
 --inner_epochs 5 ^
 --epochs 10 ^
 --PDR 1.0 ^
 --scale 3 ^
 --alpha 0.5 ^
 --AR fedavg >> "%LOG%" 2>&1
echo ===== END %DATE% %TIME% ===== >> "%LOG%"


echo All experiments completed.
pause
