@echo off

REM Create logs directory
if not exist logs mkdir logs

echo Running experiments...

REM ==================== RUN 1 ====================
set EXP=split_fed_label_flipping_5_to_9_inner_epochs=5_epochs=10_PDR=1.0_scale=1_mudhog
set LOG=logs\%EXP%.log

echo Running %EXP%
echo ===== START %DATE% %TIME% ===== > "%LOG%"
python main.py ^
 --experiment_name "%EXP%" ^
 --setup "split_fed" ^
 --attack "label_flipping 5->9" ^
 --inner_epochs 5 ^
 --epochs 10 ^
 --PDR 1.0 ^
 --scale 1 ^
 --alpha 0.5 ^
 --AR mudhog >> "%LOG%" 2>&1
echo ===== END %DATE% %TIME% ===== >> "%LOG%"


REM ==================== RUN 2 ====================
set EXP=split_fed_label_flipping_5_to_9_inner_epochs=5_epochs=10_PDR=1.0_scale=2_mudhog
set LOG=logs\%EXP%.log

echo Running %EXP%
echo ===== START %DATE% %TIME% ===== > "%LOG%"
python main.py ^
 --experiment_name "%EXP%" ^
 --setup "split_fed" ^
 --attack "label_flipping 5->9" ^
 --inner_epochs 5 ^
 --epochs 10 ^
 --PDR 1.0 ^
 --scale 2 ^
 --alpha 0.5 ^
 --AR mudhog >> "%LOG%" 2>&1
echo ===== END %DATE% %TIME% ===== >> "%LOG%"


REM ==================== RUN 3 ====================
set EXP=split_fed_label_flipping_5_to_9_inner_epochs=5_epochs=10_PDR=1.0_scale=3_mudhog
set LOG=logs\%EXP%.log

echo Running %EXP%
echo ===== START %DATE% %TIME% ===== > "%LOG%"
python main.py ^
 --experiment_name "%EXP%" ^
 --setup "split_fed" ^
 --attack "label_flipping 5->9" ^
 --inner_epochs 5 ^
 --epochs 10 ^
 --PDR 1.0 ^
 --scale 3 ^
 --alpha 0.5 ^
 --AR mudhog >> "%LOG%" 2>&1
echo ===== END %DATE% %TIME% ===== >> "%LOG%"


echo All experiments completed.
pause
