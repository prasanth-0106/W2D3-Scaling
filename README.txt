WEEK 2 - DAY 3 | TASK 2: SCALING

Files:
- task2_scaling.py : complete solution
- requirements.txt: required Python packages
- expected_output.txt: sample output

Run in PowerShell from this folder:
    py -m pip install -r requirements.txt
    py task2_scaling.py

The assignment gives example KNN accuracies (0.93 and 0.97), not fixed
guaranteed values. Actual accuracy depends on the split and model settings.
This script uses test_size=0.2, random_state=42, stratify=y, and k=5.
All scalers used for model evaluation are fitted on training data only.
