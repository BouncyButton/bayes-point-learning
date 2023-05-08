#!/bin/bash

mkdir generated
filename="find-rs-100-ripper-100-inc-merged.pkl"
# Run a.py and save output to a.txt
python baseline_bp_comparison_3.py accuracy ../$filename >generated/baseline_bp_comparison_acc.tex
python baseline_bp_comparison_3.py f1 ../$filename >generated/baseline_bp_comparison_f1.tex
python baseline_bp_comparison_std.py accuracy ../$filename >generated/baseline_bp_comparison_std.tex
python bk_single_comparison.py accuracy ../$filename >generated/baseline_bestk_comparison_acc.tex
python bk_single_comparison.py f1 ../$filename >generated/baseline_bestk_comparison_f1.tex
python baselines_table.py accuracy ../$filename >generated/baselines_acc.tex
python baselines_table.py f1 ../$filename >generated/baselines_f1.tex
python bp_table.py accuracy ../$filename >generated/bp_acc.tex
python bp_table.py f1 ../$filename >generated/bp_f1.tex
# python complexity.py complexity > complexity.tex
# python complexity.py specificity > specificity.tex
