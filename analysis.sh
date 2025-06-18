#!/bin/bash

echo "Copying analysis files for ML comparison..."

cp docker_ubuntu_24.04/ml_results_comparison/analysis_ml_comparison.md ./analysis_ml_docker.md
cp local_ubuntu_22.04/ml_results_comparison/analysis_ml_comparison.md ./analysis_ml_local.md
cp raspberry_pi_ubuntu/ml_results_comparison/analysis_ml_comparison.md ./analysis_ml_rasp.md

echo "Done!"