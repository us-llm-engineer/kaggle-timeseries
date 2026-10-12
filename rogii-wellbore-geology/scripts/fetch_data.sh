#!/usr/bin/env bash
# Download and unpack the competition data (requires the Kaggle CLI, an API token and accepted competition rules).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
kaggle competitions download -c rogii-wellbore-geology-prediction -p data
# sha256 of the archive used for the results in this folder:
echo "46fd84d5e7e16bc8d16bbfb38e95f06ac7fc154437d1c07203f80a8c6dd852fd  data/rogii-wellbore-geology-prediction.zip" | sha256sum -c - \
  || echo "warning: archive differs from the one used here"
unzip -q -o data/rogii-wellbore-geology-prediction.zip -d data/extracted
ls data/extracted
