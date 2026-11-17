#!/bin/bash
# One-time setup on the Chameleon node. Run:  bash setup.sh
set -e
cd "$(dirname "$0")/.."
sudo apt-get update -qq && sudo apt-get install -y -qq python3-venv tmux > /dev/null
python3 -m venv ~/venv
source ~/venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt
python -c "import torch; print('PyTorch', torch.__version__, '| GPU available:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
echo "CPU cores: $(nproc) | free disk: $(df -h ~ | tail -1 | awk '{print $4}')"
if ! command -v gsutil > /dev/null; then
  curl -sSL https://sdk.cloud.google.com | bash -s -- --disable-prompts > /dev/null
  echo 'source ~/google-cloud-sdk/path.bash.inc' >> ~/.bashrc
fi
echo "SETUP DONE. Now run:  source ~/.bashrc && gcloud auth login --no-launch-browser"
