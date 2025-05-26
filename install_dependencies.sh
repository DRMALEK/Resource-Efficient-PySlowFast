#/bin/bash

# create virtual env
python -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install torch, torchvision
pip install torch torchvision torchaudio

# Install Detectron2
git clone https://github.com/facebookresearch/detectron2 detectron2_repo
pip install -e detectron2_repo


# Install PyTorchVideo
git clone https://github.com/facebookresearch/pytorchvideo.git
pip install -e pytorchvideo

# Upgrade pip to the latest version
pip install --upgrade pip

# Check if requirements.txt exists
if [ ! -f requirements.txt ]; then
    echo "requirements.txt not found. Please move to project directory"
    exit 1
fi

# Install dependencies for the project
pip install -r requirements.txt

# (Work directroy relative)
export PYTHONPATH=/workspace/Code/slowfast/slowfast:$PYTHONPATH

# Setup local branches
# First, fetch all remote branches
#git fetch --all

# List all remote branches to see what's available
#git branch -r

# Create local tracking branches for each remote branch
#git branch -a | grep remotes | grep -v HEAD | grep -v master | grep -v main | while read branch; do
#    git branch --track "${branch#remotes/origin/}" "$branch"
#done