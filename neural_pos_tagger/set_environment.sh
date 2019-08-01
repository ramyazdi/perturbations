#!/bin/bash
echo "Starting to set up Mood environment in 5 seconds..."
sleep 5
echo "Virtual Environment Setup"
mkdir -p ~/env
cd ~/env
python3 --version
pip3 --version
pip3 install --user virtualenv
virtualenv --system-site-packages mdmt
source ~/env/mdmt/bin/activate
echo "Install Python Packages"
pip3 --version
pip3 install --upgrade numpy scipy pandas six requests nltk h5py numexpr scikit-learn matplotlib lime
pip3 install --upgrade torch torchvision torchtext
pip3 install spacy textacy lime
pip3 install allennlp
echo "Conda Environment Setup"
cd ~/bin
wget https://repo.anaconda.com/archive/Anaconda3-5.3.1-Linux-x86_64.sh
bash Anaconda3-5.3.1-Ltinux-x86_64.sh
echo "alias 'conda_env' ='source /home/grotman@st.technion.ac.il/bin/anaconda3/etc/profile.d/conda.sh'" >> ~/.bash_profile
source ~/bin/anaconda3/etc/profile.d/conda.sh
conda create --name mdmt_pytorch
conda activate mdmt_pytorch
conda install pip numpy scipy pandas tabulate six requests beautifulsoup4 nltk h5py lxml numexpr scikit-learn matplotlib boto3
conda install spacy
conda install pytorch torchvision -c pytorch
pip install textacy lime
alias "mdmt_pytorch='conda_env && conda activate mdmt_pytorch && export PYTHONPATH=/home/grotman@st.technion.ac.il/Data/Projects/PycharmProjects/MDMT/:\$PYTHONPATH && cd Data/Projects/PycharmProjects/MDMT/'" >> ~/.bash_profile
source ~/.bash_profile
echo "Finished setting up mdmt environment!"
