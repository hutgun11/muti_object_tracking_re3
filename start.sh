conda create -n myenv  python=3.6
source activate myenv
pip install tensorflow==1.5.0
pip install -r requirements.txt
conda activate myenv

cd demo
python batch_demo_gun.py