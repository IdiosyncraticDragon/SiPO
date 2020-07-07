mkdir running_records
mkdir logs
conda env create -f requirements.yaml -n SiPO-env
conda activate SiPO-env
pip install -i  https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
