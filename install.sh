git clone git@github.com:pytorch/fairseq.git
cd fairseq
git checkout 7e75884
cp -frap ../knn_dta/* fairseq/fairseq/

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda install -c conda-forge rdkit
conda install -c pytorch faiss-gpu
pip install future scipy scikit-learn lifelines requests tensorboard tensorboardX

# rm -rf $pwd/fairseq/tasks/fairseq_task.py
# cp kNN-DTA/knn_dta/tasks/fairseq_task.py $pwd/fairseq/tasks/
# mv kNN-DTA/knn_dta/tasks/fairseq_task.py $pwd/fairseq/tasks/

pip install -e .

