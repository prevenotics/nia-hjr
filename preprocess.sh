python preprocessing/1_generate_mat.py --path '/workspace/dataset/1.원천데이터'


find “/workspace/datset/1.Train” -type f –regex '.*[LU].*\RA.mat$' >/workspace/dataset/1.Training/train_RA_for_class_mat.csv
find “/workspace/datset/2.Val” -type f –regex '.*[LU].*\RA.mat$' >/workspace/dataset/2.Val/val_RA_for_class_mat.csv
find “/workspace/datset/3.Test” -type f –regex '.*[LU].*\RA.mat$' >/workspace/dataset/test_RA.csv

find “/workspace/datset/1.Train” -type f –regex '.*[LU].*\RE.mat$' >/workspace/dataset/1.Training/train_RE_for_class_mat.csv
find “/workspace/datset/2.Val” -type f –regex '.*[LU].*\RE.mat$' >/workspace/dataset/2.Val/val_RE_for_class_mat.csv
find “/workspace/datset/3.Test” -type f –regex '.*[LU].*\RE.mat$' >/workspace/dataset/test_RE.csv

find “/workspace/datset/1.Train” -type f –regex '.*D.*\RA.mat$' >/workspace/dataset/1.Training/train_drone_RA_for_class_mat.csv
find “/workspace/datset/2.Val” -type f –regex '.*D.*\RA.mat$' >/workspace/dataset/2.Val/val_drone_RA_for_class_mat.csv
find “/workspace/datset/3.Test” -type f –regex '.*D.*\RA.mat$' >/workspace/dataset/test_drone_RA.csv

find “/workspace/datset/1.Train” -type f –regex '.*D.*\RE.mat$' >/workspace/dataset/1.Training/train_drone_RE_for_class_mat.csv
find “/workspace/datset/2.Val” -type f –regex '.*D.*\RE.mat$' >/workspace/dataset/2.Val/val_drone_RE_for_class_mat.csv
find “/workspace/datset/3.Test” -type f –regex '.*D.*\RE.mat$' >/workspace/dataset/test_drone_RE.csv

python preprocessing/2_generate_mat_for_train.py --path '/workspace/dataset/1.Training/3.mat' --train_val 'train'
python preprocessing/2_generate_mat_for_train.py --path '/workspace/dataset/2.Val/3.mat' --train_val 'val'













/root/work/hjr/dataset/3.mat_drone
find '/root/work/hjr/dataset/3.mat_drone' -type f -regex '.*D.*\RE.mat$' >/root/work/hjr/dataset/total_mat_RE_drone.txt
find '/root/work/hjr/dataset/3.mat_drone' -type f -regex '.*D.*\RA.mat$' >/root/work/hjr/dataset/total_mat_RA_drone.txt