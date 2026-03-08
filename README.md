Cluster: OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=2 python train.py

Notes:
- Training stopped after 19 epochs because the validation loss did not improve. Model trained for 9 epochs saved.

To copy data from diffsym:
cp -r /export/home/mpopescu/diffsym/data/preprocessed_parquet /export/home/mpopescu/Symbolic_Diffusion/data