# Step 0
Download official competition data from Kaggle here: https://www.kaggle.com/competitions/asl-fingerspelling/data
Download auxiliary files (like columns names, tokenizer, symmetric keypoints definition etc) from https://github.com/ChristofHenkel/kaggle-asl-fingerspelling-1st-place-solution/tree/main/datamount. 

# Step 1: Data preprocessing
Run preprocess_data.py, don't forget to change the paths in args. Alternatively, you can preprocess the data directly on Kaggle with this notebook https://www.kaggle.com/code/darraghdog/asl-fingerspelling-preprocessing-train, and download it. 

In this step we save each single ASL fingerspelling in it's own numpy file and also remove the keypoints that are not used. Competitor used a lot of keypoints, but we use only hands keypoints. 

You can uncomments values POSE, LIP, REYE, LEYE etc if you want to experiment with more keypoints.

After running this step, there should be directory with the structure like this: some_path/train_landmarks/101971546/1989551538.npy, where for each id there are many numpy files, and there around around 68 ids.

# Step 2: training autoencoder with RVQ 

Run `torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:23232 main.py`. Dont forget to change the data_folder path to point to processed dataset above. I worked with 2 GPUs but it can be trained on single GPU as well.

The encoder architecture is from https://github.com/ChristofHenkel/kaggle-asl-fingerspelling-1st-place-solution, I adjusted dimensions and fixed minor padding bug. The decoder is a symmetric version of it. 

