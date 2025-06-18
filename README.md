# Flwr-nnUNet
Flower-nnUNet for distributed training of 3D Medical Image Segmentation.

## Quick start with the spleen dataset

1. Download and extract the Medical Segmentation Decathlon spleen set:

   ```bash
   mkdir dataset && cd dataset
   wget https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar
   tar -xvf Task09_Spleen.tar
   ```

2. Convert and preprocess with nnUNet (assuming dataset id 009):

   ```bash
   nnUNetv2_convert_MSD_dataset -i /path/to/Task09_Spleen
   nnUNetv2_plan_and_preprocess -d 009 --verify_dataset_integrity
   ```

3. Set environment variables and start the Flower server and clients:

   ```bash
   export PREPROCESSED_ROOT=/path/to/nnUNet_preprocessed
   export OUTPUT_ROOT=/tmp/nnunet_output
   export NUM_CLIENTS=2
   export NUM_ROUNDS=5
   ```

   Then run `flower-supernode` with `server_app.py:app` and the desired number
   of clients using `client_app.py:app`.
