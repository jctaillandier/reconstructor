# Input reconstruction attacks

- Using Auto Encoder

- Currently implemented: VAE and autoencoder
        - Loss for AE is column wise Damage 
        - Loss for VAE is KL divergence + L1

- Test set meant to pass as one batch, and in last epoch, will save generated images in Experiment folder

- Input to model is Adult data without 'sex' and 'income'

- 'sex' column in output is the original, unchanged values
