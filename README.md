# Input reconstruction attacks

- Using Auto Encoder

<<<<<<< HEAD
- Currently takes in as input sanitized data from
        - gansan:
        - disparate impact remover : <https://github.com/IBM/AIF360/blob/master/aif360/algorithms/preprocessing/disparate_impact_remover.py>
        -  

- Loss used is L1 distance between autoencoder output and original image (for both test and train)


awk 'FNR > 1' *-A=0-No=1--E=10.csv >> ../0a_no1_e20.csv
=======
- Currently implemented 

- Loss used is L1 distance between autoencoder output and original image (for both test and train)

- If possible, test set will pass as one batch, and in last epoch, will save generated images in Experiment folder
>>>>>>> 57d843800f1265dfb95ab5c3e03d25fea6cb6521


