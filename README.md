# Input reconstruction attacks

- Using Auto Encoder

- Currently takes in as input sanitized data from
        - gansan:
        - disparate impact remover : <https://github.com/IBM/AIF360/blob/master/aif360/algorithms/preprocessing/disparate_impact_remover.py>
        -  

- Loss used is L1 distance between autoencoder output and original image (for both test and train)


awk 'FNR > 1' *-A=0-No=1--E=10.csv >> ../0a_no1_e20.csv


