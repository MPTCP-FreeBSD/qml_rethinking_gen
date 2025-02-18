#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import tensorcircuit as tc


def main(training_data):  
    def convolutional_layer(c, q1, q2, param):
        c.r(q1, theta=param[0], alpha=param[1], phi=param[2])
        c.r(q2, theta=param[3], alpha=param[4], phi=param[5])
        c.exp1(q1, q2, theta=param[6], unitary=tc.gates._zz_matrix)
        c.r(q1, theta=param[7], alpha=param[8], phi=param[9])
        c.r(q2, theta=param[10], alpha=param[11], phi=param[12])
        return c
    
    def pooling_layer(c, q1, q2, param):
        c.cr(q1, q2, theta=param[0], alpha=param[1], phi=param[2])
        return c
    
    def accuracy(params):
        accuracy_data = 0
        for j in range(len(gs_list)):
            c = tc.MPSCircuit(nqubits, tensors=gs_list[j])
            c.set_split_rules({"max_singular_values": 40})
            
            convolutional_layer(c, 0, 1, params[:13])
            convolutional_layer(c, 2, 3, params[:13])
            convolutional_layer(c, 4, 5, params[:13])
            convolutional_layer(c, 6, 7, params[:13])
            convolutional_layer(c, 8, 9, params[:13])
            convolutional_layer(c, 10, 11, params[:13])
            convolutional_layer(c, 12, 13, params[:13])
            convolutional_layer(c, 14, 15, params[:13])
            convolutional_layer(c, 16, 17, params[:13])
            convolutional_layer(c, 18, 19, params[:13])
            convolutional_layer(c, 20, 21, params[:13])
            convolutional_layer(c, 22, 23, params[:13])
            convolutional_layer(c, 24, 25, params[:13])
            convolutional_layer(c, 26, 27, params[:13])
            convolutional_layer(c, 28, 29, params[:13])
            convolutional_layer(c, 30, 31, params[:13])
            
            convolutional_layer(c, 1, 2, params[:13])
            convolutional_layer(c, 3, 4, params[:13])
            convolutional_layer(c, 5, 6, params[:13])
            convolutional_layer(c, 7, 8, params[:13])
            convolutional_layer(c, 9, 10, params[:13])
            convolutional_layer(c, 11, 12, params[:13])
            convolutional_layer(c, 13, 14, params[:13])
            convolutional_layer(c, 15, 16, params[:13])
            convolutional_layer(c, 17, 18, params[:13])
            convolutional_layer(c, 19, 20, params[:13])
            convolutional_layer(c, 21, 22, params[:13])
            convolutional_layer(c, 23, 24, params[:13])
            convolutional_layer(c, 25, 26, params[:13])
            convolutional_layer(c, 27, 28, params[:13])
            convolutional_layer(c, 29, 30, params[:13])
            convolutional_layer(c, 31, 0, params[:13])
            
            pooling_layer(c, 0, 1, params[13:16])
            pooling_layer(c, 2, 3, params[13:16])
            pooling_layer(c, 4, 5, params[13:16])
            pooling_layer(c, 6, 7, params[13:16])
            pooling_layer(c, 8, 9, params[13:16])
            pooling_layer(c, 10, 11, params[13:16])
            pooling_layer(c, 12, 13, params[13:16])
            pooling_layer(c, 14, 15, params[13:16])
            pooling_layer(c, 16, 17, params[13:16])
            pooling_layer(c, 18, 19, params[13:16])
            pooling_layer(c, 20, 21, params[13:16])
            pooling_layer(c, 22, 23, params[13:16])
            pooling_layer(c, 24, 25, params[13:16])
            pooling_layer(c, 26, 27, params[13:16])
            pooling_layer(c, 28, 29, params[13:16])
            pooling_layer(c, 30, 31, params[13:16])
            
            convolutional_layer(c, 1, 3, params[16:29])
            convolutional_layer(c, 5, 7, params[16:29])
            convolutional_layer(c, 9, 11, params[16:29])
            convolutional_layer(c, 13, 15, params[16:29])
            convolutional_layer(c, 17, 19, params[16:29])
            convolutional_layer(c, 21, 23, params[16:29])
            convolutional_layer(c, 25, 27, params[16:29])
            convolutional_layer(c, 29, 31, params[16:29])
            
            convolutional_layer(c, 3, 5, params[16:29])
            convolutional_layer(c, 7, 9, params[16:29])
            convolutional_layer(c, 11, 13, params[16:29])
            convolutional_layer(c, 15, 17, params[16:29])
            convolutional_layer(c, 19, 21, params[16:29])
            convolutional_layer(c, 23, 25, params[16:29])
            convolutional_layer(c, 27, 29, params[16:29])
            convolutional_layer(c, 31, 1, params[16:29])
            
            pooling_layer(c, 1, 3, params[29:32])
            pooling_layer(c, 5, 7, params[29:32])
            pooling_layer(c, 9, 11, params[29:32])
            pooling_layer(c, 13, 15, params[29:32])
            pooling_layer(c, 17, 19, params[29:32])
            pooling_layer(c, 21, 23, params[29:32])
            pooling_layer(c, 25, 27, params[29:32])
            pooling_layer(c, 29, 31, params[29:32])
            
            convolutional_layer(c, 3, 7, params[32:45])
            convolutional_layer(c, 11, 15, params[32:45])
            convolutional_layer(c, 19, 23, params[32:45])
            convolutional_layer(c, 27, 31, params[32:45])
            
            convolutional_layer(c, 7, 11, params[32:45])
            convolutional_layer(c, 15, 19, params[32:45])
            convolutional_layer(c, 23, 27, params[32:45])
            convolutional_layer(c, 31, 3, params[32:45])
            
            pooling_layer(c, 3, 7, params[45:48])
            pooling_layer(c, 11, 15, params[45:48])
            pooling_layer(c, 19, 23, params[45:48])
            pooling_layer(c, 27, 31, params[45:48])
            
            convolutional_layer(c, 7, 15, params[48:61])
            convolutional_layer(c, 23, 31, params[48:61])
            
            convolutional_layer(c, 15, 23, params[48:61])
            convolutional_layer(c, 31, 7, params[48:61])
            
            pooling_layer(c, 7, 15, params[61:64])
            pooling_layer(c, 23, 31, params[61:64])
            
            convolutional_layer(c, 15, 31, params[64:77])
            
            zz15 = np.real(c.expectation_ps(z=[15]))
            zz31 = np.real(c.expectation_ps(z=[31]))
            zz1531 = np.real(c.expectation_ps(z=[15, 31]))
            
            proj_00 = (1+zz1531+zz15+zz31)/4
            proj_01 = (1-zz1531-zz15+zz31)/4
            proj_10 = (1-zz1531+zz15-zz31)/4
            proj_11 = (1+zz1531-zz15-zz31)/4
            
            if label_list[j] == 0:
                if proj_00 < proj_01 and proj_00 < proj_10 and proj_00 < proj_11:
                    accuracy_data += 1
            elif label_list[j] == 1:
                if proj_01 < proj_00 and proj_01 < proj_10 and proj_01 < proj_11:
                    accuracy_data += 1
            elif label_list[j] == 2:
                if proj_10 < proj_00 and proj_10 < proj_01 and proj_10 < proj_11:
                    accuracy_data += 1
            else:
                if proj_11 < proj_00 and proj_11 < proj_01 and proj_11 < proj_10:
                    accuracy_data += 1
                

        return accuracy_data*100/len(gs_list)
        
    nqubits = 32
    best_params = np.loadtxt(f"{training_data}_training_data/BEST_PARAMS_j1j2_{training_data}")
    
    label_list = np.loadtxt(f"{training_data}_training_data/LABELS_{training_data}")    
    gs_list = np.load(f"{training_data}_training_data/train_groundstates.npy", allow_pickle=True)
        
    train_accuracy = accuracy(best_params)
    print(train_accuracy)



    train_error = 1-train_accuracy/100
    print("Train_Error:",train_error)
    print("Training Data:",training_data)
    print("Number of Qubits:",nqubits)
    print("-"*20)

    import os

    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Get the second-to-last folder name
    parent_dir = os.path.dirname(current_dir)
    second_last_folder = os.path.basename(parent_dir)

    # Define the "error_save" folder path within the second-to-last directory
    error_save_folder = os.path.join(parent_dir, "qml_error_save")

    # Check if the "error_save" folder exists, if not, create it
    if not os.path.exists(error_save_folder):
        os.makedirs(error_save_folder)

    # Construct the file path
    file_path = os.path.join(error_save_folder, f"{second_last_folder}_nqubits_{nqubits}_training_data_{training_data}_train_error.txt")

    # Save the error value to the .txt file
    np.savetxt(file_path, [train_error])
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data", default=5, type=int)
    args = parser.parse_args()
    for train_data in [5,8,10,14,20]:
        args.training_data = train_data
        main(**vars(args))