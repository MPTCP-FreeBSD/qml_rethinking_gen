import argparse
import numpy as np
import cma
from qibo.symbols import Z, I
from qibo import hamiltonians
from qibo import models, gates
import qibo
qibo.set_backend("numpy")
import time
import os

def main(training_data):
    def MPO_3():
        symbolic_expr = Z(3)*I(7)
        hamiltonian = hamiltonians.SymbolicHamiltonian(form=symbolic_expr)
        return hamiltonian
        
    def MPO_7():
        symbolic_expr = Z(7)
        hamiltonian = hamiltonians.SymbolicHamiltonian(form=symbolic_expr)
        return hamiltonian
        
        
    def MPO_37():
        symbolic_expr = Z(3)*Z(7)
        hamiltonian = hamiltonians.SymbolicHamiltonian(form=symbolic_expr)
        return hamiltonian
    
    def convolutional_layer(c, q1, q2, param):
        c.add(gates.U3(q1, theta=param[0], phi=param[1], lam=param[2]))
        c.add(gates.U3(q1, theta=param[3], phi=param[4], lam=param[5]))
        c.add(gates.CNOT(q2, q1))
        c.add(gates.RZ(q1, theta=param[6]))
        c.add(gates.RY(q2, theta=param[7]))
        c.add(gates.CNOT(q1, q2))
        c.add(gates.RY(q2, theta=param[8]))
        c.add(gates.CNOT(q2, q1))
        c.add(gates.U3(q1, theta=param[9], phi=param[10], lam=param[11]))
        c.add(gates.U3(q1, theta=param[12], phi=param[13], lam=param[14]))
        return c
    
    def pooling_layer(c, q1, q2, param):
        c.add(gates.U3(q1, theta=param[0], phi=param[1], lam=param[2]))
        c.add(gates.U3(q1, theta=param[3], phi=param[4], lam=param[5]))
        c.add(gates.CNOT(q2, q1))
        c.add(gates.RZ(q1, theta=param[6]))
        c.add(gates.RY(q2, theta=param[7]))
        c.add(gates.CNOT(q1, q2))
        c.add(gates.RY(q2, theta=param[8]))
        c.add(gates.CNOT(q2, q1))
        c.add(gates.U3(q1, theta=param[9], phi=param[10], lam=param[11]))
        c.add(gates.U3(q1, theta=param[12], phi=param[13], lam=param[14]))
        return c
    
    def loss(params):
        cost = 0
        circuit = models.Circuit(nqubits)
        
        convolutional_layer(circuit, 0, 1, params[:15])
        convolutional_layer(circuit, 2, 3, params[:15])
        convolutional_layer(circuit, 4, 5, params[:15])
        convolutional_layer(circuit, 6, 7, params[:15])
        
        convolutional_layer(circuit, 1, 2, params[:15])
        convolutional_layer(circuit, 3, 4, params[:15])
        convolutional_layer(circuit, 5, 6, params[:15])
        convolutional_layer(circuit, 7, 0, params[:15])
        
        pooling_layer(circuit, 0, 1, params[15:30])
        pooling_layer(circuit, 2, 3, params[15:30])
        pooling_layer(circuit, 4, 5, params[15:30])
        pooling_layer(circuit, 6, 7, params[15:30])
        
        convolutional_layer(circuit, 1, 3, params[30:45])
        convolutional_layer(circuit, 5, 7, params[30:45])
        
        convolutional_layer(circuit, 3, 5, params[30:45])
        convolutional_layer(circuit, 7, 1, params[30:45])
        
        pooling_layer(circuit, 1, 3, params[45:60])
        pooling_layer(circuit, 5, 7, params[45:60])
        
        convolutional_layer(circuit, 3, 7, params[60:75])
        
        for j in range(training_data):
            final_state = circuit(gs_list[j]).state()

            z3 = np.real(ham3.expectation(final_state))
            z7 = np.real(ham7.expectation(final_state))
            zz37 = np.real(ham37.expectation(final_state))
            
            proj_00 = (1+zz37+z3+z7)/4
            proj_01 = (1-zz37-z3+z7)/4
            proj_10 = (1-zz37+z3-z7)/4
            proj_11 = (1+zz37-z3-z7)/4
            
            if label_list[j] == 0:
                cost += proj_00
            elif label_list[j] == 1:
                cost += proj_01
            elif label_list[j] == 2:
                cost += proj_10
            else:
                cost += proj_11
                
            # adding extra terms to equalize incorrect classes might help in the optimization, 
            # although it is not required
            """
            if label_list[j] == 0:
                cost += proj_00 + ((proj_01-proj_10)**2 + (proj_01-proj_11)**2 + (proj_10-proj_11)**2)/3
            elif label_list[j] == 1:
                cost += proj_01 + ((proj_00-proj_10)**2 + (proj_00-proj_11)**2 + (proj_10-proj_11)**2)/3
            elif label_list[j] == 2:
                cost += proj_10 + ((proj_00-proj_01)**2 + (proj_00-proj_11)**2 + (proj_01-proj_11)**2)/3
            else:
                cost += proj_11 + ((proj_00-proj_01)**2 + (proj_00-proj_10)**2 + (proj_01-proj_10)**2)/3
            """
                
            if count[0] % 500 == 0:
                accuracy_train = accuracy(params)
                accuracy_total.append(1 - accuracy_train/100)
                np.savetxt(f"ACCURACY_train_j1j2_{training_data}_500STEPS", [accuracy_total], newline='')

            count[0] += 1

        return cost/training_data
    
    def accuracy(params):
        accuracy_data = 0
        circuit = models.Circuit(nqubits)
        
        convolutional_layer(circuit, 0, 1, params[:15])
        convolutional_layer(circuit, 2, 3, params[:15])
        convolutional_layer(circuit, 4, 5, params[:15])
        convolutional_layer(circuit, 6, 7, params[:15])
        
        convolutional_layer(circuit, 1, 2, params[:15])
        convolutional_layer(circuit, 3, 4, params[:15])
        convolutional_layer(circuit, 5, 6, params[:15])
        convolutional_layer(circuit, 7, 0, params[:15])
        
        pooling_layer(circuit, 0, 1, params[15:30])
        pooling_layer(circuit, 2, 3, params[15:30])
        pooling_layer(circuit, 4, 5, params[15:30])
        pooling_layer(circuit, 6, 7, params[15:30])
        
        convolutional_layer(circuit, 1, 3, params[30:45])
        convolutional_layer(circuit, 5, 7, params[30:45])
        
        convolutional_layer(circuit, 3, 5, params[30:45])
        convolutional_layer(circuit, 7, 1, params[30:45])
        
        pooling_layer(circuit, 1, 3, params[45:60])
        pooling_layer(circuit, 5, 7, params[45:60])
        
        convolutional_layer(circuit, 3, 7, params[60:75])
        
        for j in range(training_data):
            final_state = circuit(gs_list[j]).state()

            z3 = np.real(ham3.expectation(final_state))
            z7 = np.real(ham7.expectation(final_state))
            zz37 = np.real(ham37.expectation(final_state))
            
            proj_00 = (1+zz37+z3+z7)/4
            proj_01 = (1-zz37-z3+z7)/4
            proj_10 = (1-zz37+z3-z7)/4
            proj_11 = (1+zz37-z3-z7)/4
            
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
                
        return accuracy_data*100/training_data
        
    
    nparams = 76
    nqubits = 8  
    
    ham7 = MPO_7()
    ham3 = MPO_3()
    ham37 = MPO_37()
    
    r_corrupt_values = [0, 2, 4, 6]
    run_duration = (30*60)/4  # 30 minutes



    for r_value in r_corrupt_values:

        # List to store train errors at each iteration
        accuracies = []
        time_series = []
        train_errors = []
        start_time = time.time()

        # Callback function to store the best train error
        def save_accuracy(es):
            """Callback function to save train error after each iteration."""
            acc = accuracy(es.result.xbest)
            accuracies.append(acc)  # Store best function value
            current_time = time.time()
            
            time_series.append(current_time - start_time)
            train_error = 1 - acc/100
            train_errors.append( train_error)
            print()
            print()
            print("+"*50)
            print(f"Iteration {es.countiter}: , Time: {current_time - start_time:.2f}s, Best Accuracy = {acc:.7f}%")
            print(f"Train Error: {train_error}")
            print()
            print()
            print("+"*50)
            #     # **Terminate if time exceeds 10 seconds**
            # if current_time - start_time > run_duration:
            #     print("Callback FUnction: Time limit exceeded. Stopping optimization.")
            #     return True  # Returning `True` stops the optimization
        # **Termination Callback Function**
        def termination_callback_fc(es):
            """Stops optimization if time exceeds 10 seconds."""
            if time.time() - start_time > run_duration:
                print(f"Termination_Callback: r_value:{r_value} Time limit exceeded. Stopping optimization.")
                return True  # Signals CMA-ES to stop
            return False  # Continue optimization
        
        start_time = time.time()

        print("+="*50)
        print(f"r_value: {r_value}")
        label_list = np.loadtxt(f'./{r_value}/LABELS_6_{r_value}')
        gs_list = np.load(f'./{r_value}/train_groundstates.npy', allow_pickle=True)

        acc = 0        
        while acc < 100 :       
            print(f"2122r_value: {r_value}")
            if time.time() - start_time > run_duration:
                print("Time Duration", time.time() - start_time)
                print(" While Loop: Time limit exceeded. Stopping optimization.")
                break
            accuracy_total = []
            count = [0]
                    
            initial_params = np.random.uniform(0, 2 * np.pi, nparams)
            xopt = cma.fmin2(loss, initial_params, 0.7, options={'tolfun': 0.5e-3,'termination_callback':termination_callback_fc},callback=save_accuracy)
            
            print(xopt[1].result.fbest)
            print(xopt[1].result.xbest)
            acc = accuracy(xopt[1].result.xbest)

            print()
            print(xopt)
            print()
            current_time = time.time()

            print()
            print("*()(*)"*20)
            print(f"Final Training time: {current_time - start_time}")
            print("*()(*)"*20)
            print()

        # Get the current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Get the second-to-last folder name
        parent_dir = os.path.dirname(current_dir)
        second_last_folder = os.path.basename(parent_dir)

        # Define the "error_save" folder path within the second-to-last directory
        error_save_folder = os.path.join(parent_dir, "training_error_graph")

        # Check if the "error_save" folder exists, if not, create it
        if not os.path.exists(error_save_folder):
            os.makedirs(error_save_folder)

        # Construct the file path
        file_path = os.path.join(error_save_folder, f"{second_last_folder}_nqubits_{nqubits}_training_data_{training_data}_r_corrupt_{r_value}_")

        # Save the error value to the .txt file
        print("Train Errors: ", train_errors)
        print("Time Series: ", time_series)
        np.savetxt(file_path+"train_error", [train_errors])
        np.savetxt(file_path+"time_series", [time_series])
        
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data", default=6, type=int)
    args = parser.parse_args()
    main(**vars(args))

