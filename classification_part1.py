import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from init import import_dataset
from sklearn.metrics import mean_squared_error
import torch
import pandas as pd
import scipy.stats as st
from math import ceil
import scipy

def draw_neural_net(weights, biases, tf, 
                    attribute_names = None,
                    figsize=(9, 9),
                    fontsizes=(11, 9)):
    '''
    Draw a neural network diagram using matplotlib based on the network weights,
    biases, and used transfer-functions. 
    
    :usage:
        >>> w = [np.array([[10, -1], [-8, 3]]), np.array([[7], [-1]])]
        >>> b = [np.array([1.5, -8]), np.array([3])]
        >>> tf = ['linear','linear']
        >>> draw_neural_net(w, b, tf)
    
    :parameters:
        - weights: list of arrays
            List of arrays, each element in list is array of weights in the 
            layer, e.g. len(weights) == 2 with a single hidden layer and
            an output layer, and weights[0].shape == (2,3) if the input 
            layer is of size two (two input features), and there are 3 hidden
            units in the hidden layer.
        - biases: list of arrays
            Similar to weights, each array in the list defines the bias
            for the given layer, such that len(biases)==2 signifies a 
            single hidden layer, and biases[0].shape==(3,) signifies that
            there are three hidden units in that hidden layer, for which
            the array defines the biases of each hidden node.
        - tf: list of strings
            List of strings defining the utilized transfer-function for each 
            layer. For use with e.g. neurolab, determine these by:
                tf = [type(e).__name__ for e in transfer_functions],
            when the transfer_functions is the parameter supplied to 
            nl.net.newff, e.g.:
                [nl.trans.TanSig(), nl.trans.PureLin()]
        - (optional) figsize: tuple of int
            Tuple of two int designating the size of the figure, 
            default is (12, 12)
        - (optional) fontsizes: tuple of int
            Tuple of two ints giving the font sizes to use for node-names and
            for weight displays, default is (15, 12).
        
    Gist originally developed by @craffel and improved by @ljhuang2017
    [https://gist.github.com/craffel/2d727968c3aaebd10359]
    
    Modifications (Nov. 7, 2018):
        * adaption for use with 02450
        * display coefficient sign and magnitude as color and 
          linewidth, respectively
        * simplifications to how the method in the gist was called
        * added optinal input of figure and font sizes
        * the usage example how  implements a recreation of the Figure 1 in
          Exercise 8 of in the DTU Course 02450
    '''

   
   
    #Determine list of layer sizes, including input and output dimensionality
    # E.g. layer_sizes == [2,2,1] has 2 inputs, 2 hidden units in a single 
    # hidden layer, and 1 outout.
    layer_sizes = [e.shape[0] for e in weights] + [weights[-1].shape[1]]
    
    # Internal renaming to fit original example of figure.
    coefs_ = weights
    intercepts_ = biases

    # Setup canvas
    fig = plt.figure(figsize=figsize)
    ax = fig.gca(); ax.axis('off');

    # the center of the leftmost node(s), rightmost node(s), bottommost node(s),
    # topmost node(s) will be placed here:
    left, right, bottom, top = [.1, .9, .1, .9]
    
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    
    # Determine normalization for width of edges between nodes:
    largest_coef = np.max([np.max(np.abs(e)) for e in weights])
    min_line_width = 1
    max_line_width = 5
    
    # Input-Arrows
    layer_top_0 = v_spacing*(layer_sizes[0] - 1)/2. + (top + bottom)/2.
    for m in range(layer_sizes[0]):
        plt.arrow(left-0.18, layer_top_0 - m*v_spacing, 0.12, 0,  
                  lw =1, head_width=0.01, head_length=0.02)

    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), 
                                v_spacing/8.,
                                color='w', ec='k', zorder=4)
            if n == 0:
                if attribute_names:
                    node_str = str(attribute_names[m])
                    
                else:
                    node_str = r'$X_{'+str(m+1)+'}$'
                plt.text(left-0.125, layer_top - m*v_spacing+v_spacing*0.1, node_str,
                         fontsize=fontsizes[0])
            elif n == n_layers -1:
                node_str =  r'$y_{'+str(m+1)+'}$'
                plt.text(n*h_spacing + left+0.10, layer_top - m*v_spacing,
                         node_str, fontsize=fontsizes[0])
                if m==layer_size-1:
                    tf_str = 'Transfer-function: \n' + tf[n-1]
                    plt.text(n*h_spacing + left, bottom, tf_str, 
                             fontsize=fontsizes[0])
            else:
                node_str = r'$H_{'+str(m+1)+','+str(n)+'}$'
                plt.text(n*h_spacing + left+0.00, 
                         layer_top - m*v_spacing+ (v_spacing/8.+0.01*v_spacing),
                         node_str, fontsize=fontsizes[0])
                if m==layer_size-1:
                    tf_str = 'Transfer-function: \n' + tf[n-1]
                    plt.text(n*h_spacing + left, bottom, 
                             tf_str, fontsize=fontsizes[0])
            ax.add_artist(circle)
            
    # Bias-Nodes
    for n, layer_size in enumerate(layer_sizes):
        if n < n_layers -1:
            x_bias = (n+0.5)*h_spacing + left
            y_bias = top + 0.005
            circle = plt.Circle((x_bias, y_bias), v_spacing/8., 
                                color='w', ec='k', zorder=4)
            plt.text(x_bias-(v_spacing/8.+0.10*v_spacing+0.01), 
                     y_bias, r'$1$', fontsize=fontsizes[0])
            ax.add_artist(circle)   
            
    # Edges
    # Edges between nodes
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                colour = 'g' if coefs_[n][m, o]>0 else 'r'
                linewidth = (coefs_[n][m, o] / largest_coef)*max_line_width+min_line_width
                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], 
                                  c=colour, linewidth=linewidth)
                ax.add_artist(line)
                xm = (n*h_spacing + left)
                xo = ((n + 1)*h_spacing + left)
                ym = (layer_top_a - m*v_spacing)
                yo = (layer_top_b - o*v_spacing)
                rot_mo_rad = np.arctan((yo-ym)/(xo-xm))
                rot_mo_deg = rot_mo_rad*180./np.pi
                xm1 = xm + (v_spacing/8.+0.05)*np.cos(rot_mo_rad)
                if n == 0:
                    if yo > ym:
                        ym1 = ym + (v_spacing/8.+0.12)*np.sin(rot_mo_rad)
                    else:
                        ym1 = ym + (v_spacing/8.+0.05)*np.sin(rot_mo_rad)
                else:
                    if yo > ym:
                        ym1 = ym + (v_spacing/8.+0.12)*np.sin(rot_mo_rad)
                    else:
                        ym1 = ym + (v_spacing/8.+0.04)*np.sin(rot_mo_rad)
                plt.text( xm1, ym1,\
                         str(round(coefs_[n][m, o],4)),\
                         rotation = rot_mo_deg, \
                         fontsize = fontsizes[1])
                
    # Edges between bias and nodes
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        if n < n_layers-1:
            layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
            layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        x_bias = (n+0.5)*h_spacing + left
        y_bias = top + 0.005 
        for o in range(layer_size_b):
            colour = 'g' if intercepts_[n][o]>0 else 'r'
            linewidth = (intercepts_[n][o] / largest_coef)*max_line_width+min_line_width
            line = plt.Line2D([x_bias, (n + 1)*h_spacing + left],
                          [y_bias, layer_top_b - o*v_spacing], 
                          c=colour,
                          linewidth=linewidth)
            ax.add_artist(line)
            xo = ((n + 1)*h_spacing + left)
            yo = (layer_top_b - o*v_spacing)
            rot_bo_rad = np.arctan((yo-y_bias)/(xo-x_bias))
            rot_bo_deg = rot_bo_rad*180./np.pi
            xo2 = xo - (v_spacing/8.+0.01)*np.cos(rot_bo_rad)
            yo2 = yo - (v_spacing/8.+0.01)*np.sin(rot_bo_rad)
            xo1 = xo2 -0.05 *np.cos(rot_bo_rad)
            yo1 = yo2 -0.05 *np.sin(rot_bo_rad)
            plt.text( xo1, yo1,\
                 str(round(intercepts_[n][o],4)),\
                 rotation = rot_bo_deg, \
                 fontsize = fontsizes[1])    
                
    # Output-Arrows
    layer_top_0 = v_spacing*(layer_sizes[-1] - 1)/2. + (top + bottom)/2.
    for m in range(layer_sizes[-1]):
        plt.arrow(right+0.015, layer_top_0 - m*v_spacing, 0.16*h_spacing, 0,  lw =1, head_width=0.01, head_length=0.02)
        
    plt.show()

def train_neural_custom(model, loss_fn, X,y, num_epoch):

    net = model()

    optimizer = torch.optim.Adam(net.parameters(), lr=5e-3, weight_decay=0.05)
    best_final_loss = 1e100

    torch.nn.init.xavier_uniform_(net[0].weight)
    torch.nn.init.xavier_uniform_(net[2].weight)

    learning_curve = np.zeros(num_epoch)

    for epoch in range(1,num_epoch):
        # model.train()
        y_est = net(X) # forward pass, predict labels on training set
        loss = loss_fn(y_est.squeeze(), y) # determine loss
        loss_value = loss.data.numpy()
        learning_curve[epoch] = loss_value

        if loss_value < best_final_loss: 
            best_net = net
            best_final_loss = loss_value
            best_learning_curve = learning_curve

        optimizer.zero_grad(); loss.backward(); optimizer.step()

    print('\t\tBest Final loss:')
    print_str = '\t\t' + str(epoch+1) + '\t' + str(best_final_loss)
    print(print_str)
        
    # Return the best curve along with its final loss and learing curve
    return best_net, best_final_loss, best_learning_curve

#############################################
#                  Del 1                    #
#############################################

# Load data file and extract variables of interest

df = import_dataset(2016)

attributeNames = ['Economy', 'Family', 'Health', 'Freedom', 'Trust', 'Generosity', 'Dystopia Residual']

# Binary target value
y = df['Happy']
X = df[['Economy', 'Family', 'Health', 'Freedom', 'Trust', 'Generosity', 'Dystopia Residual']].copy()

# Standardize the training and test set based on training set mean and std

for idx in range(X.shape[1]):
    mu = np.mean(X.iloc[:,idx], 0)
    sigma = np.std(X.iloc[:,idx], 0)

    X.iloc[:,idx] = (X.iloc[:,idx] - mu) / sigma

#############################################
#                  Del 3                    #
#############################################

# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(K,shuffle=True)
CV2 = model_selection.KFold(K,shuffle=True)


#############################################
#       LINEAR MODEL PARAMETERS             #
# Here we determine the range for our regulisation term Lambda.
lambda_interval = np.logspace(-3, 2, 50)
overall_optimal_lambdas = {}
overall_test_errors_logistic = {}
#############################################

#############################################
#       NEURAL MODEL PARAMETERS             #
# Parameters for the Neural Network
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()
n_hidden_units_interval = range(1,10)
loss_fn = torch.nn.BCELoss()
num_epoch = 550
overall_test_errors_nn = {}
overall_hidden_layers = []
#############################################

#############################################
#       BASELINE MODEL PARAMETERS           #
# Here we determine the range for our regulisation term Lambda.
overall_test_errors_baseline = {}
#############################################

idx = 0 # Index value for first level cross validation

# We only use the first cross validation level to calculate the generalisation error based...
# ... on the optimal hyperparameters we found in the second cross validation level.
print("Starting...")

print("Initialising Pandas Dataframe for optimal Lambdas, Hidden layers and respective results")

results_df = pd.DataFrame(columns = ["Outer Fold", "\(h_i^*\)", "\(E_{i,h^*}^{test}\)","\(\lambda_i^*\)", "\(E_\{i,\lambda^*\}^{test}\)", "E_{i,base}^{test}"])

stats_df = pd.DataFrame( columns = ["Outer Fold", "ANN vs Baseline", "ANN vs Logistic", "Baseline vs Logistic"])

for train_index1, test_index1 in CV.split(X,y):
    idx += 1
    idx2 = 0 # Index value for second level cross validation

    X_train_first_level, y_train_first_level = X.iloc[train_index1,:], y[train_index1]
    X_test_first_level, y_test_first_level = X.iloc[test_index1,:], y[test_index1]

    X_train_first_level_torch = torch.Tensor(X_train_first_level.values, device=device )
    y_train_first_level_torch = torch.Tensor(y_train_first_level.values, device=device )
    X_test_first_level_torch = torch.Tensor(X_test_first_level.values, device=device )
    y_test_first_level_torch = torch.Tensor(y_test_first_level.values, device=device )

    first_level_opt_lambda = {}
    first_level_opt_hidden_layer = {}

    # Create Second Level Cross Level Validation to optimise hyperparameters
    # We Optimise the lambda and h (hidden layers) in this part of the crossvalidation
    # When we have found the optimal parameters we use those parameters on the first level cross validation
    # to find the optimal model.
    for train_index2, test_index2 in CV2.split(X_train_first_level,y_train_first_level):
        idx2 += 1

        print(f"Outer Level: {idx}, inner level Cross Validation split: {idx2}")

        X_train_second_level, y_train_second_level = X.iloc[train_index2,:], y[train_index2]
        X_test_second_level, y_test_second_level = X.iloc[test_index2,:], y[test_index2]

        # Doing the 2nd level partition of the train data.
        X_train_inner, X_test_inner, y_train_inner, y_test_inner = train_test_split(X_train_second_level, 
                                                                                    y_train_second_level, 
                                                                                    train_size=0.75)
        
        X_train_second_level_torch = torch.Tensor(X_train_inner.values, device=device )
        y_train_second_level_torch = torch.Tensor(y_train_inner.values, device=device )
        X_test_second_level_torch = torch.Tensor(X_test_inner.values, device=device )
        y_test_second_level_torch = torch.Tensor(y_test_inner.values, device=device )

        test_error_rate = np.zeros(len(lambda_interval))
        test_error_rate_nn = np.zeros(len(n_hidden_units_interval))

        # Loop over lambda_interval values
        for k in range(0, len(lambda_interval)):
            mdl = LogisticRegression(penalty='l2', C=1/lambda_interval[k] )

            mdl.fit(X_train_inner, y_train_inner)

            y_test_est = mdl.predict(X_test_inner).T
            
            test_error_rate[k] = np.sum(y_test_est != y_test_inner) / len(y_test_est)

        opt_lambda_idx = np.argmin(test_error_rate)
        opt_lambda = lambda_interval[opt_lambda_idx]
        first_level_opt_lambda[idx] = opt_lambda

        # Loop over n_hidden_units_interval values
        for j in range(0, len(n_hidden_units_interval)):
            nn_model_inner = lambda: torch.nn.Sequential(
                    torch.nn.Linear(X.shape[1], n_hidden_units_interval[j]), #M features to H hiden units
                    # 1st transfer function, either Tanh or ReLU:
                    torch.nn.Tanh(),                            #torch.nn.ReLU(),
                    torch.nn.Linear(n_hidden_units_interval[j], 1), # H hidden units to 1 output neuron
                    torch.nn.Sigmoid()
                    )

            # print('Training model of type:\n{}\n'.format(str(nn_model())))

            net, final_loss_inner, best_learning_curve_inner = train_neural_custom(nn_model_inner,
                                                            loss_fn,
                                                            X=X_train_second_level_torch,
                                                            y=y_train_second_level_torch,
                                                            num_epoch=num_epoch)
            # print('\n\tBest loss: {} for amount of hidden layers: {}\n'.format(final_loss_inner,n_hidden_units_interval[j]))
            
            # Determine estimated class labels for test set
            y_test_est_inner_nn = net(X_test_second_level_torch) # activation of final note, i.e. prediction of network
            y_test_est = (y_test_est_inner_nn > .5).type(dtype=torch.uint8) # threshold output of sigmoidal function
            y_test = y_test_est_inner_nn.type(dtype=torch.uint8)
            # Determine errors and error rate
            e = (y_test_est != y_test)
            error_rate_nn = (sum(e).type(torch.float)/len(y_test)).data.numpy()
            # We save the final loss for the model with this hidden level

            test_error_rate_nn[j] = error_rate_nn    

        opt_hidden_layer_idx = np.argmin(test_error_rate_nn)
        opt_hidden_layer = n_hidden_units_interval[opt_hidden_layer_idx]
        first_level_opt_hidden_layer[idx] = opt_hidden_layer

        print(f"For Outer split number: {idx}, Optimal hidden layer found to be: {opt_hidden_layer}")
        print(f"For Outer split number: {idx}, Optimal lambda found to be: {opt_lambda}")

    # NEURAL MODEL
    # Fit new models based on the original test split / partition by the K-fold.
    print(f"Training / Testing NN Model with hidden layers: {opt_hidden_layer}")
    # We will use the optimal lambda we just found to model the Sequence Neural Network Model
    overall_hidden_layers.append(opt_hidden_layer)
    nn_model_outer = lambda: torch.nn.Sequential(
            torch.nn.Linear(X.shape[1], n_hidden_units_interval[opt_hidden_layer_idx]), #M features to H hiden units
            # 1st transfer function, either Tanh or ReLU:
            torch.nn.Tanh(),                            #torch.nn.ReLU(),
            torch.nn.Linear(n_hidden_units_interval[opt_hidden_layer_idx], 1), # H hidden units to 1 output neuron
            torch.nn.Sigmoid()
            )

    net, final_loss_outer, best_learning_curve = train_neural_custom(nn_model_outer,
                                                    loss_fn,
                                                    X=X_train_first_level_torch,
                                                    y=y_train_first_level_torch,
                                                    num_epoch=num_epoch)
    print('\n\tBest loss: {} for outer level cross validation: {}\n'.format(final_loss_outer,idx))

    # Determine estimated class labels for test set
    y_test_est_outer_nn = net(X_test_first_level_torch) # activation of final note, i.e. prediction of network
    y_test_est = (y_test_est_outer_nn > .5).type(dtype=torch.uint8) # threshold output of sigmoidal function
    y_test = y_test_est_outer_nn.type(dtype=torch.uint8)
    # Determine errors and error rate
    e = (y_test_est != y_test)
    error_rate = (sum(e).type(torch.float)/len(y_test)).data.numpy()
    # We save the final loss for the model with this hidden level

    overall_test_errors_nn[idx] = error_rate

    # LOGISTIC MODEL
    # We will use the optimal lambda we just found to model the LogisticRegression Model
    opt_mdl_logistic = LogisticRegression(penalty='l2', C=1/opt_lambda )
    opt_mdl_logistic.fit(X_train_first_level, y_train_first_level)
    y_train_est_outer = opt_mdl_logistic.predict(X_train_first_level).T
    y_test_est_outer = opt_mdl_logistic.predict(X_test_first_level).T
    # Add the test_error_rate to the overall test_error_rate so we can calculate the generalisation error in the end.
    err_logistic = np.sum(y_test_est_outer != y_test_first_level) / len(y_test_est_outer)
    overall_test_errors_logistic[idx] = err_logistic

    # BASELINE MODEL
    y_pred = [1]*len(y_test_first_level)
    mse_baseline = np.sum(y_pred != y_test_first_level) / len(y_pred)
    overall_test_errors_baseline[idx] = mse_baseline

    results_df.loc[idx-1] = [idx] + [opt_hidden_layer] + [round(np.mean(error_rate_nn),3)] + [round(opt_lambda,3)] + [round(err_logistic,3)] + [round(mse_baseline,3)]

    r_nn_vs_baseline = np.mean(error_rate_nn-mse_baseline)
    r_nn_vs_logistic = np.mean(error_rate_nn-err_logistic)
    r_baseline_vs_logistic = np.mean(mse_baseline-err_logistic)


    stats_df.loc[idx-1] = [idx] + [round(r_nn_vs_baseline,3)] + [round(r_nn_vs_logistic,3)] + [round(r_baseline_vs_logistic,3)]

#############################################
#             COMPARE MODELS                #
#############################################

z_baseline = np.asarray(list(overall_test_errors_baseline.values()))
z_neural_net = np.asarray(list(overall_test_errors_nn.values()))
z_logistic = np.asarray(list(overall_test_errors_logistic.values()))
    
z1 = abs(z_neural_net - z_logistic)
z2 = abs(z_neural_net - z_baseline)
z3 = abs(z_logistic - z_baseline)

conf_z1 = st.t.interval(alpha=0.95, df=len(z1)-1, loc=np.mean(z1), scale=st.sem(z1))
conf_z2 = st.t.interval(alpha=0.95, df=len(z2)-1, loc=np.mean(z2), scale=st.sem(z2))
conf_z3 = st.t.interval(alpha=0.95, df=len(z3)-1, loc=np.mean(z3), scale=st.sem(z3))

print("="*25)
print(f"z1 confidence interval: {conf_z1}, p-value: {np.mean(2*st.t.cdf(-np.abs(np.mean(z1)) / st.sem(z1), df=len(z1) - 1))}")
print(f"z2 confidence interval: {conf_z2}, p-value: {np.mean(2*st.t.cdf(-np.abs(np.mean(z2)) / st.sem(z2), df=len(z2) - 1))}")
print(f"z3 confidence interval: {conf_z3}, p-value: {np.mean(2*st.t.cdf(-np.abs(np.mean(z3)) / st.sem(z3), df=len(z3) - 1))}")
print("="*25)
print()

color_list = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',
              'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']

print(results_df.to_latex(index=False))

# Display the MSE across folds NN
plt.bar(np.arange(1, K+1), np.squeeze(np.asarray(list(overall_test_errors_nn.values()))), color=color_list)
plt.xlabel('Fold')
plt.xticks(np.arange(1, K+1))
plt.ylabel('Mean Error Rate')
plt.title('Test Mean Error Rate for the Neural Net')

plt.show()

# Display the MSE across folds Logistic
plt.bar(np.arange(1, K+1), np.squeeze(np.asarray(list(overall_test_errors_logistic.values()))), color=color_list)
plt.xlabel('Fold')
plt.xticks(np.arange(1, K+1))
plt.ylabel('Test Error Rate')
plt.title('Test Test Error Rate for the Logistic model')

plt.show()

# Display the MSE across folds Baseline
plt.bar(np.arange(1, K+1), np.squeeze(np.asarray(list(overall_test_errors_baseline.values()))), color=color_list)
plt.xlabel('Fold')
plt.xticks(np.arange(1, K+1))
plt.ylabel('Test Error Rate')
plt.title('Test Test Error Rate for the Baseline model')

plt.show()

# Display Diagram
print('Diagram of best neural net in last fold:')
weights = [net[i].weight.data.numpy().T for i in [0,2]]
biases = [net[i].bias.data.numpy() for i in [0,2]]
tf =  [str(net[i]) for i in [1,2]]
draw_neural_net(weights, biases, tf, attribute_names=attributeNames)

# Display the learning curve for the best net in the current fold
plt.plot(best_learning_curve, color=color_list[0])
plt.xlabel('Iterations')
plt.xlim((0, num_epoch))
plt.ylabel('Loss')
plt.title('Learning curves')

plt.show()

print("-"*25 + " Ended " + "-"*25)

print("Overall Optimal Hidden Layers:")
print(overall_hidden_layers)
print()
print("Max Count Hidden Layer Amount:")
print(f"{max(overall_hidden_layers,key=overall_hidden_layers.count)}")
print()
mean_overall_error_logistic = sum(overall_test_errors_logistic.values()) / len(overall_test_errors_logistic.values())
mean_overall_test_errors_nn = sum(overall_test_errors_nn.values()) / len(overall_test_errors_nn.values())
mean_overall_error_baseline = sum(overall_test_errors_baseline.values()) / len(overall_test_errors_baseline.values())

print(f"Mean Overall Error (Generalisation Error) for LogisticRegression Model: {mean_overall_error_logistic}")
print(f"Mean Overall Error (Generalisation Error) for Neural Network Model: {np.mean(mean_overall_test_errors_nn)}")
print(f"Mean Overall Error (Generalisation Error) for Baseline Model: {mean_overall_error_baseline}")
print("-"*25 + " Ended " + "-"*25)

print("Stats Table:")
print(stats_df.to_latex(index=False))

conf_z1 = st.t.interval(alpha=0.95, df=len(stats_df["ANN vs Baseline"])-1, loc=np.mean(stats_df["ANN vs Baseline"]), scale=st.sem(stats_df["ANN vs Baseline"]))
conf_z2 = st.t.interval(alpha=0.95, df=len(stats_df["ANN vs Logistic"])-1, loc=np.mean(stats_df["ANN vs Logistic"]), scale=st.sem(stats_df["ANN vs Logistic"]))
conf_z3 = st.t.interval(alpha=0.95, df=len(stats_df["Baseline vs Logistic"])-1, loc=np.mean(stats_df["Baseline vs Logistic"]), scale=st.sem(stats_df["Baseline vs Logistic"]))

z1 = stats_df["ANN vs Baseline"]
z2 = stats_df["ANN vs Logistic"]
z3 = stats_df["Baseline vs Logistic"]

print("="*25)
print(f"z1 confidence interval: {conf_z1}, p-value: {np.mean(2*st.t.cdf(-np.abs(np.mean(z1)) / st.sem(z1), df=len(z1) - 1))}")
print(f"z2 confidence interval: {conf_z2}, p-value: {np.mean(2*st.t.cdf(-np.abs(np.mean(z2)) / st.sem(z2), df=len(z2) - 1))}")
print(f"z3 confidence interval: {conf_z3}, p-value: {np.mean(2*st.t.cdf(-np.abs(np.mean(z3)) / st.sem(z3), df=len(z3) - 1))}")
print("="*25)
print()

plt.show()
