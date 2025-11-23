import ROOT
import numpy as np
import math
import ctypes
import matplotlib.pyplot as plt

# Constants
nPoints = 100
sigma_alpha2 = 0.1
minRange = 1e-3
maxRange = 1.0

NBINS = 20
edges = np.array([0., 0.2555, 0.511, 0.7665, 1.022, 1.2775, 1.533, 1.7885, 2.044, 2.2995, 
                  2.555, 2.8105, 3.066, 3.3215, 3.577, 3.8325, 4.088, 4.3435, 4.599, 4.8545, 5.11])

# First set of values
valuess = np.array([43, 41, 37, 20, 17, 4, 5, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
valuesb = np.array([12, 13, 16, 15, 8, 8, 12, 10, 9, 3, 4, 6, 3, 5, 3, 7, 8, 4, 7, 2])

# Second set of values
valuess1 = np.array([41, 13, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
valuesb2 = np.array([8, 5, 13, 10, 8, 5, 3, 7, 2, 1, 4, 0, 3, 0, 2, 0, 0, 0, 0, 0])

osc_limits =np.array([[0.00013392857142857303, 0.009953271028037385],
[0.0054910714285714285, 0.009906542056074767],
[0.015, 0.009626168224299065],
[0.02303571428571429, 0.009205607476635513],
[0.029464285714285714, 0.008738317757009346],
[0.033348214285714294, 0.008317757009345795],
[0.038571428571428576, 0.007710280373831776],
[0.045, 0.006682242990654206],
[0.04968750000000001, 0.005794392523364486],
[0.05343750000000001, 0.004859813084112149],
[0.05758928571428572, 0.0033177570093457925],
[0.05973214285714286, 0.002149532710280372],
[0.06080357142857143, 0.000841121495327106],
[0.060937500000000006, 0.00009345794392523477]])

# Global histograms
hOriginal_s = ROOT.TH1F("hOriginal_s", "Original Histogram", NBINS, edges)
hOriginal_b = ROOT.TH1F("hOriginal_b", "Original Histogram", NBINS, edges)

# Fill histograms for the first set of values
for i in range(1, NBINS + 1):
    hOriginal_b.SetBinContent(i, valuesb[i - 1])
    hOriginal_s.SetBinContent(i, valuess[i - 1])

# Second set of histograms
hOriginal_s1 = ROOT.TH1F("hOriginal_s1", "Second Histogram", NBINS, edges)
hOriginal_b2 = ROOT.TH1F("hOriginal_b2", "Second Histogram", NBINS, edges)

# Fill histograms for the second set of values
for i in range(1, NBINS + 1):
    hOriginal_b2.SetBinContent(i, valuesb2[i - 1])
    hOriginal_s1.SetBinContent(i, valuess1[i - 1])

# Function to map index to a logarithmic scale (inverted to fit 1 - value)
def logScale(index, maxIndex, minVal, maxVal):
    normalized = float(index) / maxIndex
    logVal = minVal * (maxVal / minVal) ** normalized
    return 1 - logVal

# Minuit fit function to compute delta x^2 for both scale factors
def fcn_combined(npar, gin, fval, par, iflag):
    a1 = par[0]
    a2 = par[1]

    deltax2_combined = 0
    pot = 100  # scaling factor
    
    # Access global histograms
    global hOriginal_s, hOriginal_b, hRescaleds
    global hOriginal_s1, hOriginal_b2, hRescaleds2

    # First calculation (with scale_factor = 2 * A2_22 - A2_11)
    deltax2_1 = 0
    for j in range(1, hOriginal_s.GetNbinsX() + 1):
        N_obs = (hRescaleds.GetBinContent(j) + hOriginal_b.GetBinContent(j)) * pot
        N_exp = ((hOriginal_s.GetBinContent(j) * (1 + a1)) + (hOriginal_b.GetBinContent(j) * (1 + a2))) * pot
        if N_obs > 0 and N_exp > 0:
            deltax2_1 += (N_exp - N_obs + (N_obs * math.log(N_obs / N_exp)))
    
    # Second calculation (with scale_factor2 = A2_22 * A2_22)
    deltax2_2 = 0
    for j in range(1, hOriginal_s1.GetNbinsX() + 1):
        N_obs = (hRescaleds2.GetBinContent(j) + hOriginal_b2.GetBinContent(j)) * pot
        N_exp = ((hOriginal_s1.GetBinContent(j) * (1 + a1)) + (hOriginal_b2.GetBinContent(j) * (1 + a2))) * pot
        if N_obs > 0 and N_exp > 0:
            deltax2_2 += (N_exp - N_obs + (N_obs * math.log(N_obs / N_exp)))

    # Combine the two delta x^2 values
    deltax2_combined = deltax2_1 + deltax2_2
    deltax2_combined = abs(2 * deltax2_combined)

    # Add nuisance parameters
    deltax2_combined += (a1 / sigma_alpha1) ** 2
    deltax2_combined += (a2 / sigma_alpha2) ** 2

    fval.value = deltax2_combined  # Store the combined result

# Minuit fit function for the first scale factor only
def fcn_first_scale_factor(npar, gin, fval, par, iflag):
    a1 = par[0]
    a2 = par[1]

    deltax2_1 = 0
    pot = 100  # scaling factor
    
    # Access global histograms
    global hOriginal_s, hOriginal_b, hRescaleds

    # First calculation (with scale_factor = 2 * A2_22 - A2_11)
    for j in range(1, hOriginal_s.GetNbinsX() + 1):
        N_obs = (hRescaleds.GetBinContent(j) + hOriginal_b.GetBinContent(j)) * pot
        N_exp = ((hOriginal_s.GetBinContent(j) * (1 + a1)) + (hOriginal_b.GetBinContent(j) * (1 + a2))) * pot
        if N_obs > 0 and N_exp > 0:
            deltax2_1 += (N_exp - N_obs + (N_obs * math.log(N_obs / N_exp)))

    deltax2_1 = abs(2 * deltax2_1)  # Only for first scale factor

    # Add nuisance parameters
    deltax2_1 += (a1 / sigma_alpha1) ** 2
    deltax2_1 += (a2 / sigma_alpha2) ** 2

    fval.value = deltax2_1  # Store the first scale factor result

# Array to hold the different sigma_alpha1 values
sigma_alpha1_values = [0.02, 0.05, 0.08]
colors = ['blue', 'red', 'black']  # Different colors for different sigma_alpha1

# Create the figure and axis for the superimposed plot
plt.figure(figsize=(8, 6))

# Loop over each value of sigma_alpha1
for idx, sigma_alpha1 in enumerate(sigma_alpha1_values):
    print(f"Running analysis for sigma_alpha1 = {sigma_alpha1}")
    
    # Initialize arrays to store A2_11, A2_22, and deltax2 values
    a2_11 = np.zeros((nPoints+1, nPoints+1))
    a2_22 = np.zeros((nPoints+1, nPoints+1))
    deltax2_values_combined = np.zeros((nPoints+1, nPoints+1))
    deltax2_values_first_scale = np.zeros((nPoints+1, nPoints+1))

    # Perform minimization for combined case
    for i in range(nPoints + 1):
        for j in range(nPoints + 1):
            A2_11 = logScale(i, nPoints, minRange, maxRange)
            A2_22 = logScale(j, nPoints, minRange, maxRange)

            scale_factor = 2 * A2_22 - A2_11
            scale_factor2 = A2_22

            # Scale the histograms
            hRescaleds = hOriginal_s.Clone("hRescaleds")
            hRescaleds.Scale(scale_factor)

            hRescaleds2 = hOriginal_s1.Clone("hRescaleds2")
            hRescaleds2.Scale(scale_factor2)

            # Set up Minuit for combined delta x²
            minuit = ROOT.TMinuit(2)  # Number of parameters: a1, a2
            minuit.SetFCN(fcn_combined)
            minuit.SetPrintLevel(-1)

            # Initialize with -0.98 for a1 and a2
            vstart = [-0.98, -0.98]
            step = [0.01, 0.01]

            minuit.DefineParameter(0, "a1", vstart[0], step[0], -1.0, 1.0)
            minuit.DefineParameter(1, "a2", vstart[1], step[1], -1.0, 1.0)

            minuit.Migrad()

            # Get fitted parameters
            a1 = ctypes.c_double(0)
            a2 = ctypes.c_double(0)
            err = ctypes.c_double(0)

            minuit.GetParameter(0, a1, err)
            minuit.GetParameter(1, a2, err)

            # Store the results from the combined minimization
            f_result = ctypes.c_double(0)
            fcn_combined(2, None, f_result, [a1.value, a2.value], 0)

            a2_11[i, j] = 1 - A2_11
            a2_22[i, j] = 1 - A2_22
            deltax2_values_combined[i, j] = f_result.value

    # Perform minimization for first scale factor only
    for i in range(nPoints + 1):
        for j in range(nPoints + 1):
            A2_11 = logScale(i, nPoints, minRange, maxRange)
            A2_22 = logScale(j, nPoints, minRange, maxRange)

            scale_factor = 2 * A2_22 - A2_11

            # Scale the histograms
            hRescaleds = hOriginal_s.Clone("hRescaleds")
            hRescaleds.Scale(scale_factor)

            # Set up Minuit for first scale factor delta x²
            minuit = ROOT.TMinuit(2)  # Number of parameters: a1, a2
            minuit.SetFCN(fcn_first_scale_factor)
            minuit.SetPrintLevel(-1)

            # Initialize with -0.98 for a1 and a2
            vstart = [-0.98, -0.98]
            step = [0.01, 0.01]

            minuit.DefineParameter(0, "a1", vstart[0], step[0], -1.0, 1.0)
            minuit.DefineParameter(1, "a2", vstart[1], step[1], -1.0, 1.0)

            minuit.Migrad()

            # Get fitted parameters
            a1 = ctypes.c_double(0)
            a2 = ctypes.c_double(0)
            err = ctypes.c_double(0)

            minuit.GetParameter(0, a1, err)
            minuit.GetParameter(1, a2, err)

            # Store the results from the first scale factor minimization
            f_result = ctypes.c_double(0)
            fcn_first_scale_factor(2, None, f_result, [a1.value, a2.value], 0)

            a2_11[i, j] = 1 - A2_11
            a2_22[i, j] = 1 - A2_22
            deltax2_values_first_scale[i, j] = f_result.value

    # Shift the delta x² values to plot contours
    min_deltax2_combined = np.min(deltax2_values_combined)
    min_deltax2_first_scale = np.min(deltax2_values_first_scale)

    deltax2_values_combined_shifted = deltax2_values_combined - min_deltax2_combined
    deltax2_values_first_scale_shifted = deltax2_values_first_scale - min_deltax2_first_scale

    # Plot the combined delta x² contour for the current sigma_alpha1
    cp1 = plt.contour(a2_11, a2_22, deltax2_values_combined_shifted, levels=[4.61], colors=colors[idx], linestyles='solid', linewidths=2.5)
    

###################################################################################################################

# Get the paths for the first (and only) contour level
    paths = cp1.allsegs[0]

# Concatenate all segments into one array of (x, y) pairs
    all_points = np.concatenate(paths)

# Find the point with minimum y (lowest a2_22)
    min_y_idx = np.argmin(all_points[:, 1])
    min_y_point = all_points[min_y_idx]

# Find the point with minimum x (lowest 1 - a2_11²)
    min_x_idx = np.argmin(all_points[:, 0])
    min_x_point = all_points[min_x_idx]

    print(f"Lowest y (a2_22): x = {min_y_point[0]:.4e}, y = {min_y_point[1]:.4e}")
    print(f"Lowest x (1 - a2_11²): x = {min_x_point[0]:.4e}, y = {min_x_point[1]:.4e}")



####################################################################################################################


    # Plot the delta x² for the first scale factor for the current sigma_alpha1
    #cp2 = plt.contour(a2_11, a2_22, deltax2_values_first_scale_shifted, levels=[4.61], colors=colors[idx], linestyles='dashed', linewidths=2.5)

# Plot the oscillation limits with dashed black lines
plt.plot(osc_limits[:, 0], osc_limits[:, 1], color='k', linestyle='dashed')

# Add a vertical line at a2_11 = 0.0407 to represent LEP limits
plt.axvline(x=0.0407, color='green', linestyle='--', label=r'LEP')
y_vals = np.unique(a2_22)
plt.fill_betweenx(y_vals, x1=1e-3, x2=0.0407, color='green', alpha=0.2)

# Add the text for neutrino oscillations at the specified location
plt.text(0.002, 0.011, r'neutrino oscillations', color='k', fontsize=14)

# Set axis labels, scaling, and title
plt.xscale("log")
plt.yscale("log")
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel(r'$1-a_{11}^2$', fontsize=20)
plt.ylabel(r'$1-a_{22}^2$', fontsize=20)
plt.title(r'DUNE ND: 3 yrs', fontsize=18)

# Add legends for different sigma_alpha1 values and the LEP limit
plt.legend([plt.Line2D([], [], color=c) for c in colors] + [plt.Line2D([], [], color='green', linestyle='--')],
           [f'sigma_alpha1 = {sigma_alpha1}' for sigma_alpha1 in sigma_alpha1_values] + [r'LEP'], loc='upper left')

# Set limits for the plot
plt.ylim(1e-3, 1)
plt.xlim(1e-3, 1)

plt.tight_layout()
plt.show()
