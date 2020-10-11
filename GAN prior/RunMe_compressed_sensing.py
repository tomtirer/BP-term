# Compressed sensing using DCGAN prior and BP / LS loss functions

# Reference: "Back-Projection based Fidelity Term for Ill-Posed Linear Inverse Problems"
# Authors: Tom Tirer and Raja Giryes
# Journal: IEEE Transactions on Image Processing, 2020.

import numpy as np
import scipy.ndimage
from utils import *
from Generator_operations import *

ratio = 0.5 # CS m/n ratio
N = 64
M = 64
n = M*N
m = np.int(np.floor(ratio*n))
flag_apply_BP_post_processing = 0 # turn on to improve the results (e.g. see IAGAN paper https://arxiv.org/abs/1906.05284)

all_LS_PSNR_results = []
all_BP_PSNR_results = []

np.random.seed(0)
A = np.random.standard_normal((m,n)) / np.sqrt(m)
AAt_inv = np.linalg.inv(np.matmul(A,A.T))
A_dagger = np.matmul(A.T,AAt_inv)

Hfunc = lambda  Z: np.matmul(A,np.reshape(Z,(n,1)))
Htfunc = lambda  Z: np.reshape(np.matmul(A.T,np.reshape(Z,(m,1))),(M,N))
Hdagger_func = lambda  Z: np.reshape(np.matmul(A_dagger,np.reshape(Z,(m,1))),(M,N))


for image_ind in range(202587,202588):

    sample_file_number = str(image_ind)
    print("Image name:",sample_file_number)

    sample_file = "test_set\\" + sample_file_number + ".jpg"

    ###################################################################
    ### prepare observations
    ###################################################################

    sample = get_image(sample_file,
                  input_height=108,
                  input_width=108,
                  resize_height=64,
                  resize_width=64,
                  crop=True,
                  grayscale=False)
    X0 = (np.array(sample).astype(np.float32)) # for CS, intensity range [-1,1]

    [M,N] = X0.shape[0:2]
    sig_e = 0

    Y_clean = np.array([])
    for c in range(X0.shape[2]):
        Y_clean = np.dstack((Y_clean, Hfunc(X0[:,:,c]))) if Y_clean.size else Hfunc(X0[:,:,c])
    [Mlr,Nlr] = Y_clean.shape[0:2]

    np.random.seed(0)
    noise = sig_e * np.random.standard_normal(Y_clean.shape)
    Y = Y_clean + noise

    scipy.misc.imsave("results\\CS\\" + sample_file_number + "_X0.png", X0)
    Adagger_Y = np.array([]) # computing Adagger*Y to allow visualization of the measurements
    for c in range(Y.shape[2]):
        Adagger_Y_c = Hdagger_func(Y[:, :, c])
        Adagger_Y = np.dstack((Adagger_Y, Adagger_Y_c)) if Adagger_Y.size else Adagger_Y_c
    Adagger_Y = np.clip(Adagger_Y, -1, 1)
    scipy.misc.imsave("results\\CS\\" + sample_file_number + "_AdagY.png", (Adagger_Y+1)*255/2)


    ###################################################################
    # solve inverse problem with DCGAN prior and LS cost
    ###################################################################
    if 1:
        # for comparison: output also the z_init to use them also with the BP loss later
        [X_LS, z_init] = solve_inv_prob_with_GAN_prior(Y,Hfunc,Htfunc,np.array([]),'LS',flag_CS=1)

        X_LS_clip = np.clip(X_LS, -1, 1)
        PSNR = 10*np.log10(2**2/np.mean((X0-X_LS_clip)**2))
        print("X_LS PSNR:",PSNR)
        all_LS_PSNR_results.append(PSNR)
        scipy.misc.imsave("results\\CS\\" + sample_file_number + "_LS.png", (X_LS_clip+1)*255/2)

        if flag_apply_BP_post_processing:
            # perform also post-processing BP step to improve results
            X_LS_finalBP = np.array([])
            for c in range(Y.shape[2]):
                X_LS_finalBP_c = X_LS[:,:,c] + Hdagger_func( Y[:,:,c] - Hfunc(X_LS[:,:,c]) )
                X_LS_finalBP = np.dstack((X_LS_finalBP, X_LS_finalBP_c)) if X_LS_finalBP.size else X_LS_finalBP_c

            X_LS_finalBP_clip = np.clip(X_LS_finalBP, -1, 1)
            PSNR = 10*np.log10(2**2/np.mean((X0-X_LS_finalBP_clip)**2))
            print("X_LS_with_finalBP PSNR:",PSNR)
            scipy.misc.imsave("results\\CS\\" + sample_file_number + "_LS_with_ppBP.png", (X_LS_finalBP_clip+1)*255/2)


    ###################################################################
    # solve inverse problem with DCGAN prior and BP cost
    ###################################################################
    if 1:
        # for comparison: using the same z_init as the LS loss above
        [X_BP, z_init] = solve_inv_prob_with_GAN_prior(Y,Hfunc,Htfunc,z_init,'BP',Hdagger_func,flag_CS=1)

        X_BP_clip = np.clip(X_BP, -1, 1)
        PSNR = 10*np.log10(2**2/np.mean((X0-X_BP_clip)**2))
        print("X_BP PSNR:",PSNR)
        all_BP_PSNR_results.append(PSNR)
        scipy.misc.imsave("results\\CS\\" + sample_file_number + "_BP.png", (X_BP_clip+1)*255/2)

        if flag_apply_BP_post_processing:
            # perform also post-processing BP step to improve results
            X_BP_finalBP = np.array([])
            for c in range(Y.shape[2]):
                X_BP_finalBP_c = X_BP[:,:,c] + Hdagger_func( Y[:,:,c] - Hfunc(X_BP[:,:,c]) )
                X_BP_finalBP = np.dstack((X_BP_finalBP, X_BP_finalBP_c)) if X_BP_finalBP.size else X_BP_finalBP_c

            X_BP_finalBP_clip = np.clip(X_BP_finalBP, -1, 1)
            PSNR = 10*np.log10(2**2/np.mean((X0-X_BP_finalBP_clip)**2))
            print("X_BP_with_finalBP PSNR:",PSNR)
            scipy.misc.imsave("results\\CS\\" + sample_file_number + "_BP_with_ppBP.png", (X_BP_finalBP_clip+1)*255/2)


print("All PSNRs using LS term:")
print(all_LS_PSNR_results)
print("All PSNRs using BP term:")
print(all_BP_PSNR_results)



