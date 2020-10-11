# Super-resolution using DCGAN prior and BP / LS loss functions

# Reference: "Back-Projection based Fidelity Term for Ill-Posed Linear Inverse Problems"
# Authors: Tom Tirer and Raja Giryes
# Journal: IEEE Transactions on Image Processing, 2020.

import numpy as np
import scipy.ndimage
from utils import *
from Generator_operations import *
import scipy.linalg

s = 3 # SR scale factor
N = 64
M = 64
flag_apply_BP_post_processing = 0 # turn on to improve the results (e.g. see IAGAN paper https://arxiv.org/abs/1906.05284)

all_LS_PSNR_results = []
all_BP_PSNR_results = []

def matlab_style_gauss2D(shape=(7,7),sigma=1.6):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def create_SR_matrix_operation(filt_size,std,img_size,sr_factor):
    siz = np.int((filt_size-1.)/2.) # size should be odd
    M = img_size

    arg = -np.arange(-siz, siz + 1) ** 2 / (2 * std * std)
    h_1D = np.exp(arg)
    h_1D[h_1D < np.finfo(h_1D.dtype).eps * h_1D.max()] = 0
    h_1D = h_1D / h_1D.sum()
    filt_vec = np.concatenate((h_1D[siz:], np.zeros(M - (2 * siz + 1)), h_1D[0:siz]))
    filt_rows = scipy.linalg.toeplitz(filt_vec)
    filt_Mat = np.kron(filt_rows, filt_rows)
    samp_rows = np.eye(M, M)
    samp_rows = samp_rows[0::sr_factor, :]
    samp_Mat = np.kron(samp_rows, samp_rows)

    return np.matmul(samp_Mat,filt_Mat)


flag_matrix_implementation = 1

if flag_matrix_implementation:
    A = create_SR_matrix_operation(filt_size=7,std=1.6,img_size=M,sr_factor=s)
    AAt_inv = np.linalg.inv(np.matmul(A, A.T))
    A_dagger = np.matmul(A.T, AAt_inv)

    y = np.matmul(A,np.ones((M*N,1)))
    Mlr = np.int(np.sqrt(y.shape[0]))
    Nlr = Mlr

    Hfunc = lambda Z: np.reshape(np.matmul(A, np.reshape(Z, (M*N, 1))), (Mlr,Nlr))
    Htfunc = lambda Z: np.reshape(np.matmul(A.T, np.reshape(Z, (Mlr*Nlr, 1))), (M, N))
    Hdagger_func = lambda Z: np.reshape(np.matmul(A_dagger, np.reshape(Z, (Mlr*Nlr, 1))), (M, N))

else:
    h = matlab_style_gauss2D(shape=(7, 7), sigma=1.6)
    Hfunc = lambda  Z: downsample2(scipy.ndimage.convolve(Z, h, mode='nearest'),s)
    Htfunc = lambda  Z: scipy.ndimage.convolve(upsample2_MN(Z,s,M,N), np.fliplr(np.flipud(np.conj(h))), mode='nearest')
    Hdagger_func = None # will be applied using conjugate gradients


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
    X0 = (np.array(sample).astype(np.float32)+1)*255/2

    [M,N] = X0.shape[0:2]
    sig_e = 0

    Y_clean = np.array([])
    for c in range(X0.shape[2]):
        Y_clean = np.dstack((Y_clean, Hfunc(X0[:,:,c]))) if Y_clean.size else Hfunc(X0[:,:,c])
    [Mlr,Nlr] = Y_clean.shape[0:2]

    np.random.seed(0)
    noise = sig_e * np.random.standard_normal(Y_clean.shape)
    Y = Y_clean + noise

    scipy.misc.imsave("results\\SR\\" + sample_file_number + "_X0.png", X0)
    scipy.misc.imsave("results\\SR\\" + sample_file_number + "_Y.png", Y)


    ###################################################################
    # solve inverse problem with DCGAN prior and LS cost
    ###################################################################
    if 1:
        # for comparison: output also the z_init to use them also with the BP loss later
        [X_LS, z_init] = solve_inv_prob_with_GAN_prior(Y,Hfunc,Htfunc,np.array([]),'LS')

        X_LS_clip = np.clip(X_LS, 0, 255)
        PSNR = 10*np.log10(255**2/np.mean((X0-X_LS_clip)**2))
        print("X_LS PSNR:",PSNR)
        all_LS_PSNR_results.append(PSNR)
        scipy.misc.imsave("results\\SR\\" + sample_file_number + "_LS.png", X_LS_clip)

        if flag_apply_BP_post_processing:
            # perform also post-processing BP step to improve results
            if flag_matrix_implementation:
                X_LS_finalBP = np.array([])
                for c in range(Y.shape[2]):
                    X_LS_finalBP_c = X_LS[:,:,c] + Hdagger_func( Y[:,:,c] - Hfunc(X_LS[:,:,c]) )
                    X_LS_finalBP = np.dstack((X_LS_finalBP, X_LS_finalBP_c)) if X_LS_finalBP.size else X_LS_finalBP_c
            else:
                HHt_cg = lambda z: np.reshape(Hfunc(Htfunc(np.reshape(z,(Mlr,Nlr)))),(Mlr*Nlr,1))+z*0
                X_LS_finalBP = np.array([])
                for c in range(Y.shape[2]):
                    temp_c = Y[:,:,c] - Hfunc(X_LS[:,:,c])
                    [cg_result, iter, residual] = cg(np.zeros((Mlr*Nlr,1)), HHt_cg, np.reshape(temp_c,(Mlr*Nlr,1)), 100, 10**-6) # cg_result = inv(H*Ht)*(Y - H*X_LS)
                    if np.sqrt(residual)>10**-1:
                        print("cg: finished after ", iter, " iterations with norm(residual) = ", np.sqrt(residual)) # if the results are not good consider using preconditioning or tikho regularization (epsilon) for HHt_cg
                    X_LS_finalBP_c = X_LS[:,:,c] + Htfunc(np.reshape(cg_result,(Mlr,Nlr)))
                    X_LS_finalBP = np.dstack((X_LS_finalBP, X_LS_finalBP_c)) if X_LS_finalBP.size else X_LS_finalBP_c

            X_LS_finalBP_clip = np.clip(X_LS_finalBP, 0, 255)
            PSNR = 10*np.log10(255**2/np.mean((X0-X_LS_finalBP_clip)**2))
            print("X_LS_with_finalBP PSNR:",PSNR)
            scipy.misc.imsave("results\\SR\\" + sample_file_number + "_LS_with_ppBP.png", X_LS_finalBP_clip)


    ###################################################################
    # solve inverse problem with DCGAN prior and BP cost
    ###################################################################
    if 1:
        # for comparison: using the same z_init as the LS loss above
        [X_BP, z_init] = solve_inv_prob_with_GAN_prior(Y,Hfunc,Htfunc,z_init,'BP',Hdagger_func)

        X_BP_clip = np.clip(X_BP, 0, 255)
        PSNR = 10*np.log10(255**2/np.mean((X0-X_BP_clip)**2))
        print("X_BP PSNR:",PSNR)
        all_BP_PSNR_results.append(PSNR)
        scipy.misc.imsave("results\\SR\\" + sample_file_number + "_BP.png", X_BP_clip)

        if flag_apply_BP_post_processing:
            # perform also post-processing BP step to improve results
            if flag_matrix_implementation:
                X_BP_finalBP = np.array([])
                for c in range(Y.shape[2]):
                    X_BP_finalBP_c = X_BP[:,:,c] + Hdagger_func( Y[:,:,c] - Hfunc(X_BP[:,:,c]) )
                    X_BP_finalBP = np.dstack((X_BP_finalBP, X_BP_finalBP_c)) if X_BP_finalBP.size else X_BP_finalBP_c
            else:
                HHt_cg = lambda z: np.reshape(Hfunc(Htfunc(np.reshape(z,(Mlr,Nlr)))),(Mlr*Nlr,1))+z*0
                X_BP_finalBP = np.array([])
                for c in range(Y.shape[2]):
                    temp_c = Y[:,:,c] - Hfunc(X_BP[:,:,c])
                    [cg_result, iter, residual] = cg(np.zeros((Mlr*Nlr,1)), HHt_cg, np.reshape(temp_c,(Mlr*Nlr,1)), 100, 10**-6) # cg_result = inv(H*Ht)*(Y - H*X_BP)
                    if np.sqrt(residual)>10**-1:
                        print("cg: finished after ", iter, " iterations with norm(residual) = ", np.sqrt(residual)) # if the results are not good consider using preconditioning or tikho regularization (epsilon) for HHt_cg
                    X_BP_finalBP_c = X_BP[:,:,c] + Htfunc(np.reshape(cg_result,(Mlr,Nlr)))
                    X_BP_finalBP = np.dstack((X_BP_finalBP, X_BP_finalBP_c)) if X_BP_finalBP.size else X_BP_finalBP_c

            X_BP_finalBP_clip = np.clip(X_BP_finalBP, 0, 255)
            PSNR = 10*np.log10(255**2/np.mean((X0-X_BP_finalBP_clip)**2))
            print("X_BP_with_finalBP PSNR:",PSNR)
            scipy.misc.imsave("results\\SR\\" + sample_file_number + "_BP_with_ppBP.png", X_BP_finalBP_clip)


print("All PSNRs using LS term:")
print(all_LS_PSNR_results)
print("All PSNRs using BP term:")
print(all_BP_PSNR_results)



