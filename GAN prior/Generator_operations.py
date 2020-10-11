# Solving inverse problems using GAN prior and BP / LS loss functions

# Reference: "Back-Projection based Fidelity Term for Ill-Posed Linear Inverse Problems"
# Authors: Tom Tirer and Raja Giryes
# Journal: IEEE Transactions on Image Processing, 2020.

import scipy.misc
import numpy as np
from model_DCGAN import DCGAN
import tensorflow as tf
from utils import *

choice = 'DCGAN'

def solve_inv_prob_with_GAN_prior(Y,Hfunc,Htfunc,z_init,loss_type='BP',Hdagger_func=None,flag_CS=0):

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    N_initials = 10
    L2_weight = 0
    use_adam_flag = 1
    LR = 0.1
    N_iter = 2001

    loss_array = np.array([])
    momentum = 0  # for GD & ADAM
    variance = 0  # for ADAM
    np.random.seed(0)
    best_loss = 1e9

    tf.reset_default_graph()
    with tf.Session(config=run_config) as sess:
        if choice == 'DCGAN':
            checkpoint_dir = "DCGAN_checkpoint"
            Gen = DCGAN(
                sess,
                input_width=108,
                input_height=108,
                output_width=64,
                output_height=64,
                batch_size=N_initials,
                sample_num=64,
                z_dim=100,
                dataset_name="celebA",
                # input_fname_pattern=FLAGS.input_fname_pattern,
                crop=True,
                checkpoint_dir=checkpoint_dir
                # sample_dir=FLAGS.sample_dir
            )
            # show_all_variables()
            if not Gen.load(checkpoint_dir)[0]:
                raise Exception("[!] Train a model first, then run test mode")

            z = np.random.uniform(-1, 1, [N_initials, Gen.z_dim])

        if flag_CS==0:
            Y = np.tile(Y/255*2-1,(N_initials, 1, 1, 1))  # transform intensity range to [-1,1]
        else:
            Y = np.tile(Y,(N_initials, 1, 1, 1))  # intensity range is already [-1,1] for CS

        if z_init.size:
            z = z_init
        else:
            z_init = z

        for iter in range(1,N_iter+1):

            # forward G
            Gz = sess.run(Gen.sampler, feed_dict={Gen.z: z})

            ##########################################################################################################
            # compute loss and derivatives for the last layer

            HGz_all = np.array([])
            for n in range(N_initials):
                HGz = np.array([])
                for c in range(Gz.shape[3]):
                    HGz = np.dstack((HGz, Hfunc(Gz[n,:,:,c]))) if HGz.size else Hfunc(Gz[n,:,:,c])
                HGz = HGz[np.newaxis, ...]
                HGz_all = np.append(HGz_all, HGz, axis=0) if HGz_all.size else HGz
            temp1 = Y - HGz_all

            if loss_type == 'LS':
                loss = np.sum(temp1**2,axis=(1,2,3)) # loss = || Y - H*Gz ||
                loss = loss[..., np.newaxis]
                loss_der = np.array([])
                for n in range(N_initials):
                    temp = np.array([])
                    for c in range(Gz.shape[3]):
                        temp = np.dstack((temp, -2*Htfunc(temp1[n,:,:,c]))) if temp.size else -2*Htfunc(temp1[n,:,:,c])
                    temp = temp[np.newaxis, ...]
                    loss_der = np.append(loss_der, temp, axis=0) if loss_der.size else temp

            if loss_type == 'BP':
                if Hdagger_func == None:
                    [Mlr, Nlr] = Y.shape[1:3]
                    HHt_cg = lambda z: np.reshape(Hfunc(Htfunc(np.reshape(z,(Mlr,Nlr)))),(Mlr*Nlr,1))+z*0
                    invHHt_temp1 = np.array([])
                    for n in range(N_initials):
                        invHHt_temp1_n = np.array([])
                        for c in range(Y.shape[3]):
                            temp1_c = temp1[n,:,:,c]
                            [cg_result, cg_iter, cg_residual] = cg(np.zeros((Mlr*Nlr,1)), HHt_cg, np.reshape(temp1_c,(Mlr*Nlr,1)), 100, 10**-6) # cg_result = inv(H*Ht)*(Y-H*Gz)
                            if np.sqrt(cg_residual)>10**-3:
                                print("cg: finished after ", cg_iter, " iterations with norm(residual) = ", np.sqrt(cg_residual)) #, " - Use preconditioning or tikho regularization (epsilon) for HHt_cg")
                            invHHt_temp1_n = np.dstack((invHHt_temp1_n, np.reshape(cg_result,(Mlr,Nlr)))) if invHHt_temp1_n.size else np.reshape(cg_result,(Mlr,Nlr))
                        invHHt_temp1_n = invHHt_temp1_n[np.newaxis, ...]
                        invHHt_temp1 = np.append(invHHt_temp1, invHHt_temp1_n, axis=0) if invHHt_temp1.size else invHHt_temp1_n

                loss_der = np.array([])
                for n in range(N_initials):
                    temp = np.array([])
                    for c in range(Gz.shape[3]):
                        if Hdagger_func == None:
                            temp = np.dstack((temp, -2*Htfunc(invHHt_temp1[n,:,:,c]))) if temp.size else -2*Htfunc(invHHt_temp1[n,:,:,c])
                        else:
                            temp = np.dstack((temp, -2*Hdagger_func(temp1[n,:,:,c]))) if temp.size else -2*Hdagger_func(temp1[n,:,:,c])

                    temp = temp[np.newaxis, ...]
                    loss_der = np.append(loss_der, temp, axis=0) if loss_der.size else temp

                loss = np.sum(loss_der**2,axis=(1,2,3)) # loss = || Ht*inv(H*Ht)*(Y - H*Gz) ||
                loss = loss[..., np.newaxis] # to make it N_initials x 1 vector

            ##########################################################################################################

            # backward G
            loss_der = np.float32(loss_der)
            dz = sess.run(Gen.compute_dz_given_dGz, feed_dict={Gen.z_sampled: z, Gen.dGz_sampled: loss_der})
            dz = np.squeeze(np.asarray(dz)) + L2_weight*z

            if not use_adam_flag:
                # simple GD
                momentum = - LR*dz + 0.9*momentum
                delta = momentum
            else:
                # ADAM
                [delta,momentum,variance] = adam_update(iter,dz,momentum,variance,LR)
            # Update parameters
            z = z + delta

            loss_array = np.append(loss_array, loss, axis=1) if loss_array.size else loss
            current_best_loss = np.min(loss)
            if current_best_loss<best_loss:
                best_loss = current_best_loss
                ind = np.argmin(loss, axis=None)
                Gz_tilde = Gz[ind, :, :, :]

            if np.mod(iter,100)==1:
                print("Generator operation: iter #", iter, ", current best loss = ", np.min(loss), ", best loss = ", np.min(best_loss), ", LR = ", LR, ", use_adam_flag = ", use_adam_flag)


        if flag_CS==0:
            Gz_tilde = (Gz_tilde+1)/2*255
        else:
            Gz_tilde = Gz_tilde # for CS
        return Gz_tilde, z_init

