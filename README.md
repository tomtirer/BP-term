# BP-term
Back-Projection based Fidelity Term for Ill-Posed Linear Inverse Problems  
https://arxiv.org/pdf/1906.06794.pdf

@article{tirer2020back,  
  &nbsp; &nbsp; title={Back-projection based fidelity term for ill-posed linear inverse problems},  
  &nbsp; &nbsp; author={Tirer, Tom and Giryes, Raja},  
  &nbsp; &nbsp; journal={IEEE Transactions on Image Processing},  
  &nbsp; &nbsp; volume={29},  
  &nbsp; &nbsp; pages={6164--6179},  
  &nbsp; &nbsp; year={2020},  
  &nbsp; &nbsp; publisher={IEEE}  
}

<br/>

**GAN prior:**  
Python implementation (using TensorFlow) of recovery based on GAN prior and the BP loss can be found in the folder "GAN prior".  
In the paper we used DCGAN generator (based on the implementation of https://github.com/carpedm20/DCGAN-tensorflow), which can be dowloaded from
https://drive.google.com/drive/folders/1V1J6PLWHAnx0sHb8HPLw2lvR09TeT9cf?usp=sharing
and put in the folder DCGAN_checkpoint/celebA_10_64_64/.

Original: <img src="/GAN%20prior/results/SR/202587_X0.png"> &nbsp;
Low-res: <img src="/GAN%20prior/results/SR/202587_Y.png"> &nbsp;
LS (Adam): <img src="/GAN%20prior/results/SR/202587_LS.png"> &nbsp;
BP (Adam): <img src="/GAN%20prior/results/SR/202587_BP.png">

<br/>

**Deep Image Prior:**  
Python implementation (using PyTorch) of recovery based on DIP and the BP loss can be found in https://github.com/jennyzu/BP-DIP-deblurring.

<br/>

**Other priors (TV, BM3D):**  
In general, applying ISTA (proximal gradient method) on the BP-term + prior is similar to the IDBP baseline https://github.com/tomtirer/IDBP.  
Clearner Matlab code with extension to FISTA (that uses Nesterov's accelerated gradient) will be uploaded.  
Python implementation of the IDBP baseline can be found in https://github.com/tomtirer/IDBP-python.
