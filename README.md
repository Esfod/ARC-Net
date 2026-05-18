# ARC-Net

This code repository contains the code for my master thesis.

**Abstract:** 

Weather degradation such as haze, rain, and snow significantly reduces the quality
of outdoor images, posing challenges for both human perception and computer
vision tasks. While all-in-one image restoration models that are capable of handling
multiple degradation types have gained traction in recent years, existing approaches
are limited to single-frame inputs and do not exploit the temporal information
available in video sequences. This thesis proposes a novel multi-frame conditional
diffusion model for all-in-one weather restoration, combining a Swin transformer-
based multi-frame encoder, an optical flow alignment module, and a degradation
encoder within a conditional Denoising Diffusion Probabilistic Model (DDPM)
framework with Denoising Diffusion Implicit Model (DDIM) sampling.

The model is trained and evaluated on three public benchmarks covering Dehazing,
Deraining, and Desnowing, and compared against four established all-in-one
models: AirNet, PromptIR, MCDRNet and BioIR. Results show that the proposed
model does not yet match the State-of-the-Art (SOTA) in overall performance, with
Dehazing being the weakest task. However, the model achieves competitive results
on Deraining and demonstrates that the all-in-one formulation generalizes better than
single-task model, outperforming single-task models by 5.02 dB PSNR on Deraining
and 2.75 dB PSNR on Desnowing. An ablation study confirms that the multi-frame
encoder is the most critical component, while the alignment module provides limited
benefits under the current training conditions. The results suggest that multi-frame
diffusion-based restoration is promising but a data-demanding direction, and that
further development with larger datasets and refined alignment strategies could
bring this approach closer to the current SOTA.

The used datasets can be downloaded from this link [link](https://drive.google.com/drive/folders/15h_J-2K6MsW7g01RqP2DFyH4GUCCIwKw?usp=sharing)
RSVD    - Desnowing
SPAC    - Deraining
REVIDE  - Dehazing

To train add datasets in /dataset and run **train.py**

The required libraries can be accessed in **requirements.txt**

The weights of my model and the SOTAs can be accessed through this [link](https://drive.google.com/drive/folders/1CO5mwZI17bh08OFXGNVZYnmIyL7aSRh8?usp=sharing)