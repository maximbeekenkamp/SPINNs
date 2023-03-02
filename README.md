# SPINNs
- M. Beekenkamp, A. Bhagavathula, P. LaDuca.

## Introduction
Separable Physics Informed Neural Networks (SPINNs), originally proposed in the paper “Separable PINN: Mitigating the Curse of Dimensionality in Physics-Informed Neural Networks.” by Cho et al., are an architectural overhaul of conventional PINNs that can approximate solutions to partial differential equations (PDEs). These architectural changes allow the authors to leverage forward-mode autodifferentiation (AD) and operate on a per-axis basis. Compared to conventional PINNs, which use point-wise processing, SPINNs presents a notable reduction in training time whilst maintaining accuracy.<br><br>

Although referenced in the paper, Cho et al. did not release the [original code](https://github.com/stnamjef/SPINN) until after we completed this problem. This repository is our implementation of the architecture proposed by the authors. By looking at a 2 dimensional heat equation, we show the improvements described in the paper.

## Results
All the algorithms were implemented with $100^2$ collocation points and an architecture of 3 hidden layers with 20 hidden feature sizes. We used the Adam optimizer with a learning rate of 0.0005 and trained for 10,000 iterations for every experiment. All experiments are trained with full batch for both models and reported error metrics are average relative L2 errors computed by ∥uˆ − u∥2/∥u∥2, where uˆ is the model prediction and u is the reference solution. All experiments were performed five times with different random seeds.

|Algorithm|Total Loss|L2 error|Time [ms/iter]|
|:-:|:-|:-|:-|
|Simple PINN|$0.71 \pm 0.33$|$ 14.43 \pm 0.43$|$26.30 \pm 0.083$|
|SA-PINN|$0.64 \pm 0.19$|$ 14.22 \pm 0.21$|$28.09 \pm 0.063$|
|S-PINN|$0.059 \pm 0.015$|$0.0099 \pm 0.0065$|$0.93 \pm 0.024$|

## Motivating Literature and Sources
Cho, Junwoo, et al. ["Separable PINN: Mitigating the Curse of Dimensionality in Physics-Informed Neural Networks."](https://arxiv.org/abs/2211.08761) arXiv preprint arXiv:2211.08761 (2022).
