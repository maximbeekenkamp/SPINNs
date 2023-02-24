# SPINNs
- M. Beekenkamp, A. Bhagavathula, P. LaDuca.

## Introduction
Separable Physics Informed Neural Networks (SPINNs), originally proposed in the paper “Separable PINN: Mitigating the Curse of Dimensionality in Physics-Informed Neural Networks.” by Cho et al., are an architectural overhaul of conventional PINNs that can approximate solutions to partial differential equations (PDEs). These architectural changes allow the authors to leverage forward-mode autodifferentiation (AD) and operate on a per-axis basis. Compared to conventional PINNs, which use point-wise processing, SPINNs presents a notable reduction in training time whilst maintaining accuracy.<br><br>

Although referenced in the paper, Cho et al. did not release the [original code](https://github.com/stnamjef/SPINN) until after we completed this problem. This repository is our implementation of the architecture proposed by the authors. By looking at a 2 dimensional heat equation, we show the improvements described in the paper.

## Results


## Motivating Literature and Sources
Cho, Junwoo, et al. ["Separable PINN: Mitigating the Curse of Dimensionality in Physics-Informed Neural Networks."](https://arxiv.org/abs/2211.08761) arXiv preprint arXiv:2211.08761 (2022).
