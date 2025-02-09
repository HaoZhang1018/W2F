import gc

import copy
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import cvxpy as cp

from .pyt_utils import all_reduce_tensor

class GradBlance():
    def __init__(
        self,
        optimizer,
        n_tasks: int,
        device: torch.device,
        max_norm: float = 1.0,
        update_weights_every: int = 1,
        optim_niter=20,
        distributed=False,
        num_gpus=1,
    ):
        self._optim = optimizer

        self.n_tasks = n_tasks
        self.device = device
        
        self.optim_niter = optim_niter
        self.update_weights_every = update_weights_every
        self.max_norm = max_norm
        self.shared_paramter = None

        self.prvs_alpha_param = None
        self.normalization_factor = np.ones((1,))
        self.init_gtg = np.eye(self.n_tasks)
        self.global_step = 0.0
        self.prvs_alpha = np.ones(self.n_tasks, dtype=np.float32)

        self.distributed = distributed
        self.num_gpus = num_gpus

        return

    @property
    def optimizer(self):
        return self._optim
    
    @property
    def param_groups(self):
        return self._optim.param_groups
    
    @property
    def state_dict(self):
        return self._optim.state_dict
    
    @property
    def load_state_dict(self):
        return self._optim.load_state_dict

    def zero_grad(self):
        '''
        clear the gradient of the parameters
        '''

        return self._optim.zero_grad(set_to_none=True)

    def step(self):
        '''
        update the parameters with the gradient
        '''
        return self._optim.step()
    
    def find_shared_paramter(self, shared):
        if self.shared_paramter == None:
            idx = 0
            self.shared_paramter = []

            for group in self._optim.param_groups:
                for p in group['params']:
                    # if p.grad is None: continue
                    if torch.all(shared[idx]):
                        self.shared_paramter.append(p)
                    idx += 1
        
        return self.shared_paramter


    def backward(self, losses):
        '''
        calculate the gradient of the parameters

        input:
        - objectives: a list of objectives
        '''
        grads, shapes, share = self._pack_grad(losses)
        pc_grads, grad_conflict = self._project_conflicting(grads, share)

        loss = self.get_weighted_loss(losses, pc_grads, share)
        if self.distributed:
            loss = all_reduce_tensor(loss, world_size=self.num_gpus)

        loss.backward()

        pc_share = self._unflatten_grad(share, shapes[0])
        # make sure the solution for shared params has norm <= self.eps
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.find_shared_paramter(pc_share), self.max_norm)
        return loss

    def _project_conflicting(self, grads, shared):
        pc_grad = copy.deepcopy(grads)
        grad_conflict = False

        for g_i in pc_grad:
            random.shuffle(grads)
            for g_j in grads:
                g_i_g_j = torch.dot(g_i, g_j)
                if g_i_g_j < 0:
                    grad_conflict = True
                    g_i = g_i - (g_i_g_j) * g_j / (g_j.norm()**2)
        
        return pc_grad, grad_conflict

    def _set_grad(self, grads, shared):
        '''
        set the modified gradients to the network
        '''

        idx = 0
        
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                p.grad = torch.where(shared[idx], grads[idx], p.grad)

                idx += 1

        assert len(grads) == idx, f"{idx}"
        return

    def _pack_grad(self, objectives):
        '''
        pack the gradient of the parameters of the network for each objective
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grads, shapes, has_grads = [], [], []

        for obj in objectives:
            self._optim.zero_grad(set_to_none=True)
            obj.backward(retain_graph=True)
            grad, shape, has_grad = self._retrieve_grad()
            grads.append(self._flatten_grad(grad, shape))
            has_grads.append(self._flatten_grad(has_grad, shape))
            shapes.append(shape)
            
        shared = torch.stack(has_grads).prod(0).bool()
        return grads, shapes, shared

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self):
        '''
        get the gradient of the parameters of the network with specific 
        objective
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grad, shape, has_grad = [], [], []
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                # tackle the multi-head scenario

                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p).to(p.device))
        return grad, shape, has_grad

    def _stop_criteria(self, gtg, alpha_t):
        return (
            (self.alpha_param.value is None)
            or (np.linalg.norm(gtg @ alpha_t - 1 / (alpha_t + 1e-10)) < 1e-3)
            or (
                np.linalg.norm(self.alpha_param.value - self.prvs_alpha_param.value)
                < 1e-6
            )
        )

    def solve_optimization(self, gtg: np.array):
        self.G_param.value = gtg
        self.normalization_factor_param.value = self.normalization_factor

        alpha_t = self.prvs_alpha
        for _ in range(self.optim_niter):
            self.alpha_param.value = alpha_t
            self.prvs_alpha_param.value = alpha_t

            try:
                self.prob.solve(solver=cp.ECOS, warm_start=True, max_iters=100)
            except:
                self.alpha_param.value = self.prvs_alpha_param.value

            if self._stop_criteria(gtg, alpha_t):
                break

            alpha_t = self.alpha_param.value

        if alpha_t is not None:
            self.prvs_alpha = alpha_t

        return self.prvs_alpha

    def _calc_phi_alpha_linearization(self):
        G_prvs_alpha = self.G_param @ self.prvs_alpha_param
        prvs_phi_tag = 1 / self.prvs_alpha_param + (1 / G_prvs_alpha) @ self.G_param
        phi_alpha = prvs_phi_tag @ (self.alpha_param - self.prvs_alpha_param)
        return phi_alpha

    def _init_optim_problem(self):
        self.alpha_param = cp.Variable(shape=(self.n_tasks,), nonneg=True)
        self.prvs_alpha_param = cp.Parameter(
            shape=(self.n_tasks,), value=self.prvs_alpha
        )
        self.G_param = cp.Parameter(
            shape=(self.n_tasks, self.n_tasks), value=self.init_gtg
        )
        self.normalization_factor_param = cp.Parameter(
            shape=(1,), value=np.array([1.0])
        )

        self.phi_alpha = self._calc_phi_alpha_linearization()

        G_alpha = self.G_param @ self.alpha_param
        constraint = []
        for i in range(self.n_tasks):
            constraint.append(
                -cp.log(self.alpha_param[i] * self.normalization_factor_param)
                - cp.log(G_alpha[i])
                <= 0
            )
        obj = cp.Minimize(
            cp.sum(G_alpha) + self.phi_alpha / self.normalization_factor_param
        )
        self.prob = cp.Problem(obj, constraint)

    def get_weighted_loss(
        self,
        losses,
        grads,
        shared,
    ):
        grads = [g[shared] for g in grads]

        if self.global_step == 0:
            self._init_optim_problem()

        if (self.global_step % self.update_weights_every) == 0:
            self.global_step += 1

            G = torch.stack(tuple(v for v in grads))
            GTG = torch.mm(G, G.t())

            self.normalization_factor = (
                torch.norm(GTG).detach().cpu().numpy().reshape((1,))
            )
            GTG = GTG / self.normalization_factor.item()
            alpha = self.solve_optimization(GTG.cpu().detach().numpy())
            alpha = torch.from_numpy(alpha)

        else:
            self.global_step += 1
            alpha = self.prvs_alpha

        weighted_loss = sum([losses[i] * alpha[i] for i in range(len(alpha))])
        return weighted_loss