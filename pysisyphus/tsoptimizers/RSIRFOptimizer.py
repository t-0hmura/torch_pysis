# [1] https://doi.org/10.1007/s002140050387
#     Bofill, 1998


import numpy as np

from pysisyphus.tsoptimizers.TSHessianOptimizer import TSHessianOptimizer

import torch

class RSIRFOptimizer(TSHessianOptimizer):
    def optimize(self):
        energy, gradient_full, H, eigvals, eigvecs, resetted = self.housekeeping()
        gradient_act = self._to_active_vec(gradient_full)
        self.update_ts_mode(eigvals, eigvecs)

        self.log(
            "Using projection to construct image potential gradient "
            f"and hessian for root(s) {self.roots}."
        )
        # Projection matrix to construct g* and H*
        if isinstance(H, torch.Tensor):
            dim = H.size(0)
            P = torch.eye(dim, device=H.device, dtype=H.dtype)
            for root in self.roots:
                trans_vec = eigvecs[:, root]
                P -= 2 * torch.outer(trans_vec, trans_vec)
            H_star = P @ H
            eigvals_, eigvecs_ = torch.linalg.eigh(H_star)
        else:
            dim = H.shape[0]
            P = np.eye(dim)
            for root in self.roots:
                trans_vec = eigvecs[:, root]
                P -= 2 * np.outer(trans_vec, trans_vec)
            H_star = P.dot(H)
            eigvals_, eigvecs_ = np.linalg.eigh(H_star)
        # Neglect small eigenvalues
        eigvals_, eigvecs_ = self.filter_small_eigvals(eigvals_, eigvecs_)

        if isinstance(H, torch.Tensor):
            grad_star = P @ gradient_act
        else:
            grad_star = P.dot(gradient_act)
        step_act = self.get_rs_step(eigvals_, eigvecs_, grad_star, name="RS-I-RFO")

        self.predicted_energy_changes.append(
            self.rfo_model(gradient_act, self.H, step_act)
        )
        step_full = self._to_full_vec(step_act)
        if isinstance(step_full, torch.Tensor):
            step_full = step_full.cpu().numpy()
        return step_full
