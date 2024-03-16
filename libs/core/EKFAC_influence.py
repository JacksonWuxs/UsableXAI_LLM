# Authors: Yaochen Zhu and Xuansheng Wu
# Emails: uqp4qh@virginia.edu, wuxsmail@163.com
# Implement the EK-FAC approximated influence calculation for LLMs

import os
import pickle

import torch


class CovarianceEstimator():
    '''
        This class estimates the uncentered covariances of input
        and pre-activation grad with running average, calculates
        the SVD decomposition of the estimated cov matrices, and 
        save them to the disk via pickle.
    '''
    def __init__(self):
        # The covariance matrices for each layer
        self.layer_covs = {}  
        
        # The number of samples for each layer
        self.num_samples_A = {}
        self.num_samples_S = {}
        self.num_samples_lambda = {}
        
        # The eigenvalues and eigenvectors for each layer
        self.layer_svds = {}
        
        # The estimated eigenvectors
        self.layer_lambdas = {}

    def update_cov(self, layer_states, mask):
        for layer_name, (a_prev, ds_cur) in layer_states.items():
            # Initialize the cov estimations
            if layer_name not in self.layer_covs:
                a_hidden_size = a_prev.size(-1)
                ds_hidden_size = ds_cur.size(-1)
                self.layer_covs[layer_name] = {
                    'A': torch.zeros((a_hidden_size, a_hidden_size)),
                    'S': torch.zeros((ds_hidden_size, ds_hidden_size))
                }
                self.num_samples_A[layer_name] = 0
                self.num_samples_S[layer_name] = 0

            # Reshape a_prev and ds_cur to (num_samples, hidden_size)
            batch_size = a_prev.size(0)
            num_steps = a_prev.size(1)
            total_samples = int(mask.sum())

            ### Update the uncentered covariance for A ###                        
            # Apply the mask to a_prev and ds_cur
            mask = mask.reshape(-1, 1)
            a_prev_reshaped = a_prev.reshape(-1, a_prev.size(-1)) # (bs * ts, dim)
            masked_a_prev = a_prev_reshaped * mask.to(a_prev_reshaped.device)

            # Calculate the uncentered covariance matrices for A and S
            batch_cov_A = torch.matmul(masked_a_prev.transpose(0, 1), masked_a_prev) 
            batch_cov_A /= total_samples
            
            self.num_samples_A[layer_name] += total_samples
                        
            # Update the running covariance matrices for A and S
            if self.num_samples_A[layer_name] == total_samples:
                self.layer_covs[layer_name]['A'] = batch_cov_A
            else:
                old_weight = (self.num_samples_A[layer_name] - total_samples) / self.num_samples_A[layer_name]
                new_weight = total_samples / self.num_samples_A[layer_name]
                self.layer_covs[layer_name]['A'] = old_weight * self.layer_covs[layer_name]['A'] + new_weight * batch_cov_A         
            
            ### Update the uncentered covariance for S ###
            ds_cur_reshaped = ds_cur.view(-1, ds_cur.size(-1))
            
            if not torch.isnan(ds_cur_reshaped).any():
                masked_ds_cur = ds_cur_reshaped * mask.to(ds_cur_reshaped.device)

                batch_cov_S = torch.matmul(masked_ds_cur.transpose(0, 1), masked_ds_cur)
                batch_cov_S /= total_samples
                
                self.num_samples_S[layer_name] += total_samples

                # Update the running covariance matrices for A and S
                if self.num_samples_S[layer_name] == total_samples:
                    self.layer_covs[layer_name]['S'] = batch_cov_S
                else:
                    old_weight = (self.num_samples_S[layer_name] - total_samples) / self.num_samples_S[layer_name]
                    new_weight = total_samples / self.num_samples_S[layer_name]
                    self.layer_covs[layer_name]['S'] = old_weight * self.layer_covs[layer_name]['S'] + new_weight * batch_cov_S
            else:
                print(f"ignore layer: {layer_name} for grads")
 
                
    def update_lambdas(self, layer_states, mask):
        # Assuming a_prev, ds_cur, and mask are PyTorch tensors with the specified shapes
        # a_prev: (batch_size, num_steps, in_size)
        # ds_cur: (batch_size, num_steps, out_size)
        # mask: (batch_size, num_steps)
        for layer_name, (a_prev, ds_cur) in layer_states.items():
            # Initialize the lambda estimations
            if layer_name not in self.layer_lambdas:
                a_hidden_size = a_prev.size(-1)
                ds_hidden_size = ds_cur.size(-1)
                self.layer_lambdas[layer_name] = torch.zeros((ds_cur.size(-1), a_prev.size(-1)))
                self.num_samples_lambda[layer_name] = 0
            
            # Obtain the kronecker product between Q_S and Q_A
            # The result has shape (in_size * out_size, in_size * out_size)
            Q_A = self.layer_svds[layer_name]["Q_A"]
            Q_S = self.layer_svds[layer_name]["Q_S"]
            
            # Obtain info regarding the data
            batch_size = a_prev.size(0)
            timesteps = a_prev.size(1)
            
            # Apply the mask
            a_prev_masked = a_prev * mask.unsqueeze(-1).to(a_prev.device)
            ds_cur_masked = ds_cur * mask.unsqueeze(-1).to(ds_cur.device)

            # Perform batched matrix multiplication to get the outer product
            # Reshape ds_cur_masked to (batch_size, num_steps, out_size, 1)
            # Reshape a_prev_masked to (batch_size, num_steps, 1, in_size)
            # batch_dtheta_steps: (batch_size, num_steps, out_size, in_size)
            # batch_dtheta = (ds_cur_masked.unsqueeze(-1) @ a_prev_masked.unsqueeze(2)).sum(axis=1)
            
            batch_dtheta = torch.zeros(batch_size, ds_cur_masked.shape[-1], 
                                       a_prev_masked.shape[-1], device=ds_cur.device)
            
            for bs in range(batch_size):
                for ts in range(timesteps):
                    batch_dtheta[bs] += ds_cur_masked[bs, ts].unsqueeze(1) @ a_prev_masked[bs, ts].unsqueeze(0)
            
            # Calculate the estimation (inefficient)
            # kron_basis = torch.kron(Q_A, Q_S) (memory OOD)
            # batch_dtheta = torch.sum(batch_dtheta_steps, dim=1).reshape(batch_size, -1)
            # batch_lambda = (torch.square(batch_dtheta @ kron_basis.T)).mean(axis=0)
           
            # https://math.stackexchange.com/questions/1879933/vector-multiplication-with-multiple-kronecker-products
            batch_lambda = torch.square(Q_S @ batch_dtheta @ Q_A.T).mean(axis=0)
            
            # Update the count
            self.num_samples_lambda[layer_name] += batch_size

            # Update the running covariance matrices for A and S
            if self.num_samples_lambda[layer_name] == batch_size:
                self.layer_lambdas[layer_name] = batch_lambda
            else:
                old_weight = (self.num_samples_lambda[layer_name] - batch_size) / self.num_samples_lambda[layer_name]
                new_weight = batch_size / self.num_samples_lambda[layer_name]
                self.layer_lambdas[layer_name] = old_weight * self.layer_lambdas[layer_name] + new_weight * batch_lambda    
       
    def get_running_covariance(self, layer_name):
        return self.layer_covs.get(layer_name, {'A': None, 'S': None})

    def get_running_lambda(self, layer_name):
        return self.layer_lambdas.get(layer_name, 0)

    def get_num_samples_A(self, layer_name):
        return self.num_samples_A.get(layer_name, 0)

    def get_num_samples_S(self, layer_name):
        return self.num_samples_S.get(layer_name, 0)

    def get_num_samples_lambda(self, layer_name):
        return self.num_samples_lambda.get(layer_name, 0)

    def calculate_eigenvalues_and_vectors(self):
        for layer_name, cov_matrices in self.layer_covs.items():
            eigenvalues_S, eigenvectors_S = torch.linalg.eigh(cov_matrices['S'], UPLO='U')
            eigenvalues_A, eigenvectors_A = torch.linalg.eigh(cov_matrices['A'], UPLO='U')

            self.layer_svds[layer_name] = {
                'Q_S': eigenvectors_S,
                'Q_A': eigenvectors_A,
            }

    def get_eigenvalues_and_vectors(self, layer_name):
        return self.layer_svds.get(layer_name, None)
    
    def save_to_disk(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)
        svds_file = os.path.join(dir, "layer_svds.pkl")
        with open(svds_file, "wb") as f:
            pickle.dump(self.layer_svds, f)
        lambdas_file = os.path.join(dir, "layer_lambdas.pkl")
        with open(lambdas_file, "wb") as f:
            pickle.dump(self.layer_lambdas, f)
            

class InfluenceEstimator():
    '''
        Give the layer_svds calculated by the EKFACCovarianceEstimator
        given the layerwise gradient of the query of interest, EKFACInfluenceEstimator
        first calculate the HVP between the approximate Hessian and grad.
        Given the layerwise gradient of a training sample, we also 
        calculate the layerwise influence as well as the total influences.
    '''
    def __init__(self, layer_svds, layer_lambdas, lambda_value=None):
        self.layer_svds = layer_svds
        self.layer_lambdas = layer_lambdas
        self.lambda_value = lambda_value

    @classmethod
    def load_from_disk(cls, dir, lambda_value=None):
        svd_path = os.path.join(dir, "layer_svds.pkl")
        assert os.path.exists(svd_path)
        with open(svd_path, "rb") as f:
            svds = pickle.load(f)
        lambda_path = os.path.join(dir, "layer_lambdas.pkl")
        assert os.path.exists(lambda_path)
        with open(lambda_path, "rb") as f:
            lambdas = pickle.load(f)
        return cls(svds, lambdas, lambda_value)
        
    def calculate_hvp(self, layer_grad_query):
        layer_hvps = {}
        for layer_name, grad in layer_grad_query.items():
            svd_data = self.layer_svds[layer_name]
            
            # Get the SVDs of the two matrices
            Q_S = svd_data['Q_S']
            Q_A = svd_data['Q_A']
            
            in_shape = Q_A.shape[-1]
            out_shape = Q_S.shape[-1]
            
            # Get the Lambdas
            Lambda = self.layer_lambdas[layer_name]

            # Calculate (\mathbf{G}+\lambda \mathbf{I})^{-1} \mathbf{v} using the provided formula
            if not self.lambda_value:
                lambda_value = Lambda.mean()*0.1
            else:
                lambda_value = self.lambda_value
            
            Lambda += lambda_value

            hvp = Q_S.T @ ((Q_S @ grad @ Q_A.T) / Lambda) @ Q_A
            layer_hvps[layer_name] = hvp.reshape(-1)
        return layer_hvps

    def calculate_layerwise_influence(self, layer_hvps_query, layer_grad_train):
        layer_influence = {}
        for layer_name, hvp in layer_hvps_query.items():
            grad_train = layer_grad_train[layer_name][1]
            # Reshape hvp and grad_train into vectors
            grad_train_vector = grad_train.reshape(-1)
            # Calculate the inner product
            influence = torch.dot(hvp, grad_train_vector)
            layer_influence[layer_name] = influence
        return layer_influence

    def calculate_total_influence(self, layer_hvps_query, layer_grad_train):
        layer_influence = self.calculate_layerwise_influence(layer_hvps_query, layer_grad_train)
        total_influence = sum(_.cpu() for _ in layer_influence.values())
        return total_influence