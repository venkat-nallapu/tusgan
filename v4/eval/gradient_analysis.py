import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os

from generator import Generator
from critic import Critic

def compute_neuro_symbolic_loss(probs_fake, cond_vec):
    """
    Apply soft logical constraints on the synthetic diaries (scaled [0, 1]).
    probs_fake shape: (B, 9, 48)
    cond_vec shape: (B, 82 or 83)
    """
    loss = 0.0
    child_mask = cond_vec[:, 0] == 1
    if child_mask.any():
        child_work = probs_fake[child_mask, 0, :]
        loss += torch.mean(child_work)
    return loss

def analyze_gradients_v4():
    device = torch.device("cpu")
    BATCH_SIZE = 16
    
    # In v4, base_channels defaults are different (G=256, D=64, but we'll use smaller for test)
    G = Generator(base_channels=64).to(device)
    D = Critic(base_channels=32).to(device)
    
    scenarios = {
        "Adults (No Soft Penalty)": 0.0,
        "Children (Soft Penalty Active)": 1.0
    }
    
    grad_norms = {}
    
    for name, is_child_val in scenarios.items():
        G.zero_grad()
        D.zero_grad()
        
        z = torch.randn(BATCH_SIZE, 128, device=device)
        cond_vec = torch.zeros(BATCH_SIZE, 83, device=device)
        cond_vec[:, 0] = is_child_val
        
        dist_ids = torch.zeros(BATCH_SIZE, dtype=torch.long, device=device)
        state_ids = torch.zeros(BATCH_SIZE, dtype=torch.long, device=device)
        
        # Generator forward pass (v4 doesn't have hard masking inside G)
        fake = G(z, cond_vec, dist_ids, state_ids, temp=1.0, hard=True)
        
        # Critic forward pass
        fake_scores = D(fake, cond_vec, dist_ids, state_ids)
        
        # Compute soft penalty loss
        # fake is [-1, 1], convert back to [0, 1] for penalty
        probs_fake = (fake.squeeze(-1) + 1.0) / 2.0
        loss_logic = compute_neuro_symbolic_loss(probs_fake, cond_vec)
        
        # Simplified Generator Loss: Adversarial + Soft Logic (weight 10.0)
        # Note: WGAN-GP generator loss is -mean(fake_scores)
        loss_G = -torch.mean(fake_scores) + 10.0 * loss_logic
        loss_G.backward()
        
        norms = []
        labels = []
        for param_name, param in list(G.named_parameters()):
            if 'weight' in param_name and param.grad is not None:
                # Filter to show some key layers across the generator depth
                if 'attn1' in param_name and 'gamma' not in param_name:
                    short_name = param_name.replace('.weight', '')
                    norms.append(param.grad.norm().item())
                    labels.append(short_name)
                elif 'up' in param_name or 'out_conv' in param_name or 'fc' in param_name:
                    short_name = param_name.replace('.weight', '')
                    norms.append(param.grad.norm().item())
                    labels.append(short_name)
                
        grad_norms[name] = (labels, norms)
        
    os.makedirs('results', exist_ok=True)
    plt.figure(figsize=(14, 7))
    
    labels = grad_norms["Adults (No Soft Penalty)"][0]
    adult_norms = grad_norms["Adults (No Soft Penalty)"][1]
    child_norms = grad_norms["Children (Soft Penalty Active)"][1]
    
    x = np.arange(len(labels))
    width = 0.35
    
    plt.bar(x - width/2, adult_norms, width, label='Adults (No Penalty)', color='#1f77b4')
    plt.bar(x + width/2, child_norms, width, label='Children (Soft Penalty)', color='#2ca02c')
    
    plt.yscale('log')
    plt.ylabel('Gradient Norm (Log Scale)')
    plt.title('v4 Generator Gradient Flow: Soft Penalties Maintain Gradient Flow')
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('results/v4_gradient_analysis.png', dpi=300)
    print("Saved plot to v4/results/v4_gradient_analysis.png")

if __name__ == "__main__":
    analyze_gradients_v4()
