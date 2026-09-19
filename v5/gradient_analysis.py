import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os

from generator import Generator
from critic import Critic
from train import compute_infonce_loss, compute_spell_duration_loss

def analyze_gradients():
    device = torch.device("cpu")
    BATCH_SIZE = 16
    
    G = Generator(base_channels=64).to(device)
    D = Critic(base_channels=32, proj_dim=64).to(device)
    
    scenarios = {
        "Adults (No Masking)": 0.0,
        "Children (Hard Masking)": 1.0
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
        
        fake_hard, fake_soft = G(z, cond_vec, dist_ids, state_ids, temp=1.0, hard=True, return_soft=True)
        scores, f_proj, c_proj = D(fake_hard, cond_vec, dist_ids, state_ids, return_proj=True)
        
        loss_infonce = compute_infonce_loss(f_proj, c_proj, temperature=0.1)
        
        probs_fake = (fake_soft.squeeze(-1) + 1.0) / 2.0
        real_avg_durations = torch.ones(9, device=device) * 5.0
        loss_duration = compute_spell_duration_loss(probs_fake, real_avg_durations)
        
        total_loss = loss_infonce + loss_duration + scores.mean()
        total_loss.backward()
        
        norms = []
        labels = []
        for param_name, param in list(G.named_parameters()):
            # Filter to show some key layers across the generator depth
            if 'weight' in param_name and param.grad is not None:
                if 'transformer1' in param_name and 'norm' not in param_name:
                    short_name = param_name.replace('.weight', '').replace('transformer1.transformer.layers.', 'tf_')
                    norms.append(param.grad.norm().item())
                    labels.append(short_name)
                elif 'up' in param_name or 'out_conv' in param_name or 'fc' in param_name:
                    short_name = param_name.replace('.weight', '')
                    norms.append(param.grad.norm().item())
                    labels.append(short_name)
                
        grad_norms[name] = (labels, norms)
        
    os.makedirs('results', exist_ok=True)
    plt.figure(figsize=(14, 7))
    
    labels = grad_norms["Adults (No Masking)"][0]
    adult_norms = grad_norms["Adults (No Masking)"][1]
    child_norms = grad_norms["Children (Hard Masking)"][1]
    
    x = np.arange(len(labels))
    width = 0.35
    
    plt.bar(x - width/2, adult_norms, width, label='Adults (No Masking)', color='#1f77b4')
    plt.bar(x + width/2, child_norms, width, label='Children (Hard Masking)', color='#d62728')
    
    plt.yscale('log')
    plt.ylabel('Gradient Norm (Log Scale)')
    plt.title('v5 Generator Gradient Flow: The "Dead Gradient" Problem from Hard Masking')
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('results/v5_gradient_analysis.png', dpi=300)
    print("Saved plot to v5/results/v5_gradient_analysis.png")

if __name__ == "__main__":
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    analyze_gradients()
