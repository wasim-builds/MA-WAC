import json
import torch
import torch.nn.functional as F
import torch.optim as optim
from model import PerceptionEncoder, MacroscopicActor, SharedLatentWorldProxy
from dataset_loader import download_real_trajectory_data, parse_trajectory_data

def check_latent_collision(predicted_joint_z, collision_threshold=0.5):
    """
    The Roll-back Auditor. Checks if the predicted future latent states 
    of any two agents are suspiciously close (indicating a collision).
    """
    batch_size, num_agents, latent_dim = predicted_joint_z.shape
    collisions = torch.zeros(batch_size, dtype=torch.bool)
    
    for b in range(batch_size):
        for i in range(num_agents):
            for j in range(i+1, num_agents):
                dist = torch.norm(predicted_joint_z[b, i] - predicted_joint_z[b, j])
                if dist < collision_threshold:
                    collisions[b] = True
                    break
    return collisions

def train_ma_wac(epochs=10):
    print(f"[*] Initializing MA-WAC Training for {epochs} epochs using REAL DATASET...")
    
    # 1. Load Real Data
    fp = download_real_trajectory_data()
    real_obs, real_actions = parse_trajectory_data(fp, seq_len=10, max_agents=3)
    
    batch_size, num_agents, obs_dim = real_obs.shape
    _, _, action_dim = real_actions.shape
    latent_dim = 16
    
    # Initialize Networks
    encoder = PerceptionEncoder(obs_dim, latent_dim)
    actor = MacroscopicActor(latent_dim, action_dim)
    slwp = SharedLatentWorldProxy(num_agents, latent_dim, action_dim)
    
    # Optimizers
    actor_opt = optim.Adam(list(encoder.parameters()) + list(actor.parameters()), lr=1e-3)
    slwp_opt = optim.Adam(slwp.parameters(), lr=1e-3)
    
    history = {"actor_loss": [], "slwp_loss": [], "rollback_rate": []}
    
    for epoch in range(epochs):
        # ---------------------------------------------------------
        # 1. Forward Pass (Perception & Actor) on REAL DATA
        # ---------------------------------------------------------
        joint_z = torch.zeros(batch_size, num_agents, latent_dim)
        joint_mu = torch.zeros(batch_size, num_agents, action_dim)
        joint_sigma = torch.zeros(batch_size, num_agents, action_dim)
        joint_tau = torch.zeros(batch_size, num_agents, 1)
        
        for i in range(num_agents):
            z = encoder(real_obs[:, i, :])
            mu, sigma, tau = actor(z)
            joint_z[:, i, :] = z
            joint_mu[:, i, :] = mu
            joint_sigma[:, i, :] = sigma
            joint_tau[:, i, :] = tau
            
        # ---------------------------------------------------------
        # 2. SLWP Prediction & Auditing
        # ---------------------------------------------------------
        predicted_next_joint_z = slwp(joint_z.detach(), joint_mu.detach(), joint_sigma.detach(), joint_tau.detach())
        collisions = check_latent_collision(predicted_next_joint_z)
        rollback_rate = collisions.float().mean().item()
        
        # ---------------------------------------------------------
        # 3. Loss Calculation & Backprop
        # ---------------------------------------------------------
        # Actor Cloning Loss (MSE against REAL expert actions)
        actor_loss = F.mse_loss(joint_mu, real_actions)
        
        # Apply Roll-back penalty if collisions predicted
        if collisions.any():
            actor_loss += 10.0 * collisions.float().sum()
            
        actor_opt.zero_grad()
        actor_loss.backward()
        actor_opt.step()
        
        # SLWP Self-Supervised Loss (Simplified: assumes next state is static for prototype)
        with torch.no_grad():
            true_next_joint_z = torch.zeros(batch_size, num_agents, latent_dim)
            for i in range(num_agents):
                true_next_joint_z[:, i, :] = encoder(real_obs[:, i, :])
                
        slwp_loss = F.mse_loss(predicted_next_joint_z, true_next_joint_z)
        slwp_opt.zero_grad()
        slwp_loss.backward()
        slwp_opt.step()
        
        print(f"Epoch {epoch+1}/{epochs} | Actor Loss: {actor_loss.item():.4f} | SLWP Loss: {slwp_loss.item():.4f} | Roll-back Rate: {rollback_rate:.2f}")
        
        history["actor_loss"].append(round(actor_loss.item(), 4))
        history["slwp_loss"].append(round(slwp_loss.item(), 4))
        history["rollback_rate"].append(round(rollback_rate, 2))
        
    # Save the training metrics to a JSON file for plotting later
    with open('data/results_variance.json', 'w') as f:
        json.dump(history, f, indent=4)
        
    print("[+] Training complete. Results saved to data/results_variance.json.")

if __name__ == "__main__":
    train_ma_wac()
