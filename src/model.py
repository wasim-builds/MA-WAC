import torch
import torch.nn as nn
import torch.nn.functional as F

class PerceptionEncoder(nn.Module):
    """
    Encodes an agent's local observation into a latent space z_t.
    """
    def __init__(self, obs_dim, latent_dim):
        super(PerceptionEncoder, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, latent_dim)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        z = self.fc2(x)
        return z

class MacroscopicActor(nn.Module):
    """
    Maps latent z_t to a macroscopic action A_t = (mu, sigma, tau).
    """
    def __init__(self, latent_dim, action_dim):
        super(MacroscopicActor, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        
        # Outputs for continuous action parameters
        self.mu_head = nn.Linear(128, action_dim)
        self.sigma_head = nn.Linear(128, action_dim)
        self.tau_head = nn.Linear(128, 1) # Predicts duration

    def forward(self, z):
        x = F.relu(self.fc1(z))
        
        mu = torch.tanh(self.mu_head(x)) # Bounded action space
        sigma = F.softplus(self.sigma_head(x)) + 1e-5 # Strictly positive variance
        tau = F.relu(self.tau_head(x)) + 1.0 # Duration must be at least 1 step
        
        return mu, sigma, tau

class SharedLatentWorldProxy(nn.Module):
    """
    Takes the joint state Z_t and joint actions A_t, and predicts Z_{t+tau}.
    Acts as the environment simulator in the latent space.
    """
    def __init__(self, num_agents, latent_dim, action_dim):
        super(SharedLatentWorldProxy, self).__init__()
        self.num_agents = num_agents
        self.latent_dim = latent_dim
        
        # We flatten the joint states and joint actions for this simplified MLP proxy.
        # In a full version, this would be a Transformer or Graph Neural Network.
        input_dim = (num_agents * latent_dim) + (num_agents * (action_dim * 2 + 1))
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_agents * latent_dim)
        )

    def forward(self, joint_z, joint_mu, joint_sigma, joint_tau):
        """
        joint_z: [batch_size, num_agents, latent_dim]
        joint_mu, joint_sigma: [batch_size, num_agents, action_dim]
        joint_tau: [batch_size, num_agents, 1]
        """
        batch_size = joint_z.size(0)
        
        # Flatten everything
        flat_z = joint_z.view(batch_size, -1)
        flat_mu = joint_mu.view(batch_size, -1)
        flat_sigma = joint_sigma.view(batch_size, -1)
        flat_tau = joint_tau.view(batch_size, -1)
        
        # Concatenate into a single input vector
        x = torch.cat([flat_z, flat_mu, flat_sigma, flat_tau], dim=-1)
        
        # Predict the next flattened joint latent state
        next_flat_z = self.net(x)
        
        # Reshape back to [batch_size, num_agents, latent_dim]
        next_joint_z = next_flat_z.view(batch_size, self.num_agents, self.latent_dim)
        return next_joint_z
