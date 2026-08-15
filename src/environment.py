import numpy as np

class MacroscopicEnvWrapper:
    """
    Wraps a base multi-agent environment (like MPE) to support temporally extended 
    macroscopic actions instead of just atomic step-by-step actions.
    """
    def __init__(self, base_env):
        self.env = base_env
    
    def reset(self):
        return self.env.reset()
        
    def step_macroscopic(self, joint_macro_actions):
        """
        Executes macroscopic actions for all agents.
        joint_macro_actions: list of dicts {'mu': np.array, 'sigma': np.array, 'tau': int}
        
        Returns:
            final_states, total_rewards, dones, infos
        """
        # For this prototype, we simulate the 'tau' duration by repeating the base mean action 'mu'
        # In a full implementation, this would interpolate or use a lower-level controller.
        
        # Find the maximum tau among all agents to determine how many atomic steps to run
        max_tau = max([int(action['tau']) for action in joint_macro_actions])
        
        total_rewards = [0] * len(joint_macro_actions)
        final_states = None
        dones = [False] * len(joint_macro_actions)
        infos = {}
        
        for step in range(max_tau):
            # Extract the atomic action for this specific step (simplified to just using 'mu')
            atomic_actions = []
            for action in joint_macro_actions:
                if step < int(action['tau']):
                    atomic_actions.append(action['mu'])
                else:
                    # Agent has finished its macroscopic action, idles or holds position
                    atomic_actions.append(np.zeros_like(action['mu']))
            
            # Step the underlying atomic environment
            try:
                states, rewards, step_dones, step_infos = self.env.step(atomic_actions)
                final_states = states
                for i in range(len(rewards)):
                    total_rewards[i] += rewards[i]
                    dones[i] = dones[i] or step_dones[i]
            except Exception as e:
                # If the base environment is not implemented yet, just return dummy data
                final_states = [np.zeros(10) for _ in joint_macro_actions]
                break
                
        return final_states, total_rewards, dones, infos
