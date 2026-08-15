import json
import matplotlib.pyplot as plt
import os

DATA_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'results_variance.json')
OUTPUT_EPS = os.path.join(os.path.dirname(__file__), '..', 'paper', 'rollback_rate.eps')

def plot_metrics():
    print(f"[*] Loading results from {DATA_FILE}...")
    with open(DATA_FILE, 'r') as f:
        data = json.load(f)
        
    epochs = range(1, len(data['actor_loss']) + 1)
    
    # Create a figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = 'tab:red'
    ax1.set_xlabel('Training Epoch')
    ax1.set_ylabel('Actor Loss (MSE + Penalty)', color=color)
    ax1.plot(epochs, data['actor_loss'], color=color, marker='o', label='Actor Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Latent Roll-back Rate (Collision %)', color=color)  
    ax2.plot(epochs, data['rollback_rate'], color=color, marker='s', linestyle='--', label='Roll-back Rate')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('MA-WAC Training: Roll-back Rate & Actor Loss on Real Data')
    fig.tight_layout()
    
    print(f"[*] Saving plot to {OUTPUT_EPS}...")
    plt.savefig(OUTPUT_EPS, format='eps')
    plt.close()
    print("[+] Plot saved successfully.")

if __name__ == '__main__':
    plot_metrics()
