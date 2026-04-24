import numpy as np
import matplotlib.pyplot as plt

def plot_reconstruction_loss(loss_file, output_image):
    loss_history = np.load(loss_file)
    plt.figure()
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o')
    plt.title('Reconstruction Loss vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    plt.savefig(output_image)
    print(f"Plot saved to {output_image}")

if __name__ == "__main__":
    plot_reconstruction_loss(
        loss_file='./outputs/plots/ae_loss_history.npy',
        output_image='./outputs/plots/reconstruction_loss_curve.png'
    )
