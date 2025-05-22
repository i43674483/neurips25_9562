
import numpy as np
import matplotlib.pyplot as plt


def paint_spectrum_compare(pred_spectrum, gt_spectrum, save_path=None):

    r = np.linspace(0, 1, 91) # change this depending on your radial distance
    theta = np.linspace(0, 2.0 * np.pi, 361)

    r, theta = np.meshgrid(r, theta)

    fig, axs = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, figsize=(12, 6))

    cax1 = axs[0].pcolormesh(theta, r, np.flipud(pred_spectrum).T, cmap='jet', shading='flat')
    axs[0].axis('off')

    cax2 = axs[1].pcolormesh(theta, r, np.flipud(gt_spectrum).T, cmap='jet', shading='flat')
    axs[1].axis('off')

    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)

    plt.close()

