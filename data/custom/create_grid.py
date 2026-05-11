import matplotlib.pyplot as plt
from PIL import Image
import os

def create_comparison_grid():

    script_dir = os.path.dirname(os.path.abspath(__file__))
    img_folder = os.path.join(script_dir, 'img_comparison')
    save_path = os.path.join(script_dir, 'comparison_grid.png')

    fig, axes = plt.subplots(3, 5, figsize=(20, 12))

    for i in range(1, 6):
        before_path = os.path.join(img_folder, f'img{i}_before.jpg')
        deeplpf_path = os.path.join(img_folder, f'img{i}_deeplpf.jpg')
        retinex_path = os.path.join(img_folder, f'img{i}_retinex.jpg')
        
        col_idx = i - 1
        
        if os.path.exists(before_path):
            img_before = Image.open(before_path)
            axes[0, col_idx].imshow(img_before)
            axes[0, col_idx].set_title(f'Image {i}', fontsize=16, pad=10)
        
        if os.path.exists(deeplpf_path):
            img_deeplpf = Image.open(deeplpf_path)
            axes[1, col_idx].imshow(img_deeplpf)

        if os.path.exists(retinex_path):
            img_retinex = Image.open(retinex_path)
            axes[2, col_idx].imshow(img_retinex)

        for row in range(3):
            axes[row, col_idx].set_xticks([])
            axes[row, col_idx].set_yticks([])
            for spine in axes[row, col_idx].spines.values():
                spine.set_visible(False)

        if col_idx == 0:
            axes[0, col_idx].set_ylabel('Before', fontsize=18, labelpad=20)
            axes[1, col_idx].set_ylabel('Preprocessed with DeepLPF', fontsize=18, labelpad=20)
            axes[2, col_idx].set_ylabel('Preprocessed with Retinex', fontsize=18, labelpad=20)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Comparison grid saved successfully to:\n{save_path}")

if __name__ == '__main__':
    create_comparison_grid()
