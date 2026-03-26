import argparse
import os
import cv2
import numpy as np
import pandas as pd
import torch

from train_retinex import HybridRetinexFuzzyModel

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

def run_inference(
    model_path, csv_path, image_col, scale_col, output_dir=None, max_samples=0
):
    print(f"Loading pretrained inference model from: {model_path}")
    print(f"Target underlying hardware device context: {device}")

    model_name = os.path.splitext(os.path.basename(model_path))[0]
    if output_dir is None:
        output_dir = os.path.join("results", model_name)

    os.makedirs(output_dir, exist_ok=True)

    model = HybridRetinexFuzzyModel()

    if not os.path.exists(model_path):
        print(
            f"Error Context: Target compiled PyTorch weights file '{model_path}' missing."
        )
        return

    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()

    print(f"Querying inputs from {csv_path}...")
    df = pd.read_csv(csv_path)

    if image_col not in df.columns or scale_col not in df.columns:
        print(
            f"Error Context: Required positional columns '{image_col}' or '{scale_col}' unavailable."
        )
        print(f"Available structural DataFrame keys: {df.columns.tolist()}")
        return

    if max_samples > 0:
        df = df.head(max_samples)

    print(f"Commencing isolated image generation. Emitting natively to: {output_dir}/")

    count = 0
    with torch.no_grad():
        for idx, row in df.iterrows():
            img_path = str(row[image_col])
            f_scale = float(row[scale_col])

            if not os.path.exists(img_path):
                print(
                    f"Notice: Broken Image URL Context at {img_path}, skipping sample automatically."
                )
                continue

            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue

            orig_h, orig_w = img_bgr.shape[:2]

            img_bgr = cv2.resize(img_bgr, (256, 256))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            img_tensor = img_rgb.transpose((2, 0, 1)).astype(np.float32) / 255.0

            s_norm = (f_scale - 1.0) / 5.0
            s_norm = max(0.0, min(1.0, s_norm))

            t_img = torch.tensor(img_tensor).unsqueeze(0).to(device)
            s_val = torch.tensor([s_norm], dtype=torch.float32).unsqueeze(0).to(device)

            output, _ = model(t_img, s_val)

            out_tensor = output.squeeze(0).cpu().numpy()
            out_tensor = np.clip(out_tensor, 0.0, 1.0)

            out_rgb = (out_tensor.transpose((1, 2, 0)) * 255.0).astype(np.uint8)
            out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

            out_bgr = cv2.resize(out_bgr, (orig_w, orig_h))

            orig_filename = os.path.basename(img_path)
            save_path = os.path.join(output_dir, orig_filename)
            cv2.imwrite(save_path, out_bgr)

            count += 1
            if count % 10 == 0:
                print(
                    f"Generated successfully isolated frames: {count}/{len(df)} ...",
                    end="\r",
                )

    print(
        f"\nGenerative Image Evaluation Complete. Total mapping output ({count} images) validated safely at context: {output_dir}"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Image Level Model Inference Architecture Wrapper"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/hybrid_retinex.pth",
        help="Compiled Weight Configurations File context path",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="data/labels.csv",
        help="Metadata list referencing structural requirements paths targeting variables",
    )
    parser.add_argument(
        "--image_col",
        type=str,
        default="training_image",
        help="String Key corresponding directly against structural paths to base inputs",
    )
    parser.add_argument(
        "--scale_col",
        type=str,
        default="Fitzpatrick scale",
        help="String constraint enforcing expected target scale mapping logic",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Force mapping variables toward strict location context bypass fallback",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Enforce limit constraint directly on pipeline representation generation queue",
    )

    args = parser.parse_args()

    run_inference(
        args.model_path,
        args.csv_path,
        args.image_col,
        args.scale_col,
        args.output_dir,
        args.max_samples,
    )
