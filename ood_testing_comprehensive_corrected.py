import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from PIL import Image, ImageEnhance, ImageFilter
import torch
from tqdm import tqdm
import os
import json
import pickle
from typing import Dict, List, Tuple
import warnings
import time
from scipy.ndimage import zoom as scipy_zoom
from skimage.util import random_noise
warnings.filterwarnings('ignore')

from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

torch.set_default_dtype(torch.float32)


class ComprehensiveOODTester:
    """
    OOD Testing with transforms.
    
    Includes:
    - Weather effects (snow, frost, fog)
    - Blur types (gaussian, defocus, motion, zoom)
    - Noise types (gaussian, shot, impulse, speckle)
    - Color transforms (brightness, contrast, saturation, hue)
    - Grayscale
    """
    
    def __init__(self, analyzer, output_dir=None, seed=42, max_neurons=120, n_boot=500):
        self.analyzer = analyzer
        self.seed = seed
        self.max_neurons = max_neurons
        self.n_boot = n_boot
        self.rng = np.random.RandomState(seed)
        
        if output_dir is None:
            output_dir = os.path.join(analyzer.output_dir, 'ood_testing_comprehensive')
        self.output_dir = output_dir
        
        for subdir in ['figures', 'data', 'verification', 'transform_checks']:
            os.makedirs(f"{output_dir}/{subdir}", exist_ok=True)
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.results = {
            'baseline': {},
            'ood_scores': {},
            'retention_metrics': {},
            'metadata': {},
            'split_indices': {},
            'selected_neurons': None
        }
        
        print("\n" + "="*80)
        print("OOD TESTING".center(80))
        print("="*80)
        print(f"Output: {output_dir}")
        print("="*80)
    
    # =========================================================================
    
    # =========================================================================
    # COMPREHENSIVE TRANSFORMS
    # =========================================================================
    
    # COLOR TRANSFORMS
    def _apply_contrast_pil(self, images: np.ndarray, factor: float) -> np.ndarray:
        """Contrast (PIL ImageEnhance)."""
        out = []
        for img in images:
            pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
            enhanced = ImageEnhance.Contrast(pil).enhance(factor)
            out.append(np.array(enhanced, dtype=np.float32) / 255.0)
        return np.stack(out, axis=0).astype(np.float32)
    
    def _apply_brightness_pil(self, images: np.ndarray, factor: float) -> np.ndarray:
        """Brightness (PIL ImageEnhance)."""
        out = []
        for img in images:
            pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
            enhanced = ImageEnhance.Brightness(pil).enhance(factor)
            out.append(np.array(enhanced, dtype=np.float32) / 255.0)
        return np.stack(out, axis=0).astype(np.float32)
    
    def _apply_saturation_pil(self, images: np.ndarray, factor: float) -> np.ndarray:
        """Saturation (PIL ImageEnhance.Color)."""
        out = []
        for img in images:
            pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
            enhanced = ImageEnhance.Color(pil).enhance(factor)
            out.append(np.array(enhanced, dtype=np.float32) / 255.0)
        return np.stack(out, axis=0).astype(np.float32)
    
    def _apply_hue_shift_matplotlib(self, images: np.ndarray, hue_shift: float) -> np.ndarray:
        """Hue shift (matplotlib.colors)."""
        out = []
        for img in images:
            hsv = rgb_to_hsv(img)
            hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 1.0
            rgb = hsv_to_rgb(hsv)
            out.append(np.clip(rgb, 0, 1).astype(np.float32))
        return np.stack(out, axis=0).astype(np.float32)
    
    # BLUR TRANSFORMS
    def _apply_gaussian_blur(self, images: np.ndarray, sigma: float) -> np.ndarray:
        """Gaussian blur (PIL)."""
        out = []
        for img in images:
            img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_uint8)
            blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=sigma))
            out.append(np.array(blurred, dtype=np.float32) / 255.0)
        return np.stack(out, axis=0).astype(np.float32)
    
    def _apply_defocus_blur(self, images: np.ndarray, radius: int) -> np.ndarray:
        """Defocus blur (disk kernel)."""
        out = []
        for img in images:
            # Create disk kernel
            y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
            kernel = x**2 + y**2 <= radius**2
            kernel = kernel.astype(float) / kernel.sum()
            
            # Apply convolution per channel
            from scipy.ndimage import convolve
            blurred = np.zeros_like(img)
            for c in range(3):
                blurred[:, :, c] = convolve(img[:, :, c], kernel, mode='reflect')
            
            out.append(np.clip(blurred, 0, 1).astype(np.float32))
        return np.stack(out, axis=0).astype(np.float32)
    
    def _apply_motion_blur(self, images: np.ndarray, size: int, angle: float) -> np.ndarray:
        """Motion blur (directional)."""
        from scipy.ndimage import convolve
        
        # Create motion blur kernel
        kernel = np.zeros((size, size))
        kernel[size//2, :] = 1.0
        kernel = kernel / kernel.sum()
        
        # Rotate kernel
        from scipy.ndimage import rotate
        kernel = rotate(kernel, angle, reshape=False, order=1)
        kernel = kernel / kernel.sum()
        
        out = []
        for img in images:
            blurred = np.zeros_like(img)
            for c in range(3):
                blurred[:, :, c] = convolve(img[:, :, c], kernel, mode='reflect')
            out.append(np.clip(blurred, 0, 1).astype(np.float32))
        return np.stack(out, axis=0).astype(np.float32)
    
    def _apply_zoom_blur(self, images: np.ndarray, zoom_factor: float) -> np.ndarray:
        """Zoom blur (radial)."""
        out = []
        h, w = images.shape[1:3]
        cy, cx = h // 2, w // 2
        
        for img in images:
            # Create zoomed versions and average
            blurred = np.zeros_like(img, dtype=np.float32)
            n_steps = 10
            for step in range(n_steps):
                alpha = 1.0 + (zoom_factor - 1.0) * step / n_steps
                # Zoom from center
                zoomed = scipy_zoom(img, (alpha, alpha, 1), order=1)
                
                # Crop or pad to original size
                zh, zw = zoomed.shape[:2]
                if zh >= h and zw >= w:
                    # Crop
                    sy = (zh - h) // 2
                    sx = (zw - w) // 2
                    blurred += zoomed[sy:sy+h, sx:sx+w] / n_steps
                else:
                    # Pad
                    pad_h = (h - zh) // 2
                    pad_w = (w - zw) // 2
                    padded = np.pad(zoomed, ((pad_h, h-zh-pad_h), (pad_w, w-zw-pad_w), (0, 0)), mode='edge')
                    blurred += padded[:h, :w] / n_steps
            
            out.append(np.clip(blurred, 0, 1).astype(np.float32))
        return np.stack(out, axis=0).astype(np.float32)
    
    # NOISE TRANSFORMS (COMPREHENSIVE)
    def _apply_gaussian_noise(self, images: np.ndarray, std: float) -> np.ndarray:
        """Gaussian noise."""
        noise = self.rng.randn(*images.shape).astype(np.float32) * std
        return np.clip(images + noise, 0, 1).astype(np.float32)
    
    def _apply_shot_noise(self, images: np.ndarray, scale: float) -> np.ndarray:
        """Shot (Poisson) noise."""
        out = []
        for img in images:
            # Poisson noise: variance = mean * scale
            noisy = self.rng.poisson(img * 255 * scale) / (255 * scale)
            out.append(np.clip(noisy, 0, 1).astype(np.float32))
        return np.stack(out, axis=0).astype(np.float32)
    
    def _apply_impulse_noise(self, images: np.ndarray, amount: float) -> np.ndarray:
        """Impulse (salt-and-pepper) noise."""
        out = []
        for img in images:
            noisy = img.copy()
            # Salt
            salt = self.rng.rand(*img.shape[:2]) < (amount / 2)
            noisy[salt] = 1.0
            # Pepper
            pepper = self.rng.rand(*img.shape[:2]) < (amount / 2)
            noisy[pepper] = 0.0
            out.append(noisy.astype(np.float32))
        return np.stack(out, axis=0).astype(np.float32)
    
    def _apply_speckle_noise(self, images: np.ndarray, std: float) -> np.ndarray:
        """Speckle (multiplicative) noise."""
        noise = self.rng.randn(*images.shape).astype(np.float32) * std
        return np.clip(images * (1 + noise), 0, 1).astype(np.float32)
    
    # WEATHER TRANSFORMS
    def _apply_snow(self, images: np.ndarray, severity: float) -> np.ndarray:
        """Snow effect."""
        out = []
        for img in images:
            h, w = img.shape[:2]
            # Create snow layer
            snow = self.rng.rand(h, w) < (0.1 * severity)
            snow = snow.astype(float)
            
            # Blur snow
            from scipy.ndimage import gaussian_filter
            snow = gaussian_filter(snow, sigma=1.0)
            
            # Apply snow
            snow = snow[:, :, np.newaxis]
            snowy = img * (1 - snow * 0.7) + snow * 0.9
            out.append(np.clip(snowy, 0, 1).astype(np.float32))
        return np.stack(out, axis=0).astype(np.float32)
    
    def _apply_frost(self, images: np.ndarray, severity: float) -> np.ndarray:
        """Frost effect."""
        out = []
        for img in images:
            h, w = img.shape[:2]
            # Create frost pattern
            frost = self.rng.rand(h, w) * severity
            from scipy.ndimage import gaussian_filter
            frost = gaussian_filter(frost, sigma=2.0)
            
            # Apply frost
            frost = frost[:, :, np.newaxis]
            frosty = img * (1 - frost * 0.5) + frost * 0.8
            out.append(np.clip(frosty, 0, 1).astype(np.float32))
        return np.stack(out, axis=0).astype(np.float32)
    
    def _apply_fog(self, images: np.ndarray, severity: float) -> np.ndarray:
        """Fog effect."""
        out = []
        fog_color = np.array([0.8, 0.8, 0.8], dtype=np.float32)
        for img in images:
            # Blend with fog
            foggy = img * (1 - severity) + fog_color * severity
            out.append(np.clip(foggy, 0, 1).astype(np.float32))
        return np.stack(out, axis=0).astype(np.float32)
    
    def _apply_grayscale(self, images: np.ndarray) -> np.ndarray:
        """Grayscale."""
        out = []
        for img in images:
            img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_uint8)
            gray = pil_img.convert('L').convert('RGB')
            out.append(np.array(gray, dtype=np.float32) / 255.0)
        return np.stack(out, axis=0).astype(np.float32)
    
    # =========================================================================
    # VERIFICATION
    # =========================================================================
    
    def _verify_and_save_sample(self, original_images: np.ndarray, 
                                transformed_images: np.ndarray, 
                                name: str) -> float:
        """Verify transform worked."""
        diff = np.abs(original_images - transformed_images).mean()
        
        # Save if suspicious or sample
        if diff < 1e-4 or self.rng.rand() < 0.15:
            n = min(2, len(original_images))
            fig, axes = plt.subplots(n, 2, figsize=(8, 4*n))
            if n == 1:
                axes = axes.reshape(1, -1)
            
            for i in range(n):
                axes[i, 0].imshow(original_images[i])
                axes[i, 0].set_title('Original', fontweight='bold', fontsize=10)
                axes[i, 0].axis('off')
                
                axes[i, 1].imshow(transformed_images[i])
                axes[i, 1].set_title(name, fontweight='bold', fontsize=10)
                axes[i, 1].axis('off')
            
            plt.tight_layout()
            save_path = f"{self.output_dir}/transform_checks/verify_{name.replace('/', '_')}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        if diff < 1e-4:
            print(f"    Warning: {name}: diff={diff:.8f} (SUSPICIOUS)")
        else:
            print(f"    OK: {name}: diff={diff:.6f}")
        
        return diff
    
    # =========================================================================
    # CREATE COMPREHENSIVE OOD DATASETS
    # =========================================================================
    
    def create_ood_datasets(self, images: np.ndarray) -> Dict[str, np.ndarray]:
        """Create OOD datasets."""
        print("\n" + "="*80)
        print("CREATING OOD DATASETS")
        print("="*80)
        
        ood_datasets = {}
        
        # 1. Contrast
        print("\n1. Contrast...")
        for factor in [0.1, 0.3, 0.5, 1.5, 2.5, 4.0]:
            name = f'contrast_{factor}'
            transformed = self._apply_contrast_pil(images, factor)
            diff = self._verify_and_save_sample(images, transformed, name)
            if diff > 1e-4:
                ood_datasets[name] = transformed
        
        # 2. Brightness
        print("\n2. Brightness...")
        for factor in [0.1, 0.3, 0.5, 1.5, 2.5, 4.0]:
            name = f'brightness_{factor}'
            transformed = self._apply_brightness_pil(images, factor)
            diff = self._verify_and_save_sample(images, transformed, name)
            if diff > 1e-4:
                ood_datasets[name] = transformed
        
        # 3. Saturation
        print("\n3. Saturation...")
        for factor in [0.0, 0.2, 0.5, 1.5, 2.5, 4.0]:
            name = f'saturation_{factor}'
            transformed = self._apply_saturation_pil(images, factor)
            diff = self._verify_and_save_sample(images, transformed, name)
            if diff > 1e-4:
                ood_datasets[name] = transformed
        
        # 4. Hue
        print("\n4. Hue...")
        for shift in [-0.5, -0.3, -0.15, 0.15, 0.3, 0.5]:
            name = f'hue_{shift:+.2f}'
            transformed = self._apply_hue_shift_matplotlib(images, shift)
            diff = self._verify_and_save_sample(images, transformed, name)
            if diff > 1e-4:
                ood_datasets[name] = transformed
        
        # 5. GAUSSIAN BLUR
        print("\n5. Gaussian blur...")
        for sigma in [1, 2, 4, 6, 10]:
            name = f'blur_gaussian_s{sigma}'
            transformed = self._apply_gaussian_blur(images, float(sigma))
            diff = self._verify_and_save_sample(images, transformed, name)
            if diff > 1e-4:
                ood_datasets[name] = transformed
        
        # 6. DEFOCUS BLUR
        print("\n6. Defocus blur...")
        for radius in [2, 4, 6]:
            name = f'blur_defocus_r{radius}'
            try:
                transformed = self._apply_defocus_blur(images, radius)
                diff = self._verify_and_save_sample(images, transformed, name)
                if diff > 1e-4:
                    ood_datasets[name] = transformed
            except Exception as e:
                print(f"    ✗ {name}: {e}")
        
        # 7. MOTION BLUR
        print("\n7. Motion blur...")
        for size in [5, 10, 15]:
            for angle in [0, 45]:
                name = f'blur_motion_s{size}_a{angle}'
                try:
                    transformed = self._apply_motion_blur(images, size, angle)
                    diff = self._verify_and_save_sample(images, transformed, name)
                    if diff > 1e-4:
                        ood_datasets[name] = transformed
                except Exception as e:
                    print(f"    ✗ {name}: {e}")
        
        # 8. ZOOM BLUR
        print("\n8. Zoom blur...")
        for zoom in [1.1, 1.2, 1.3]:
            name = f'blur_zoom_{zoom}'
            try:
                transformed = self._apply_zoom_blur(images, zoom)
                diff = self._verify_and_save_sample(images, transformed, name)
                if diff > 1e-4:
                    ood_datasets[name] = transformed
            except Exception as e:
                print(f"    ✗ {name}: {e}")
        
        # 9. GAUSSIAN NOISE
        print("\n9. Gaussian noise...")
        for std in [0.02, 0.05, 0.10, 0.15, 0.25]:
            name = f'noise_gaussian_{std:.2f}'
            transformed = self._apply_gaussian_noise(images, std)
            diff = self._verify_and_save_sample(images, transformed, name)
            if diff > 1e-4:
                ood_datasets[name] = transformed
        
        # 10. SHOT NOISE
        print("\n10. Shot noise...")
        for scale in [0.5, 1.0, 2.0]:
            name = f'noise_shot_{scale}'
            try:
                transformed = self._apply_shot_noise(images, scale)
                diff = self._verify_and_save_sample(images, transformed, name)
                if diff > 1e-4:
                    ood_datasets[name] = transformed
            except Exception as e:
                print(f"    ✗ {name}: {e}")
        
        # 11. IMPULSE NOISE
        print("\n11. Impulse noise...")
        for amount in [0.01, 0.03, 0.05]:
            name = f'noise_impulse_{amount:.2f}'
            transformed = self._apply_impulse_noise(images, amount)
            diff = self._verify_and_save_sample(images, transformed, name)
            if diff > 1e-4:
                ood_datasets[name] = transformed
        
        # 12. SPECKLE NOISE
        print("\n12. Speckle noise...")
        for std in [0.05, 0.10, 0.20]:
            name = f'noise_speckle_{std:.2f}'
            transformed = self._apply_speckle_noise(images, std)
            diff = self._verify_and_save_sample(images, transformed, name)
            if diff > 1e-4:
                ood_datasets[name] = transformed
        
        # 13. SNOW
        print("\n13. Snow...")
        for severity in [0.5, 1.0, 2.0]:
            name = f'weather_snow_{severity}'
            try:
                transformed = self._apply_snow(images, severity)
                diff = self._verify_and_save_sample(images, transformed, name)
                if diff > 1e-4:
                    ood_datasets[name] = transformed
            except Exception as e:
                print(f"    ✗ {name}: {e}")
        
        # 14. FROST
        print("\n14. Frost...")
        for severity in [0.3, 0.5, 0.8]:
            name = f'weather_frost_{severity}'
            try:
                transformed = self._apply_frost(images, severity)
                diff = self._verify_and_save_sample(images, transformed, name)
                if diff > 1e-4:
                    ood_datasets[name] = transformed
            except Exception as e:
                print(f"    ✗ {name}: {e}")
        
        # 15. FOG
        print("\n15. Fog...")
        for severity in [0.3, 0.5, 0.7]:
            name = f'weather_fog_{severity}'
            transformed = self._apply_fog(images, severity)
            diff = self._verify_and_save_sample(images, transformed, name)
            if diff > 1e-4:
                ood_datasets[name] = transformed
        
        # 16. GRAYSCALE
        print("\n16. Grayscale...")
        transformed = self._apply_grayscale(images)
        diff = self._verify_and_save_sample(images, transformed, 'grayscale')
        if diff > 1e-4:
            ood_datasets['grayscale'] = transformed
        
        print(f"\nCreated {len(ood_datasets)} valid OOD datasets")
        print("  (Transforms: blur, noise, weather, color)")
        
        self._create_verification_grid(images, ood_datasets)
        
        return ood_datasets
    
    def _create_verification_grid(self, images_id: np.ndarray, ood_datasets: Dict):
        """Create overview grid."""
        print("\nCreating verification grid...")
        
        n_samples = min(3, len(images_id))
        sample_indices = self.rng.choice(len(images_id), n_samples, replace=False)
        
        # Select representative conditions (mix of types)
        conditions = sorted(ood_datasets.keys())[:16]
        
        n_conditions = len(conditions)
        n_cols = min(5, n_conditions + 1)
        n_rows = n_samples * ((n_conditions + n_cols - 1) // (n_cols - 1))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        axes = axes.flatten()
        
        plot_idx = 0
        for row in range(n_samples):
            idx = sample_indices[row]
            
            # Original
            axes[plot_idx].imshow(images_id[idx])
            axes[plot_idx].set_title('Original', fontweight='bold', fontsize=9)
            axes[plot_idx].axis('off')
            plot_idx += 1
            
            # Transforms for this sample
            for condition in conditions[:n_cols-1]:
                if condition in ood_datasets:
                    axes[plot_idx].imshow(ood_datasets[condition][idx])
                    axes[plot_idx].set_title(condition[:20].replace('_', '\n'), fontsize=7)
                    axes[plot_idx].axis('off')
                plot_idx += 1
        
        # Hide unused
        for i in range(plot_idx, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        save_path = f"{self.output_dir}/verification/comprehensive_overview.png"
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
    
    # =========================================================================
    # FEATURE EXTRACTION, ENCODING MODELS, RETENTION (same as before)
    # =========================================================================
    
    def _extract_features_no_cache(self, images: np.ndarray, model_name: str) -> np.ndarray:
        """Extract features."""
        if len(images) == 0:
            raise ValueError("Empty images array")
        
        extractor = self.analyzer.feature_extractors[model_name]
        extractor.eval()
        device = self.analyzer.device
        
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device).view(1,3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device).view(1,3,1,1)
        
        batch_size = self.analyzer.batch_size
        features_list = []
        
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i+batch_size]
                batch_t = torch.from_numpy(batch).permute(0, 3, 1, 2)
                batch_t = batch_t.to(dtype=torch.float32, device=device)
                batch_norm = (batch_t - mean) / std
                
                out = extractor(batch_norm)
                
                if isinstance(out, torch.Tensor):
                    feats = out
                elif isinstance(out, (tuple, list)):
                    feats = out[0]
                elif isinstance(out, dict):
                    feats = list(out.values())[0]
                else:
                    raise RuntimeError(f"Unexpected output type: {type(out)}")
                
                feats_np = feats.detach().cpu().numpy()
                if feats_np.ndim > 2:
                    feats_np = feats_np.reshape(feats_np.shape[0], -1)
                
                if feats_np.shape[0] != len(batch):
                    raise RuntimeError(f"Feature batch size mismatch")
                
                features_list.extend(list(feats_np))
        
        if len(features_list) == 0:
            raise RuntimeError("No features extracted")
        
        return np.vstack(features_list).astype(np.float32)
    
    def _train_encoding_models(self, features_train, neural_train):
        """Train encoding models."""
        n_neurons = neural_train.shape[1]
        models = []
        scalers = []
        
        alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
        
        for neuron_idx in range(n_neurons):
            y = neural_train[:, neuron_idx]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(features_train)
            ridge = RidgeCV(alphas=alphas, cv=3)
            ridge.fit(X_scaled, y)
            models.append(ridge)
            scalers.append(scaler)
        
        return models, scalers
    
    def _predict_with_models(self, features_test, models, scalers):
        """Predict."""
        n_test = features_test.shape[0]
        n_neurons = len(models)
        predictions = np.zeros((n_test, n_neurons), dtype=np.float32)
        
        for neuron_idx in range(n_neurons):
            X_scaled = scalers[neuron_idx].transform(features_test)
            predictions[:, neuron_idx] = models[neuron_idx].predict(X_scaled)
        
        return predictions
    
    def _compute_scores(self, neural_true, neural_pred):
        """Compute scores."""
        n_neurons = neural_true.shape[1]
        scores = np.zeros(n_neurons, dtype=np.float32)
        
        for i in range(n_neurons):
            true_i = neural_true[:, i]
            pred_i = neural_pred[:, i]
            
            if len(np.unique(true_i)) > 1 and len(np.unique(pred_i)) > 1:
                try:
                    r, _ = pearsonr(true_i, pred_i)
                    if not np.isnan(r) and not np.isinf(r):
                        scores[i] = r
                    else:
                        scores[i] = 0.0
                except:
                    scores[i] = 0.0
            else:
                scores[i] = 0.0
        
        return scores
    
    def _compute_retention_metrics(self, baseline_scores: np.ndarray, 
                                   ood_scores: np.ndarray) -> Dict:
        """Compute retention with bootstrap CIs."""
        n_neurons = len(baseline_scores)
        
        retention = np.zeros(n_neurons, dtype=np.float32)
        threshold = 0.01
        
        for i in range(n_neurons):
            baseline_i = baseline_scores[i]
            ood_i = ood_scores[i]
            
            if baseline_i > threshold:
                retention[i] = ood_i / baseline_i
            else:
                retention[i] = np.nan
        
        valid_mask = ~np.isnan(retention)
        n_valid = np.sum(valid_mask)
        
        if n_valid == 0:
            return {
                'median_retention_percent': np.nan,
                'mean_retention_percent': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'n_valid': 0,
                'n_total': n_neurons,
                'median_absolute_diff': float(np.median(ood_scores - baseline_scores))
            }
        
        valid_retention = retention[valid_mask]
        absolute_diff = ood_scores - baseline_scores
        
        # Bootstrap CIs
        boot_medians = []
        m = len(valid_retention)
        for _ in range(self.n_boot):
            idxs = self.rng.randint(0, m, size=m)
            boot_sample = valid_retention[idxs]
            boot_medians.append(np.median(boot_sample))
        
        ci_lower = float(np.percentile(boot_medians, 2.5))
        ci_upper = float(np.percentile(boot_medians, 97.5))
        
        return {
            'median_retention_percent': float(np.median(valid_retention) * 100),
            'mean_retention_percent': float(np.mean(valid_retention) * 100),
            'std_retention_percent': float(np.std(valid_retention) * 100),
            'ci_lower': ci_lower * 100,
            'ci_upper': ci_upper * 100,
            'n_valid': int(n_valid),
            'n_total': int(n_neurons),
            'median_absolute_diff': float(np.median(absolute_diff)),
            'mean_absolute_diff': float(np.mean(absolute_diff))
        }
    
    # =========================================================================
    # MAIN PIPELINE (same structure, calls comprehensive transforms)
    # =========================================================================
    
    def run_ood_test(self, test_ratio=0.2):
        """Run comprehensive OOD test."""
        print("\n" + "="*80)
        print("RUNNING OOD TEST")
        print("="*80)
        print(f"\nTest ratio: {test_ratio}")
        print(f"Bootstrap iterations: {self.n_boot}")
        print("="*80)
        
        # Get data
        images_id = self.analyzer.neural_data['images']
        neural_responses = self.analyzer.neural_data['responses']
        n_images = len(images_id)
        n_neurons_original = neural_responses.shape[1]
        
        print(f"\nOriginal data: {n_images} images × {n_neurons_original} neurons")
        
        # Subsample neurons
        if n_neurons_original > self.max_neurons:
            print(f"\nWARNING: Subsampling to {self.max_neurons} neurons by variance")
            neuron_var = np.var(neural_responses, axis=0)
            sel = np.argsort(neuron_var)[-self.max_neurons:]
            neural_responses = neural_responses[:, sel]
            n_neurons = self.max_neurons
            self.results['selected_neurons'] = sel.tolist()
            print(f"Selected neurons: {sel[:5].tolist()}... (first 5)")
        else:
            n_neurons = n_neurons_original
            self.results['selected_neurons'] = list(range(n_neurons))
        
        # Split
        indices = np.arange(n_images)
        n_test_requested = int(n_images * test_ratio)
        n_train_requested = n_images - n_test_requested
        
        print(f"\nRequested split: {n_train_requested} train / {n_test_requested} test")
        
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_ratio,
            random_state=self.seed,
            shuffle=True
        )
        
        n_train = len(train_idx)
        n_test = len(test_idx)
        
        print(f"Actual split: {n_train} train / {n_test} test")
        
        if n_test < 20:
            raise ValueError(f"Test set too small: {n_test} < 20")
        
        images_train = images_id[train_idx]
        images_test_id = images_id[test_idx]
        neural_train = neural_responses[train_idx]
        neural_test = neural_responses[test_idx]
        
        # Save split
        split_file = f"{self.output_dir}/data/split_indices.json"
        with open(split_file, 'w') as f:
            json.dump({
                'train_idx': train_idx.tolist(),
                'test_idx': test_idx.tolist(),
                'n_train': n_train,
                'n_test': n_test,
                'test_ratio': test_ratio,
                'seed': self.seed,
                'n_boot': self.n_boot,
                'selected_neurons': self.results['selected_neurons'],
                'methodology': 'Comprehensive OOD with bootstrap CIs'
            }, f, indent=2)
        print(f"Saved split: {split_file}")
        
        self.results['metadata'] = {
            'n_train': n_train,
            'n_test': n_test,
            'n_neurons': n_neurons,
            'n_neurons_original': n_neurons_original,
            'test_ratio': test_ratio,
            'seed': self.seed,
            'n_boot': self.n_boot,
            'max_neurons': self.max_neurons
        }
        self.results['split_indices'] = {
            'train_idx': train_idx.tolist(),
            'test_idx': test_idx.tolist()
        }
        
        # TRAIN
        print("\n" + "="*80)
        print("STEP 1: TRAINING ENCODING MODELS")
        print("="*80)
        
        print("\nExtracting training features...")
        start_time = time.time()
        vit_train = self._extract_features_no_cache(images_train, 'vit')
        resnet_train = self._extract_features_no_cache(images_train, 'resnet')
        print(f"  ViT: {vit_train.shape} ({time.time()-start_time:.1f}s)")
        print(f"  ResNet: {resnet_train.shape}")
        
        print(f"\nTraining {n_neurons} models per architecture...")
        start_time = time.time()
        vit_models, vit_scalers = self._train_encoding_models(vit_train, neural_train)
        resnet_models, resnet_scalers = self._train_encoding_models(resnet_train, neural_train)
        print(f"Trained in {time.time()-start_time:.1f}s")
        
        # BASELINE
        print("\n" + "="*80)
        print("STEP 2: BASELINE (ID TEST)")
        print("="*80)
        
        print("\nExtracting ID test features...")
        vit_test_id = self._extract_features_no_cache(images_test_id, 'vit')
        resnet_test_id = self._extract_features_no_cache(images_test_id, 'resnet')
        
        print("\nPredicting...")
        vit_pred_id = self._predict_with_models(vit_test_id, vit_models, vit_scalers)
        resnet_pred_id = self._predict_with_models(resnet_test_id, resnet_models, resnet_scalers)
        
        vit_baseline_scores = self._compute_scores(neural_test, vit_pred_id)
        resnet_baseline_scores = self._compute_scores(neural_test, resnet_pred_id)
        
        vit_baseline_median = float(np.median(vit_baseline_scores))
        resnet_baseline_median = float(np.median(resnet_baseline_scores))
        
        vit_positive = np.sum(vit_baseline_scores > 0.01)
        resnet_positive = np.sum(resnet_baseline_scores > 0.01)
        
        print(f"\n{'='*80}")
        print("BASELINE RESULTS")
        print(f"{'='*80}")
        print(f"ViT:    median r = {vit_baseline_median:.4f} ({vit_positive}/{n_neurons} neurons > 0.01)")
        print(f"ResNet: median r = {resnet_baseline_median:.4f} ({resnet_positive}/{n_neurons} neurons > 0.01)")
        print(f"{'='*80}")
        
        self.results['baseline'] = {
            'vit': {
                'scores': vit_baseline_scores,
                'median': vit_baseline_median,
                'n_positive': int(vit_positive)
            },
            'resnet': {
                'scores': resnet_baseline_scores,
                'median': resnet_baseline_median,
                'n_positive': int(resnet_positive)
            }
        }
        
        # CREATE OOD
        ood_datasets = self.create_ood_datasets(images_test_id)
        
        # TEST OOD
        print("\n" + "="*80)
        print("STEP 4: TESTING ON OOD DATASETS")
        print("="*80)
        
        for ood_name, ood_images in tqdm(ood_datasets.items(), desc="OOD conditions"):
            vit_feat_ood = self._extract_features_no_cache(ood_images, 'vit')
            resnet_feat_ood = self._extract_features_no_cache(ood_images, 'resnet')
            
            vit_pred_ood = self._predict_with_models(vit_feat_ood, vit_models, vit_scalers)
            resnet_pred_ood = self._predict_with_models(resnet_feat_ood, resnet_models, resnet_scalers)
            
            vit_ood_scores = self._compute_scores(neural_test, vit_pred_ood)
            resnet_ood_scores = self._compute_scores(neural_test, resnet_pred_ood)
            
            vit_metrics = self._compute_retention_metrics(vit_baseline_scores, vit_ood_scores)
            resnet_metrics = self._compute_retention_metrics(resnet_baseline_scores, resnet_ood_scores)
            
            self.results['ood_scores'][ood_name] = {
                'vit': {
                    'scores': vit_ood_scores,
                    'median': float(np.median(vit_ood_scores))
                },
                'resnet': {
                    'scores': resnet_ood_scores,
                    'median': float(np.median(resnet_ood_scores))
                }
            }
            
            self.results['retention_metrics'][ood_name] = {
                'vit': vit_metrics,
                'resnet': resnet_metrics
            }
        
        # RESULTS
        self._print_results()
        self._create_visualizations()
        self._save_results()
        
        print("\n" + "="*80)
        print("OOD TESTING COMPLETE")
        print(f"Results: {self.output_dir}/")
        print("="*80)
        
        return self.results
    
    def _print_results(self):
        """Print results."""
        print("\n" + "="*80)
        print("RESULTS (with Bootstrap 95% CIs)")
        print("="*80)
        
        vit_baseline = self.results['baseline']['vit']['median']
        resnet_baseline = self.results['baseline']['resnet']['median']
        
        print(f"\nBASELINE:")
        print(f"  ViT:    {vit_baseline:.4f}")
        print(f"  ResNet: {resnet_baseline:.4f}")
        
        print(f"\n{'='*80}")
        print("OOD RESULTS")
        print(f"{'='*80}")
        print(f"{'Transform':<30} {'ViT %':<20} {'ResNet %':<20}")
        print("-"*80)
        
        for ood_name in sorted(self.results['retention_metrics'].keys()):
            vit_ret = self.results['retention_metrics'][ood_name]['vit']['median_retention_percent']
            vit_ci_l = self.results['retention_metrics'][ood_name]['vit']['ci_lower']
            vit_ci_u = self.results['retention_metrics'][ood_name]['vit']['ci_upper']
            
            resnet_ret = self.results['retention_metrics'][ood_name]['resnet']['median_retention_percent']
            resnet_ci_l = self.results['retention_metrics'][ood_name]['resnet']['ci_lower']
            resnet_ci_u = self.results['retention_metrics'][ood_name]['resnet']['ci_upper']
            
            if not np.isnan(vit_ret):
                vit_str = f"{vit_ret:.1f}[{vit_ci_l:.1f},{vit_ci_u:.1f}]"
                resnet_str = f"{resnet_ret:.1f}[{resnet_ci_l:.1f},{resnet_ci_u:.1f}]"
            else:
                vit_str = "N/A"
                resnet_str = "N/A"
            
            print(f"{ood_name:<30} {vit_str:<20} {resnet_str:<20}")
        
        print(f"{'='*80}")
    
    def _create_visualizations(self):
        """Create visualizations."""
        print("\nCreating visualizations...")
        
        transforms = sorted(self.results['retention_metrics'].keys())
        
        vit_ret = [self.results['retention_metrics'][t]['vit']['median_retention_percent'] for t in transforms]
        resnet_ret = [self.results['retention_metrics'][t]['resnet']['median_retention_percent'] for t in transforms]
        
        vit_ci_l = [self.results['retention_metrics'][t]['vit']['ci_lower'] for t in transforms]
        vit_ci_u = [self.results['retention_metrics'][t]['vit']['ci_upper'] for t in transforms]
        resnet_ci_l = [self.results['retention_metrics'][t]['resnet']['ci_lower'] for t in transforms]
        resnet_ci_u = [self.results['retention_metrics'][t]['resnet']['ci_upper'] for t in transforms]
        
        vit_ret = [0 if np.isnan(x) else x for x in vit_ret]
        resnet_ret = [0 if np.isnan(x) else x for x in resnet_ret]
        
        vit_ci_l = [vit_ret[i] if np.isnan(vit_ci_l[i]) else vit_ci_l[i] for i in range(len(transforms))]
        vit_ci_u = [vit_ret[i] if np.isnan(vit_ci_u[i]) else vit_ci_u[i] for i in range(len(transforms))]
        resnet_ci_l = [resnet_ret[i] if np.isnan(resnet_ci_l[i]) else resnet_ci_l[i] for i in range(len(transforms))]
        resnet_ci_u = [resnet_ret[i] if np.isnan(resnet_ci_u[i]) else resnet_ci_u[i] for i in range(len(transforms))]
        
        vit_err_l = [vit_ret[i] - vit_ci_l[i] for i in range(len(transforms))]
        vit_err_u = [vit_ci_u[i] - vit_ret[i] for i in range(len(transforms))]
        resnet_err_l = [resnet_ret[i] - resnet_ci_l[i] for i in range(len(transforms))]
        resnet_err_u = [resnet_ci_u[i] - resnet_ret[i] for i in range(len(transforms))]
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        ax = axes[0]
        x = np.arange(len(transforms))
        w = 0.35
        
        vit_yerr = np.array([vit_err_l, vit_err_u])
        resnet_yerr = np.array([resnet_err_l, resnet_err_u])
        
        ax.bar(x - w/2, vit_ret, w, label='ViT', color='#2ecc71', alpha=0.8,
               yerr=vit_yerr, capsize=2)
        ax.bar(x + w/2, resnet_ret, w, label='ResNet', color='#e74c3c', alpha=0.8,
               yerr=resnet_yerr, capsize=2)
        ax.axhline(y=100, color='gray', linestyle='--', linewidth=2)
        ax.set_xlabel('OOD Transform', fontsize=11)
        ax.set_ylabel('Retention (%) with 95% CI', fontsize=11)
        ax.set_title('Comprehensive OOD Retention', fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([t[:20] for t in transforms], rotation=90, ha='center', fontsize=6)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        ax = axes[1]
        vit_r = [self.results['ood_scores'][t]['vit']['median'] for t in transforms]
        resnet_r = [self.results['ood_scores'][t]['resnet']['median'] for t in transforms]
        ax.scatter(vit_r, resnet_r, s=80, alpha=0.6, c='#3498db')
        max_val = max(max(vit_r), max(resnet_r))
        min_val = min(min(vit_r), min(resnet_r))
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Identity')
        ax.set_xlabel('ViT r', fontsize=12)
        ax.set_ylabel('ResNet r', fontsize=12)
        ax.set_title('Absolute r Comparison', fontweight='bold', fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        save_path = f"{self.output_dir}/figures/comprehensive_ood_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
    
    def _save_results(self):
        """Save results."""
        print("\nSaving results...")
        
        summary = {
            'baseline': {
                'vit_r': self.results['baseline']['vit']['median'],
                'resnet_r': self.results['baseline']['resnet']['median']
            },
            'metadata': self.results['metadata'],
            'methodology': 'Comprehensive OOD with bootstrap CIs',
            'ood_results': {}
        }
        
        for ood_name in self.results['retention_metrics'].keys():
            summary['ood_results'][ood_name] = {
                'vit_r': self.results['ood_scores'][ood_name]['vit']['median'],
                'resnet_r': self.results['ood_scores'][ood_name]['resnet']['median'],
                'vit_retention': self.results['retention_metrics'][ood_name]['vit']['median_retention_percent'],
                'resnet_retention': self.results['retention_metrics'][ood_name]['resnet']['median_retention_percent'],
                'vit_ci': [
                    self.results['retention_metrics'][ood_name]['vit']['ci_lower'],
                    self.results['retention_metrics'][ood_name]['vit']['ci_upper']
                ],
                'resnet_ci': [
                    self.results['retention_metrics'][ood_name]['resnet']['ci_lower'],
                    self.results['retention_metrics'][ood_name]['resnet']['ci_upper']
                ],
                'n_valid': self.results['retention_metrics'][ood_name]['vit']['n_valid']
            }
        
        with open(f"{self.output_dir}/summary_comprehensive.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        with open(f"{self.output_dir}/data/full_results.pkl", 'wb') as f:
            pickle.dump(self.results, f)
        
        print(f"Saved results")


def main():
    print("\nOOD testing with grayscale images.")


if __name__ == "__main__":
    main()
