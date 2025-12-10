import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, ttest_rel, wilcoxon
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import os
import json
import pickle
from typing import Dict, List, Tuple, Optional
import sys

# Set PyTorch default dtype
torch.set_default_dtype(torch.float32)

# Allen SDK
try:
    from allensdk.core.brain_observatory_cache import BrainObservatoryCache
    import allensdk
    ALLEN_AVAILABLE = True
except ImportError:
    print("AllenSDK not available. Try fixing with: pip install allensdk")
    ALLEN_AVAILABLE = False


def choose_best_allen_experiment(boc, targeted_structures: List[str], 
                                 stimuli: List[str] = ['natural_scenes'],
                                 auto_pick: bool = True):
    """
    AllenSDK API usage with fallback.
    """
    print(f"\n{'='*80}")
    print("ALLEN EXPERIMENT SELECTION")
    print(f"{'='*80}")
    print(f"Searching for: {targeted_structures} with stimuli {stimuli}")
    
    experiments = []
    
    # Try correct API first
    try:
        experiments = boc.get_ophys_experiments(
            targeted_structures=targeted_structures,
            stimuli=stimuli
        )
        print(f"Found {len(experiments)} experiments")
        
    except TypeError as e:
        # Fallback for older SDK
        print(f"Primary query failed, trying fallback...")
        containers = boc.get_experiment_containers(targeted_structures=targeted_structures)
        experiments = []
        for c in containers:
            try:
                exps = boc.get_ophys_experiments(
                    experiment_container_ids=[c['id']],
                    stimuli=stimuli
                )
                if exps:
                    experiments.extend(exps)
            except Exception:
                continue
        print(f"Found {len(experiments)} experiments via fallback")
    
    if not experiments:
        raise ValueError(f"No experiments found for {targeted_structures} with {stimuli}")
    
    if len(experiments) == 1:
        return experiments[0]
    
    # Score by quality
    print(f"\nScoring {len(experiments)} experiments by quality...")
    best_exp = None
    best_score = -1
    
    for exp in experiments:
        exp_id = exp['id']
        score = 0
        
        try:
            data_set = boc.get_ophys_experiment_data(exp_id)
            
            # Score by cell count
            try:
                n_cells = len(data_set.get_cell_specimen_ids())
                score += n_cells * 2
            except:
                n_cells = 0
            
            # Score by presentations
            try:
                stim_table = data_set.get_stimulus_table('natural_scenes')
                n_pres = len(stim_table)
                score += n_pres
            except:
                n_pres = 0
            
            # Score by data quality
            try:
                dff = data_set.get_dff_traces()
                if dff is not None:
                    score += 10
            except:
                pass
            
            print(f"  Exp {exp_id}: cells={n_cells}, pres={n_pres}, score={score}")
            
        except Exception as e:
            print(f"  Exp {exp_id}: probe failed ({e})")
            score = 0
        
        if score > best_score:
            best_score = score
            best_exp = exp
    
    if best_exp is None:
        best_exp = experiments[0]
    
    print(f"\nSelected experiment {best_exp['id']} (score={best_score})")
    print(f"{'='*80}\n")
    
    return best_exp


class ProperNeuralAlignmentAnalyzer:
    
    def __init__(self, 
                 output_dir: str = './analysis_results',
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 batch_size: int = 16,
                 seed: int = 42):
        """Initialize analyzer with production settings."""
        self.output_dir = output_dir
        self.device = device
        self.batch_size = batch_size
        self.seed = seed
        
        # Set seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # Create directories
        for subdir in ['figures', 'data', 'neural_data']:
            os.makedirs(f"{output_dir}/{subdir}", exist_ok=True)
        
        # ImageNet normalization constants (stored as attributes)
        self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        self.imagenet_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
        
        self.models = {}
        self.feature_extractors = {}
        self.results = {
            'rdm_correlations': {},
            'neural_predictivity': {},
            'ood_predictivity': {},
            'noise_ceilings': {},
            'statistical_tests': {}
        }
        
        print("\n" + "="*80)
        print("NEURAL ALIGNMENT ANALYZER".center(80))
        print("="*80)
        print(f"Output directory: {output_dir}")
        print(f"Device: {device}")
        print(f"Batch size: {batch_size}")
        print(f"PyTorch dtype: {torch.get_default_dtype()}")
        print(f"Random seed: {seed}")
        print("="*80)
    
    def load_models(self, vit_checkpoint: Optional[str] = None, 
                   resnet_checkpoint: Optional[str] = None):
        """Load models with verification."""
        print("\n" + "="*80)
        print("LOADING MODELS")
        print("="*80)
        
        # Load ViT
        print("\nLoading Vision Transformer (ViT-B/16)...")
        try:
            self.models['vit'] = models.vit_b_16(weights='IMAGENET1K_V1')
            if vit_checkpoint:
                print(f"Loading ViT checkpoint from {vit_checkpoint}")
                checkpoint = torch.load(vit_checkpoint, map_location=self.device)
                self.models['vit'].load_state_dict(
                    checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
                )
                print("Loaded ViT from checkpoint")
            else:
                print("Using ImageNet pretrained ViT")
        except Exception as e:
            raise RuntimeError(f"Failed to load ViT: {e}")
        
        # Load ResNet
        print("\nLoading ResNet-34...")
        try:
            self.models['resnet'] = models.resnet34(weights='IMAGENET1K_V1')
            if resnet_checkpoint:
                print(f"Loading ResNet checkpoint from {resnet_checkpoint}")
                checkpoint = torch.load(resnet_checkpoint, map_location=self.device)
                self.models['resnet'].load_state_dict(
                    checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
                )
                print("Loaded ResNet from checkpoint")
            else:
                print("Using ImageNet pretrained ResNet")
        except Exception as e:
            raise RuntimeError(f"Failed to load ResNet: {e}")
        
        # Move to device and eval mode
        for name, model in self.models.items():
            model = model.to(self.device)
            model.eval()
            param_dtype = next(model.parameters()).dtype
            print(f"  {name.upper()} parameters dtype: {param_dtype}")
            self.models[name] = model
        
        self._setup_feature_extractors()
        print("\nModels loaded and ready")
    
    def _setup_feature_extractors(self):
        """
        Setup feature extraction
        Uses public API where possible, defensive dtype handling.
        """
        
        class RobustViTFeatureExtractor(nn.Module):
            """
            Ensures all internal tensors (pos_embedding, class_token) are on correct device/dtype
            Defensive dtype checking
            """
            def __init__(self, model, device):
                super().__init__()
                self.model = model
                self.device = device
                
                # Move internal tensors to device with correct dtype
                if hasattr(self.model, 'class_token'):
                    self.model.class_token = self.model.class_token.to(device=device, dtype=torch.float32)
                
                if hasattr(self.model, 'encoder'):
                    if hasattr(self.model.encoder, 'pos_embedding'):
                        self.model.encoder.pos_embedding = self.model.encoder.pos_embedding.to(
                            device=device, dtype=torch.float32
                        )
            
            def forward(self, x):
                # Defensive dtype check
                if x.dtype != torch.float32:
                    x = x.to(dtype=torch.float32)
                
                # Extract features using encoder path
                # This is the CORRECT way to get ViT representations
                x = self.model._process_input(x)
                n = x.shape[0]
                
                # Expand class token for batch
                batch_class_token = self.model.class_token.expand(n, -1, -1)
                x = torch.cat([batch_class_token, x], dim=1)
                
                # Add positional embedding
                x = self.model.encoder.dropout(x + self.model.encoder.pos_embedding)
                
                # Pass through encoder layers
                x = self.model.encoder.layers(x)
                x = self.model.encoder.ln(x)
                
                # Return CLS token (first token)
                return x[:, 0]
        
        class RobustResNetFeatureExtractor(nn.Module):
            """
            ResNet feature extractor.
            
            Uses standard forward pass up to avgpool (before classifier).
            """
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, x):
                # Defensive dtype check
                if x.dtype != torch.float32:
                    x = x.to(dtype=torch.float32)
                
                # Standard ResNet forward up to avgpool
                x = self.model.conv1(x)
                x = self.model.bn1(x)
                x = self.model.relu(x)
                x = self.model.maxpool(x)
                x = self.model.layer1(x)
                x = self.model.layer2(x)
                x = self.model.layer3(x)
                x = self.model.layer4(x)
                x = self.model.avgpool(x)
                return torch.flatten(x, 1)
        
        self.feature_extractors['vit'] = RobustViTFeatureExtractor(
            self.models['vit'], self.device
        ).to(self.device)
        
        self.feature_extractors['resnet'] = RobustResNetFeatureExtractor(
            self.models['resnet']
        ).to(self.device)
        
        print("Feature extractors initialized")
    
    def load_allen_data_with_images(self, 
                                    targeted_structures: List[str] = ['VISp'],
                                    num_neurons: int = 100,
                                    max_images: int = 118):
        """
        Load Allen data with experiment selection.
        
        AllenSDK API with dtype handling.
        """
        if not ALLEN_AVAILABLE:
            raise ImportError("AllenSDK required. Install with: pip install allensdk")
        
        print("\n" + "="*80)
        print("LOADING ALLEN BRAIN OBSERVATORY DATA")
        print("="*80)
        
        # Initialize cache
        cache_dir = './allen_cache'
        os.makedirs(cache_dir, exist_ok=True)
        boc = BrainObservatoryCache(manifest_file=os.path.join(cache_dir, 'manifest.json'))
        
        # Robust experiment selection
        best_exp = choose_best_allen_experiment(
            boc=boc,
            targeted_structures=targeted_structures,
            stimuli=['natural_scenes'],
            auto_pick=True
        )
        
        exp_id = best_exp['id']
        container_id = best_exp.get('experiment_container_id')
        
        print(f"Loading experiment data...")
        data_set = boc.get_ophys_experiment_data(exp_id)
        
        # Get stimulus template
        print("Retrieving natural scene images...")
        stimulus_template = data_set.get_stimulus_template('natural_scenes')
        print(f"Retrieved {len(stimulus_template)} stimulus images")
        
        # Get neural responses
        print("Extracting neural responses...")
        scene_table = data_set.get_stimulus_table('natural_scenes')
        dff_traces = data_set.get_dff_traces()[1]
        
        # Get unique scenes
        unique_scenes = scene_table['frame'].unique()
        unique_scenes = unique_scenes[unique_scenes >= 0]
        unique_scenes = unique_scenes[:max_images]
        
        # Extract matched pairs
        neural_responses = []
        images = []
        image_ids = []
        
        print(f"Matching {len(unique_scenes)} images to neural responses...")
        for scene_idx in tqdm(unique_scenes):
            if scene_idx < len(stimulus_template):
                img = stimulus_template[scene_idx]
                img_pil = Image.fromarray(img.astype(np.uint8)).convert('RGB')
                img_resized = img_pil.resize((224, 224))
                images.append(np.array(img_resized, dtype=np.float32))
            else:
                continue
            
            # Get neural response for this scene
            scene_presentations = scene_table[scene_table['frame'] == scene_idx]
            responses_list = []
            for _, row in scene_presentations.iterrows():
                start_frame = int(row['start'])
                end_frame = int(row['end'])
                mean_response = dff_traces[:, start_frame:end_frame].mean(axis=1)
                responses_list.append(mean_response)
            
            mean_response = np.mean(responses_list, axis=0)
            neural_responses.append(mean_response)
            image_ids.append(int(scene_idx))
        
        # Convert to arrays with explicit float32
        images = np.array(images, dtype=np.float32) / 255.0
        neural_responses = np.array(neural_responses, dtype=np.float32)
        
        # Verify dtypes and ranges
        assert images.dtype == np.float32, f"Images must be float32, got {images.dtype}"
        assert neural_responses.dtype == np.float32, f"Responses must be float32, got {neural_responses.dtype}"
        assert 0 <= images.min() and images.max() <= 1, "Images must be in [0,1]"
        
        print(f"\n{'='*80}")
        print("DTYPE & RANGE VERIFICATION")
        print(f"{'='*80}")
        print(f"Images: dtype={images.dtype}, shape={images.shape}, range=[{images.min():.4f}, {images.max():.4f}]")
        print(f"Neural: dtype={neural_responses.dtype}, shape={neural_responses.shape}")
        print(f"All checks passed")
        print(f"{'='*80}\n")
        
        # Subsample neurons if needed
        n_neurons_available = neural_responses.shape[1]
        if num_neurons < n_neurons_available:
            selected_neurons = np.random.choice(n_neurons_available, num_neurons, replace=False)
            neural_responses = neural_responses[:, selected_neurons]
        else:
            num_neurons = n_neurons_available
        
        # Store data
        self.neural_data = {
            'responses': neural_responses,
            'images': images,
            'image_ids': np.array(image_ids),
            'metadata': {
                'source': 'Allen Brain Observatory',
                'experiment_id': exp_id,
                'container_id': container_id,
                'brain_area': targeted_structures,
                'n_neurons': num_neurons,
                'n_images': len(images),
                'allensdk_version': allensdk.__version__
            }
        }
        
        # Save
        np.save(f"{self.output_dir}/neural_data/responses.npy", neural_responses)
        np.save(f"{self.output_dir}/neural_data/images.npy", images)
        
        # Visualize
        self._visualize_data_loading()
        
        print(f"\nData loading complete: {len(images)} images × {num_neurons} neurons")
        print(f"Experiment {exp_id} from {targeted_structures}")
        print(f"Images and responses properly matched\n")
        
        return self.neural_data
    
    def _visualize_data_loading(self):
        """Create verification visualization."""
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        
        for i in range(min(10, len(self.neural_data['images']))):
            ax = axes[i//5, i%5]
            ax.imshow(self.neural_data['images'][i])
            ax.set_title(f"Scene {self.neural_data['image_ids'][i]}\n" + 
                        f"ΔF/F: {self.neural_data['responses'][i].mean():.3f}",
                        fontsize=8)
            ax.axis('off')
        
        plt.suptitle('Allen Natural Scenes - Real Stimulus Images', 
                    fontweight='bold', fontsize=14)
        plt.tight_layout()
        save_path = f"{self.output_dir}/figures/data_loading_check.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved verification: {save_path}")
        plt.close()
    
    def extract_features(self, model_name: str) -> np.ndarray:
    
        if 'images' not in self.neural_data:
            raise ValueError("Must load Allen data first!")
        
        images = self.neural_data['images']  # (n_images, 224, 224, 3), float32, [0,1]
        extractor = self.feature_extractors[model_name]
        extractor.eval()
        
        # Check cache
        cache_file = f"{self.output_dir}/data/features_{model_name}.npy"
        if os.path.exists(cache_file):
            print(f"Loading cached features from {cache_file}")
            return np.load(cache_file)
        
        # Prepare normalization (move to device once)
        mean = self.imagenet_mean.to(device=self.device).view(1, 3, 1, 1)
        std = self.imagenet_std.to(device=self.device).view(1, 3, 1, 1)
        
        n_images = len(images)
        features_list = []
        
        print(f"\n{'='*80}")
        print(f"EXTRACTING {model_name.upper()} FEATURES")
        print(f"{'='*80}")
        print(f"Processing {n_images} images in batches of {self.batch_size}...")
        
        with torch.no_grad():
            for i in tqdm(range(0, n_images, self.batch_size), desc=f"{model_name}_batches"):
                batch = images[i:i+self.batch_size]  # (B, H, W, C)
                
                # Convert to tensor: (B, C, H, W)
                batch_t = torch.from_numpy(batch).permute(0, 3, 1, 2)
                batch_t = batch_t.to(dtype=torch.float32, device=self.device)
                
                # Apply ImageNet normalization
                batch_normalized = (batch_t - mean) / std
                
                # Extract features
                out = extractor(batch_normalized)
                
                # Handle different output types
                if isinstance(out, torch.Tensor):
                    feats = out
                elif isinstance(out, (tuple, list)):
                    # Take first tensor
                    feats = None
                    for elt in out:
                        if isinstance(elt, torch.Tensor):
                            feats = elt
                            break
                    if feats is None:
                        raise RuntimeError("Model returned tuple/list with no tensor")
                elif isinstance(out, dict):
                    # Try common keys
                    feats = None
                    for k in ['last_hidden_state', 'features', 'logits', 'pooler_output']:
                        if k in out and isinstance(out[k], torch.Tensor):
                            feats = out[k]
                            break
                    if feats is None:
                        for v in out.values():
                            if isinstance(v, torch.Tensor):
                                feats = v
                                break
                    if feats is None:
                        raise RuntimeError("Model returned dict with no tensors")
                else:
                    raise RuntimeError(f"Unsupported output type: {type(out)}")
                
                # Convert to numpy
                feats_np = feats.detach().cpu().numpy()
                
                # Flatten if needed
                if feats_np.ndim > 2:
                    feats_np = feats_np.reshape(feats_np.shape[0], -1)
                
                features_list.extend(list(feats_np))
        
        if not features_list:
            raise RuntimeError("No features extracted")
        
        features = np.vstack(features_list)
        print(f"Extracted features: {features.shape}")
        print(f"Feature dtype: {features.dtype}")
        print(f"{'='*80}\n")
        
        # Cache to disk
        np.save(cache_file, features)
        print(f"Cached features to {cache_file}")
        
        return features
    
    def compute_rdm(self, representations: np.ndarray, metric: str = 'correlation') -> np.ndarray:
        """
        Compute Representational Dissimilarity Matrix (image × image).

        Args:
            representations: (n_images, n_features) array
            metric: 'correlation' or any scipy pdist metric

        Returns:
            rdm: (n_images, n_images) dissimilarity matrix
        """
        assert representations.ndim == 2, "representations must be shape (n_images, n_features)"

        # Defensive: ensure float dtype and finite values
        if not np.issubdtype(representations.dtype, np.floating):
            representations = representations.astype(np.float32)
        representations = np.nan_to_num(representations, nan=0.0, posinf=0.0, neginf=0.0)

        if metric == 'correlation':
            # CORRECT: compute correlation between ROWS (images).
            corr = np.corrcoef(representations)  # rows = images

            # Guard against NaNs from constant rows
            if np.isnan(corr).any():
                corr = np.nan_to_num(corr, nan=0.0)

            # Enforce exact symmetry (numerical safety)
            corr = (corr + corr.T) / 2.0

            # Convert to dissimilarity and ensure diagonal is exactly zero
            rdm = 1.0 - corr
            np.fill_diagonal(rdm, 0.0)
        else:
            rdm = squareform(pdist(representations, metric=metric))
            # Enforce exact symmetry & zero diagonal
            rdm = (rdm + rdm.T) / 2.0
            np.fill_diagonal(rdm, 0.0)

        # Verify square shape
        n = representations.shape[0]
        if rdm.shape != (n, n):
            raise RuntimeError(f"RDM shape mismatch: expected ({n},{n}), got {rdm.shape}")

        return rdm

    
    def compare_rdms(self, rdm1: np.ndarray, rdm2: np.ndarray) -> Tuple[float, float]:
        """Compare RDMs using Spearman correlation (upper triangle)."""
        mask = np.triu(np.ones_like(rdm1, dtype=bool), k=1)
        rdm1_flat = rdm1[mask]
        rdm2_flat = rdm2[mask]
        corr, p_value = spearmanr(rdm1_flat, rdm2_flat)
        return corr, p_value
    
    def fit_encoding_model(self, model_features: np.ndarray, 
                          neural_responses: np.ndarray,
                          alphas: List[float] = [0.01, 0.1, 1.0, 10.0, 100.0],
                          n_folds: int = 5) -> Dict:
        """
        Fit linear encoding models
        Args:
            model_features: (n_images, n_features) array
            neural_responses: (n_images, n_neurons) array
            alphas: Regularization strengths for Ridge
            n_folds: Number of outer CV folds
        
        Returns:
            results: Dict with predictions, scores, median predictivity
        """
        n_images, n_neurons = neural_responses.shape
        
        # Guard for too few images
        if n_images < 2:
            raise ValueError("Need at least 2 images for CV")
        n_folds = min(n_folds, n_images)
        
        predictions = np.zeros((n_images, n_neurons), dtype=np.float32)
        test_scores = np.zeros(n_neurons, dtype=np.float32)
        
        outer_kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.seed)
        
        print(f"\nFitting encoding models for {n_neurons} neurons...")
        print(f"  Outer folds: {n_folds}")
        print(f"  Regularization alphas: {alphas}")
        print(f"  Scaler fit inside each fold (no leakage)")
        
        for neuron_idx in tqdm(range(n_neurons)):
            y = neural_responses[:, neuron_idx]
            fold_predictions = np.zeros(n_images, dtype=np.float32)
            fold_scores = []
            
            # Scaler is fit INSIDE each fold
            for train_idx, test_idx in outer_kf.split(model_features):
                X_train_raw, X_test_raw = model_features[train_idx], model_features[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # FIT SCALER ON TRAINING (prevents leakage)
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train_raw)
                X_test = scaler.transform(X_test_raw)  # Transform only (no fit)
                
                # Inner CV size based on training set
                train_size = X_train.shape[0]
                inner_cv = max(2, min(5, train_size))
                
                # Fit Ridge with inner CV for alpha selection
                ridge = RidgeCV(alphas=alphas, cv=inner_cv)
                ridge.fit(X_train, y_train)
                
                # Predict on held-out test set
                y_pred = ridge.predict(X_test)
                fold_predictions[test_idx] = y_pred
                
                # Score
                if len(np.unique(y_test)) > 1:
                    r, _ = pearsonr(y_test, y_pred)
                    fold_scores.append(r)
                else:
                    fold_scores.append(0.0)
            
            predictions[:, neuron_idx] = fold_predictions
            test_scores[neuron_idx] = np.mean(fold_scores) if fold_scores else 0.0
        
        results = {
            'predictions': predictions,
            'test_scores': test_scores,
            'median_predictivity': float(np.median(test_scores)),
            'mean_predictivity': float(np.mean(test_scores)),
            'scaler': None  # Not stored (fit inside folds)
        }
        
        print(f"Encoding models complete")
        print(f"  Median predictivity: {results['median_predictivity']:.4f}")
        
        return results
    
    def estimate_noise_ceiling(self, neural_responses: np.ndarray, 
                              n_splits: int = 20) -> Dict:
        """Estimate noise ceiling via split-half reliability."""
        n_images, n_neurons = neural_responses.shape
        
        if n_neurons < 4:
            print("Warning: Too few neurons for reliable noise ceiling, returning fallback")
            return {'lower': 0.0, 'upper': 0.0, 'split_correlations': []}
        
        split_correlations = []
        
        for _ in range(n_splits):
            perm = np.random.permutation(n_neurons)
            n_half = n_neurons // 2
            split1 = perm[:n_half]
            split2 = perm[n_half:n_half*2]
            
            if len(split1) == 0 or len(split2) == 0:
                continue
            
            resp1 = neural_responses[:, split1].mean(axis=1)
            resp2 = neural_responses[:, split2].mean(axis=1)
            
            if len(np.unique(resp1)) > 1 and len(np.unique(resp2)) > 1:
                corr, _ = pearsonr(resp1, resp2)
                split_correlations.append(corr)
        
        if not split_correlations:
            return {'lower': 0.0, 'upper': 0.0, 'split_correlations': []}
        
        mean_split_corr = np.mean(split_correlations)
        
        return {
            'lower': mean_split_corr,
            'upper': mean_split_corr,
            'split_correlations': split_correlations
        }
    
    def perform_rdm_analysis(self):
        """Complete RDM analysis with CORRECT image×image correlation."""
        print("\n" + "="*80)
        print("RDM ANALYSIS (Image × Image Correlation)")
        print("="*80)
        
        # Extract features
        vit_features = self.extract_features('vit')
        resnet_features = self.extract_features('resnet')
        neural_responses = self.neural_data['responses']
        
        # Compute RDMs (CORRECT: default rowvar=True for image×image)
        print("\nComputing RDMs...")
        print("  Using np.corrcoef default (rowvar=True) for image×image correlation")
        rdm_vit = self.compute_rdm(vit_features)
        rdm_resnet = self.compute_rdm(resnet_features)
        rdm_neural = self.compute_rdm(neural_responses)
        
        # Verify shapes
        n_images = len(self.neural_data['images'])
        assert rdm_vit.shape == (n_images, n_images)
        assert rdm_resnet.shape == (n_images, n_images)
        assert rdm_neural.shape == (n_images, n_images)
        print(f"RDM shapes verified: ({n_images}, {n_images})")
        
        # Compare RDMs
        print("\nComparing RDMs (Spearman correlation)...")
        vit_neural_corr, vit_p = self.compare_rdms(rdm_vit, rdm_neural)
        resnet_neural_corr, resnet_p = self.compare_rdms(rdm_resnet, rdm_neural)
        vit_resnet_corr, vr_p = self.compare_rdms(rdm_vit, rdm_resnet)
        
        # Store
        self.results['rdm_correlations'] = {
            'vit_neural': {'correlation': vit_neural_corr, 'p_value': vit_p},
            'resnet_neural': {'correlation': resnet_neural_corr, 'p_value': resnet_p},
            'vit_resnet': {'correlation': vit_resnet_corr, 'p_value': vr_p}
        }
        
        # Print
        print(f"\n{'='*80}")
        print("RDM RESULTS")
        print(f"{'='*80}")
        print(f"ViT → Neural:     r = {vit_neural_corr:.4f}, p = {vit_p:.2e}")
        print(f"ResNet → Neural:  r = {resnet_neural_corr:.4f}, p = {resnet_p:.2e}")
        print(f"ViT → ResNet:     r = {vit_resnet_corr:.4f}, p = {vr_p:.2e}")
        print(f"{'='*80}\n")
        
        # Visualize
        self._visualize_rdms(rdm_vit, rdm_resnet, rdm_neural)
        
        # Save
        np.save(f"{self.output_dir}/data/rdm_vit.npy", rdm_vit)
        np.save(f"{self.output_dir}/data/rdm_resnet.npy", rdm_resnet)
        np.save(f"{self.output_dir}/data/rdm_neural.npy", rdm_neural)
        
        return self.results['rdm_correlations']
    
    def _visualize_rdms(self, rdm_vit, rdm_resnet, rdm_neural):
        """Create RDM visualizations."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        vmin = min(rdm_vit.min(), rdm_resnet.min(), rdm_neural.min())
        vmax = max(rdm_vit.max(), rdm_resnet.max(), rdm_neural.max())
        
        im1 = axes[0].imshow(rdm_vit, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0].set_title('ViT-B/16 RDM', fontsize=14, fontweight='bold')
        plt.colorbar(im1, ax=axes[0])
        
        im2 = axes[1].imshow(rdm_resnet, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[1].set_title('ResNet-34 RDM', fontsize=14, fontweight='bold')
        plt.colorbar(im2, ax=axes[1])
        
        im3 = axes[2].imshow(rdm_neural, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[2].set_title('Neural RDM (V1)', fontsize=14, fontweight='bold')
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/figures/rdm_comparison.png", dpi=300)
        print(f"Saved: rdm_comparison.png")
        plt.close()
    
    def perform_neural_predictivity_analysis(self):
        """Neural encoding analysis with NO DATA LEAKAGE."""
        print("\n" + "="*80)
        print("NEURAL PREDICTIVITY ANALYSIS (FIXED: No data leakage)")
        print("="*80)
        
        # Extract features
        vit_features = self.extract_features('vit')
        resnet_features = self.extract_features('resnet')
        neural_responses = self.neural_data['responses']
        
        # Noise ceiling
        print("\nEstimating noise ceiling...")
        noise_ceiling = self.estimate_noise_ceiling(neural_responses)
        self.results['noise_ceilings']['full'] = noise_ceiling
        print(f"Noise ceiling: {noise_ceiling['lower']:.4f}")
        
        # Fit encoding models (FIXED: scaler inside folds)
        print("\nFitting ViT encoding model (scaler inside CV folds)...")
        vit_results = self.fit_encoding_model(vit_features, neural_responses)
        
        print("\nFitting ResNet encoding model (scaler inside CV folds)...")
        resnet_results = self.fit_encoding_model(resnet_features, neural_responses)
        
        # Store
        self.results['neural_predictivity']['vit'] = vit_results
        self.results['neural_predictivity']['resnet'] = resnet_results
        
        # Statistical comparison
        self._compare_predictivity_distributions(vit_results, resnet_results)
        
        # Print
        print(f"\n{'='*80}")
        print("NEURAL PREDICTIVITY RESULTS")
        print(f"{'='*80}")
        print(f"Noise Ceiling:        {noise_ceiling['lower']:.4f}")
        print(f"\nViT-B/16:")
        print(f"  Median r:           {vit_results['median_predictivity']:.4f}")
        print(f"  % of ceiling:       {vit_results['median_predictivity']/noise_ceiling['upper']*100:.1f}%")
        print(f"\nResNet-34:")
        print(f"  Median r:           {resnet_results['median_predictivity']:.4f}")
        print(f"  % of ceiling:       {resnet_results['median_predictivity']/noise_ceiling['upper']*100:.1f}%")
        print(f"{'='*80}\n")
        
        # Visualize
        self._visualize_neural_predictivity(vit_results, resnet_results, noise_ceiling)
        
        return self.results['neural_predictivity']
    
    def _compare_predictivity_distributions(self, vit_results, resnet_results):
        """Statistical comparison."""
        vit_scores = vit_results['test_scores']
        resnet_scores = resnet_results['test_scores']
        
        t_stat, t_p = ttest_rel(vit_scores, resnet_scores)
        w_stat, w_p = wilcoxon(vit_scores, resnet_scores)
        
        diff = vit_scores - resnet_scores
        cohens_d = diff.mean() / diff.std()
        
        self.results['statistical_tests']['predictivity_comparison'] = {
            'paired_t_test': {'t': t_stat, 'p': t_p},
            'wilcoxon': {'w': w_stat, 'p': w_p},
            'cohens_d': cohens_d
        }
        
        print(f"\nStatistical Comparison:")
        print(f"  Paired t-test:  t = {t_stat:.3f}, p = {t_p:.4f}")
        print(f"  Wilcoxon test:  W = {w_stat:.0f}, p = {w_p:.4f}")
        print(f"  Cohen's d:      {cohens_d:.3f}")
    
    def _visualize_neural_predictivity(self, vit_results, resnet_results, noise_ceiling):
        """Create predictivity visualizations."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Violin plot
        ax = axes[0]
        data = [vit_results['test_scores'], resnet_results['test_scores']]
        parts = ax.violinplot(data, positions=[1, 2], showmeans=True, showmedians=True)
        ax.axhline(y=noise_ceiling['upper'], color='gray', linestyle='--', 
                  linewidth=2, label='Noise Ceiling')
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['ViT', 'ResNet'])
        ax.set_ylabel('Neural Predictivity (Pearson r)', fontsize=12)
        ax.set_title('Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Bar plot
        ax = axes[1]
        medians = [vit_results['median_predictivity'], resnet_results['median_predictivity']]
        bars = ax.bar(['ViT', 'ResNet'], medians, color=['#2ecc71', '#e74c3c'], alpha=0.7)
        ax.axhline(y=noise_ceiling['upper'], color='gray', linestyle='--', linewidth=2)
        for bar, med in zip(bars, medians):
            ax.text(bar.get_x() + bar.get_width()/2., med,
                   f'{med:.3f}', ha='center', va='bottom', fontweight='bold')
        ax.set_ylabel('Median Predictivity', fontsize=12)
        ax.set_title('Median Performance', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Scatter
        ax = axes[2]
        ax.scatter(resnet_results['test_scores'], vit_results['test_scores'],
                  alpha=0.5, s=40)
        max_val = max(vit_results['test_scores'].max(), resnet_results['test_scores'].max())
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, linewidth=2)
        ax.set_xlabel('ResNet Predictivity', fontsize=12)
        ax.set_ylabel('ViT Predictivity', fontsize=12)
        ax.set_title('Neuron-by-Neuron', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/figures/neural_predictivity.png", dpi=300)
        print(f"Saved: neural_predictivity.png")
        plt.close()
    
    def save_results(self):
        """Save all results."""
        print(f"\n{'='*80}")
        print("SAVING RESULTS")
        print(f"{'='*80}")
        
        # JSON summary
        results_json = {}
        for key, value in self.results.items():
            if isinstance(value, dict):
                results_json[key] = {}
                for k, v in value.items():
                    if isinstance(v, dict):
                        results_json[key][k] = {
                            kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv
                            for kk, vv in v.items() if not isinstance(vv, (np.ndarray, list))
                        }
        
        with open(f"{self.output_dir}/results_summary.json", 'w') as f:
            json.dump(results_json, f, indent=2)
        print(f"Saved: results_summary.json")
        
        # Pickle
        with open(f"{self.output_dir}/data/complete_results.pkl", 'wb') as f:
            pickle.dump(self.results, f)
        print(f"Saved: complete_results.pkl")
        
        # Text report
        self._create_text_report()
        
        print(f"{'='*80}\n")
    
    def _create_text_report(self):
        """Generate text report."""
        with open(f"{self.output_dir}/ANALYSIS_REPORT.txt", 'w') as f:
            f.write("="*80 + "\n")
            f.write("NEURAL ALIGNMENT ANALYSIS REPORT\n")
            f.write("ViT-B/16 vs ResNet-34: Brain Predictivity\n")
            f.write("="*80 + "\n\n")
                        
            f.write("ANALYSIS METADATA\n")
            f.write("-"*80 + "\n")
            f.write(f"Neural Data: {self.neural_data['metadata']['source']}\n")
            f.write(f"Brain Area: {self.neural_data['metadata']['brain_area']}\n")
            f.write(f"Neurons: {self.neural_data['metadata']['n_neurons']}\n")
            f.write(f"Images: {self.neural_data['metadata']['n_images']}\n")
            f.write(f"Experiment: {self.neural_data['metadata']['experiment_id']}\n\n")
            
            if 'rdm_correlations' in self.results:
                f.write("RDM ANALYSIS (CORRECTED)\n")
                f.write("-"*80 + "\n")
                rdm = self.results['rdm_correlations']
                f.write(f"ViT-Neural:    r = {rdm['vit_neural']['correlation']:.4f} ")
                f.write(f"(p={rdm['vit_neural']['p_value']:.2e})\n")
                f.write(f"ResNet-Neural: r = {rdm['resnet_neural']['correlation']:.4f} ")
                f.write(f"(p={rdm['resnet_neural']['p_value']:.2e})\n\n")
            
            if 'neural_predictivity' in self.results:
                f.write("NEURAL PREDICTIVITY (NO LEAKAGE)\n")
                f.write("-"*80 + "\n")
                vit = self.results['neural_predictivity']['vit']
                resnet = self.results['neural_predictivity']['resnet']
                nc = self.results['noise_ceilings']['full']
                
                f.write(f"Noise Ceiling: {nc['lower']:.4f}\n\n")
                f.write(f"ViT-B/16:  {vit['median_predictivity']:.4f} ")
                f.write(f"({vit['median_predictivity']/nc['upper']*100:.1f}% of ceiling)\n")
                f.write(f"ResNet-34: {resnet['median_predictivity']:.4f} ")
                f.write(f"({resnet['median_predictivity']/nc['upper']*100:.1f}% of ceiling)\n\n")
            
            f.write("="*80 + "\n")
            f.write("ANALYSIS COMPLETE\n")
            f.write("Results and visualizations saved.\n")
            f.write("="*80 + "\n")
        
        print(f"Saved: ANALYSIS_REPORT.txt")


def main():

    # Initialize
    analyzer = ProperNeuralAlignmentAnalyzer(
        output_dir='./analysis_results',
        batch_size=16,
        seed=42
    )
    
    # Load models
    analyzer.load_models()
    
    # Load Allen data
    analyzer.load_allen_data_with_images(
        targeted_structures=['VISp'],
        num_neurons=100,
        max_images=118
    )
    
    # RDM Analysis
    analyzer.perform_rdm_analysis()
    
    # Neural Predictivity
    analyzer.perform_neural_predictivity_analysis()
    
    # Save
    analyzer.save_results()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print(f"Results: {analyzer.output_dir}/")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
