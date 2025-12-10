import sys
from base_neural_analysis import ProperNeuralAlignmentAnalyzer
from ood_testing_comprehensive_corrected import ComprehensiveOODTester


def main():
    print("\n" + "="*80)
    print("COMPREHENSIVE OOD TESTING".center(80))
    print("="*80)
    
    print("\nFEATURES:")
    print("  OOD Testing with transforms:")
    print("     - Weather: snow, frost, fog")
    print("     - Blur: gaussian, defocus, motion, zoom")
    print("     - Noise: gaussian, shot, impulse, speckle")
    print("     - Color: brightness, contrast, saturation, hue")
    print("     - Grayscale")
    print("  Bootstrap 95% CIs")
    print("  Symmetric errorbars")
    print("="*80)
    
    input("\nPress Enter to begin...")
    
    print("\n" + "="*80)
    print("LOADING ANALYZER")
    print("="*80)
    
    analyzer = ProperNeuralAlignmentAnalyzer(
        output_dir='./analysis_results',
        batch_size=16,
        seed=42
    )
    
    print("\nLoading models...")
    analyzer.load_models()
    
    print("\nLoading Allen data...")
    analyzer.load_allen_data_with_images(
        targeted_structures=['VISp'],
        num_neurons=100,
        max_images=118
    )
    
    print("Ready")
    
    # =========================================================================
    # INITIALIZE TESTER
    # =========================================================================
    print("\n" + "="*80)
    print("INITIALIZING OOD TESTER")
    print("="*80)
    
    ood_tester = ComprehensiveOODTester(
        analyzer=analyzer,
        seed=42,
        max_neurons=120,
        n_boot=500
    )
    
    # =========================================================================
    # RUN TEST
    # =========================================================================
    print("\n" + "="*80)
    print("RUNNING OOD TEST")
    print("="*80)
    
    results = ood_tester.run_ood_test(test_ratio=0.2)
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    print(f"\nBaseline ViT:    {results['baseline']['vit']['median']:.4f}")
    print(f"Baseline ResNet: {results['baseline']['resnet']['median']:.4f}")
    
    print(f"\nSplits:")
    print(f"  Training images: {results['metadata']['n_train']} images")
    print(f"  Test images:  {results['metadata']['n_test']} images")
    print(f"  Neurons: {results['metadata']['n_neurons']}")
    print(f"  Bootstrap iterations: {results['metadata']['n_boot']}")
    
    print(f"\nOOD Transforms:")
    print(f"  Total: {len(results['ood_scores'])} transforms")
    
    # Show transform categories
    transforms = list(results['ood_scores'].keys())
    categories = {
        'contrast': len([t for t in transforms if 'contrast' in t]),
        'brightness': len([t for t in transforms if 'brightness' in t]),
        'saturation': len([t for t in transforms if 'saturation' in t]),
        'hue': len([t for t in transforms if 'hue' in t]),
        'blur': len([t for t in transforms if 'blur' in t]),
        'noise': len([t for t in transforms if 'noise' in t]),
        'weather': len([t for t in transforms if 'weather' in t]),
        'grayscale': len([t for t in transforms if 'grayscale' in t])
    }
    
    print("\nTransform breakdown:")
    for cat, count in categories.items():
        if count > 0:
            print(f"  {cat.capitalize()}: {count}")
    
    print(f"\nOutput directory: {ood_tester.output_dir}/")
    print("  ├── figures/comprehensive_ood_analysis.png")
    print("  ├── verification/comprehensive_overview.png")
    print("  ├── transform_checks/")
    print("  ├── data/split_indices.json")
    print("  ├── data/full_results.pkl")
    print("  └── summary_comprehensive.json")


if __name__ == "__main__":
    main()
