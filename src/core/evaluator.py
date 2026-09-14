# src/core/evaluator.py
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json

from src.core.config import ProjectConfig
from src.inference.predictor import WaveformPredictor
from src.data.generator import WaveformGenerator
from src.utils.metrics import compute_match
from scipy.stats import gaussian_kde

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResults:
    """Container for benchmark results"""
    matches: np.ndarray
    mean_match: float
    std_match: float
    min_match: float
    max_match: float
    percentiles: Dict[int, float]
    parameters: np.ndarray
    best_idx: int
    worst_idx: int
    
    def to_dict(self) -> dict:
        return {
            'mean_match': float(self.mean_match),
            'std_match': float(self.std_match),
            'min_match': float(self.min_match),
            'max_match': float(self.max_match),
            'percentiles': {k: float(v) for k, v in self.percentiles.items()},
            'n_samples': len(self.matches)
        }

class Evaluator:
    """Evaluate trained models against ground truth waveforms"""
    
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.predictor = WaveformPredictor(checkpoint_path, device)
        
        # Load metadata to get training configuration
        with open(os.path.join(checkpoint_path, 'meta.json')) as f:
            self.meta = json.load(f)
        
        self.waveform = self.meta['waveform']
        self.waveform_length = self.meta['waveform_length']
        self.delta_t = self.meta['delta_t']

        # Load the training config so the benchmark draws test parameters from
        # the SAME distribution the model was trained on. Without this the
        # benchmark samples the generator's default ranges (e.g. spins in
        # [-0.7, 0.7]); a model trained on a restricted range -- e.g. zero spin,
        # so (chirp_mass, eta) fully determines the waveform -- would then be
        # scored against out-of-distribution ground truth and look far worse
        # than it is.
        try:
            self.config = ProjectConfig.load(checkpoint_path)
            self.sampling_ranges = self.config.sampling_ranges
            f_lower = self.config.f_lower
        except (FileNotFoundError, KeyError, TypeError):
            self.config = None
            self.sampling_ranges = None            # fall back to generator defaults
            f_lower = self.meta.get('f_lower', 20.0)

        # Initialize generator with same settings as training. The resize mode
        # must match training so the benchmark ground truth is on the same grid.
        fixed_window = bool(self.meta.get('td_fixed_window', False))
        self.generator = WaveformGenerator(
            waveform_length=self.waveform_length,
            delta_t=self.delta_t,
            f_lower=f_lower,
            approximant=self.waveform,
            fixed_window=fixed_window,
        )
        
        # Output directory for plots
        self.output_dir = os.path.join(checkpoint_path, 'evaluation')
        os.makedirs(self.output_dir, exist_ok=True)
    
    def benchmark(self, n_samples: int = 1000, 
                 batch_size: int = 32,
                 plot: bool = True) -> BenchmarkResults:
        """
        Run comprehensive benchmark against ground truth
        
        Args:
            n_samples: Number of test samples to generate
            batch_size: Batch size for prediction
            plot: Whether to generate plots
        
        Returns:
            BenchmarkResults object with statistics
        """
        logger.info(f"Running benchmark with {n_samples} samples...")
        
        # Generate test parameters from the training distribution (same ranges),
        # so we evaluate in-distribution rather than against the generator's
        # wider default parameter ranges.
        parameters = self.generator.sample_parameters(
            n_samples, **(self.sampling_ranges or {}))
        
        # Generate ground truth waveforms
        logger.info("Generating ground truth waveforms...")
        true_waveforms = []
        for params in parameters:
            h_true = self.generator._generate_single_waveform(params, clean=True)
            true_waveforms.append(h_true)
        
        # Predict waveforms
        logger.info("Generating predictions...")
        h_plus_list, h_cross_list = self.predictor.batch_predict(
            parameters, batch_size=batch_size
        )
        
        # Compute matches
        logger.info("Computing matches...")
        matches = np.zeros(n_samples)
        for i in range(n_samples):
            h_pred = h_plus_list[i].data
            h_true = true_waveforms[i]
            
            # Ensure same length
            min_len = min(len(h_pred), len(h_true))
            h_pred = h_pred[:min_len]
            h_true = h_true[:min_len]
            
            matches[i] = compute_match(h_true, h_pred, self.delta_t)

        # A degenerate (e.g. all-zero) prediction makes compute_match return
        # NaN/inf; that is a failed reconstruction, so score it 0 rather than
        # letting NaN poison the statistics and the plotting (polyfit/kde).
        matches = np.nan_to_num(matches, nan=0.0, posinf=0.0, neginf=0.0)

        # Compute statistics
        results = BenchmarkResults(
            matches=matches,
            mean_match=np.mean(matches),
            std_match=np.std(matches),
            min_match=np.min(matches),
            max_match=np.max(matches),
            percentiles={
                25: np.percentile(matches, 25),
                50: np.percentile(matches, 50),
                75: np.percentile(matches, 75),
                90: np.percentile(matches, 90),
                95: np.percentile(matches, 95),
                99: np.percentile(matches, 99)
            },
            parameters=parameters,
            best_idx=np.argmax(matches),
            worst_idx=np.argmin(matches)
        )
        
        # Log results
        logger.info(f"Mean match: {results.mean_match:.4f} ± {results.std_match:.4f}")
        logger.info(f"Min/Max: {results.min_match:.4f} / {results.max_match:.4f}")
        logger.info(f"Median: {results.percentiles[50]:.4f}")
        logger.info(f"95th percentile: {results.percentiles[95]:.4f}")
        
        # Save results
        results_file = os.path.join(self.output_dir, 'benchmark_results.json')
        with open(results_file, 'w') as f:
            json.dump(results.to_dict(), f, indent=2)
        
        # Generate plots if requested
        if plot:
            self.generate_plots(results, true_waveforms, h_plus_list)
        
        return results
    
    def generate_plots(self, results: BenchmarkResults, 
                      true_waveforms: List, 
                      predictions: List):
        """Generate comprehensive evaluation plots"""
        
        plt.rcParams.update({
            'font.size': 14,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.titlesize': 18,
            'font.family': 'serif',
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.linewidth': 1.2,
            'lines.linewidth': 2,
        })
        
        self._plot_match_distribution(results)
        
        self._plot_match_vs_parameters(results)
        
        self._plot_best_worst_cases(results, true_waveforms, predictions)
        
        self._plot_parameter_heatmap(results)
        
        self._plot_summary_figure(results)
        
        logger.info(f"Plots saved to {self.output_dir}")
    
    def _plot_match_distribution(self, results: BenchmarkResults):
        """Plot histogram and KDE of match distribution"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram with KDE
        ax1.hist(results.matches, bins=50, density=True, alpha=0.7, 
                color='skyblue', edgecolor='black', label='Histogram')
        
        if len(results.matches) > 1:
            kde = gaussian_kde(results.matches)
            x_range = np.linspace(results.matches.min(), results.matches.max(), 200)
            ax1.plot(x_range, kde(x_range), 'r-', lw=2, label='KDE')
        
        # Add statistical lines
        ax1.axvline(results.mean_match, color='green', linestyle='--', 
                   label=f'Mean: {results.mean_match:.4f}')
        ax1.axvline(results.percentiles[50], color='orange', linestyle='--',
                   label=f'Median: {results.percentiles[50]:.4f}')
        
        # Add sigma regions
        for sigma in [1, 2, 3]:
            lower = results.mean_match - sigma * results.std_match
            upper = results.mean_match + sigma * results.std_match
            ax1.axvspan(lower, upper, alpha=0.1, color='gray')
        
        ax1.set_xlabel('Match')
        ax1.set_ylabel('Probability Density')
        ax1.set_title('Match Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot and violin plot
        parts = ax2.violinplot([results.matches], positions=[1], 
                               showmeans=True, showmedians=True)
        ax2.boxplot([results.matches], positions=[1], widths=0.15)
        
        ax2.set_ylabel('Match')
        ax2.set_title('Match Statistics')
        ax2.set_xticks([1])
        ax2.set_xticklabels([self.waveform])
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'match_distribution.png'), dpi=150)
        plt.close()
    
    def _plot_match_vs_parameters(self, results: BenchmarkResults):
        """Plot match vs individual parameters"""
        param_names = ['m1', 'm2', 'spin1z', 'spin2z', 'inclination', 'eccentricity']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (ax, name) in enumerate(zip(axes, param_names)):
            param_values = results.parameters[:, i]
            
            # Scatter plot
            scatter = ax.scatter(param_values, results.matches,
                               c=results.matches, cmap='viridis',
                               s=10, alpha=0.6)

            # Add trend line -- skip when the parameter is constant (e.g. spin
            # pinned to 0), which has no trend and makes polyfit's normalisation
            # divide by zero.
            if np.ptp(param_values) > 0:
                z = np.polyfit(param_values, results.matches, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(param_values.min(), param_values.max(), 100)
                ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, lw=2)
            
            ax.set_xlabel(name)
            ax.set_ylabel('Match')
            ax.set_title(f'Match vs {name}')
            ax.grid(True, alpha=0.3)
            
            # Add colorbar for first plot
            if i == 0:
                plt.colorbar(scatter, ax=ax, label='Match')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'match_vs_parameters.png'), dpi=150)
        plt.close()
    
    def _plot_best_worst_cases(self, results: BenchmarkResults,
                               true_waveforms: List,
                               predictions: List):
        """Plot comparison of best and worst matches"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        for idx, (case_idx, title) in enumerate([(results.best_idx, 'Best Match'),
                                                  (results.worst_idx, 'Worst Match')]):
            row = idx
            
            # Get waveforms
            h_true = true_waveforms[case_idx]
            h_pred = predictions[case_idx].data

            # x-axis in geometric time t/M (dimensionless), the NR/surrogate
            # convention: mass-normalised and merger-centred, so waveforms are
            # comparable across masses. The model grid pins the merger at
            # MERGER_FRAC*L; M is the case's total mass. (T_sun = G M_sun / c^3.)
            T_SUN = 4.925491025543576e-6
            M = float(results.parameters[case_idx, 0] + results.parameters[case_idx, 1])
            merger_idx = self.generator.MERGER_FRAC * self.waveform_length
            time = (np.arange(len(h_true)) - merger_idx) * self.delta_t / (M * T_SUN)

            # Ensure same length
            min_len = min(len(h_true), len(h_pred))
            h_true = h_true[:min_len]
            h_pred = h_pred[:min_len]
            time = time[:min_len]

            # Waveform comparison
            axes[row, 0].plot(time, h_true, 'b-', label='True', alpha=0.7)
            axes[row, 0].plot(time, h_pred, 'r--', label='Predicted', alpha=0.7)
            axes[row, 0].set_xlabel('t / M  (geometric time, merger at 0)')
            axes[row, 0].set_ylabel('Strain')
            axes[row, 0].set_title(f'{title} - Match: {results.matches[case_idx]:.4f}')
            axes[row, 0].legend()
            axes[row, 0].grid(True, alpha=0.3)

            # Residual
            residual = h_pred - h_true
            axes[row, 1].plot(time, residual, 'g-', alpha=0.7)
            axes[row, 1].set_xlabel('t / M  (geometric time, merger at 0)')
            axes[row, 1].set_ylabel('Residual')
            axes[row, 1].set_title(f'{title} - Residual')
            axes[row, 1].grid(True, alpha=0.3)
            
            # Add parameter info
            params = results.parameters[case_idx]
            param_text = (f'm1={params[0]:.1f}, m2={params[1]:.1f}\n'
                         f's1z={params[2]:.2f}, s2z={params[3]:.2f}\n'
                         f'inc={params[4]:.2f}, ecc={params[5]:.3f}')
            axes[row, 1].text(0.02, 0.98, param_text,
                            transform=axes[row, 1].transAxes,
                            fontsize=10, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'best_worst_comparison.png'), dpi=150)
        plt.close()
    
    def _plot_parameter_heatmap(self, results: BenchmarkResults):
        """Create 2D heatmap of match in parameter space"""
        from scipy.interpolate import griddata
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Define parameter pairs to plot
        pairs = [
            ('m1', 'm2', 0, 1),
            ('m1', 'chirp_mass', 0, -1),
            ('mass_ratio', 'total_mass', -2, -3)
        ]
        
        # Component masses are always needed to derive chirp mass etc.
        m1, m2 = results.parameters[:, 0], results.parameters[:, 1]

        for ax, (xlabel, ylabel, idx1, idx2) in zip(axes, pairs):
            if idx1 >= 0:
                x = results.parameters[:, idx1]
            else:
                # Compute derived parameters
                if xlabel == 'mass_ratio':
                    x = m2 / m1
                elif xlabel == 'total_mass':
                    x = m1 + m2

            if idx2 >= 0:
                y = results.parameters[:, idx2]
            else:
                # Compute derived parameters
                if ylabel == 'chirp_mass':
                    y = (m1 * m2)**(3/5) / (m1 + m2)**(1/5)
                elif ylabel == 'total_mass':
                    y = m1 + m2
            
            # Create grid
            xi = np.linspace(x.min(), x.max(), 100)
            yi = np.linspace(y.min(), y.max(), 100)
            Xi, Yi = np.meshgrid(xi, yi)
            
            # Interpolate matches onto grid
            Zi = griddata((x, y), results.matches, (Xi, Yi), method='cubic')
            
            # Plot heatmap
            im = ax.contourf(Xi, Yi, Zi, levels=20, cmap='viridis')
            ax.scatter(x, y, c=results.matches, s=5, alpha=0.5, cmap='viridis')
            
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f'Match vs {xlabel}-{ylabel}')
            plt.colorbar(im, ax=ax, label='Match')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'parameter_heatmap.png'), dpi=150)
        plt.close()
    
    def _plot_summary_figure(self, results: BenchmarkResults):
        """Create summary figure with key statistics"""
        fig = plt.figure(figsize=(16, 10))
        
        # Create grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Match distribution
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.hist(results.matches, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(results.mean_match, color='red', linestyle='--', label=f'Mean: {results.mean_match:.4f}')
        ax1.set_xlabel('Match')
        ax1.set_ylabel('Count')
        ax1.set_title('Match Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Statistics text box
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis('off')
        stats_text = (
            f"Statistics ({results.matches.shape[0]} samples)\n"
            f"{'='*30}\n"
            f"Mean:    {results.mean_match:.4f} ± {results.std_match:.4f}\n"
            f"Median:  {results.percentiles[50]:.4f}\n"
            f"Min/Max: {results.min_match:.4f} / {results.max_match:.4f}\n"
            f"\nPercentiles:\n"
            f"  25%: {results.percentiles[25]:.4f}\n"
            f"  75%: {results.percentiles[75]:.4f}\n"
            f"  90%: {results.percentiles[90]:.4f}\n"
            f"  95%: {results.percentiles[95]:.4f}\n"
            f"  99%: {results.percentiles[99]:.4f}\n"
            f"\nTraining Config:\n"
            f"  Waveform: {self.waveform}\n"
            f"  Length: {self.waveform_length}\n"
            f"  Delta_t: {self.delta_t}"
        )
        ax2.text(0.1, 0.9, stats_text, transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Scatter plots for key parameters
        param_indices = [0, 1, 4, 5]  # m1, m2, inclination, eccentricity
        param_names = ['m1', 'm2', 'inclination', 'eccentricity']
        
        for i, (idx, name) in enumerate(zip(param_indices, param_names)):
            ax = fig.add_subplot(gs[1 + i//2, i%2])
            ax.scatter(results.parameters[:, idx], results.matches, 
                      c=results.matches, cmap='viridis', s=5, alpha=0.6)
            ax.set_xlabel(name)
            ax.set_ylabel('Match')
            ax.set_title(f'Match vs {name}')
            ax.grid(True, alpha=0.3)
        
        # QQ plot
        ax_qq = fig.add_subplot(gs[2, 2])
        from scipy import stats
        stats.probplot(results.matches, dist="norm", plot=ax_qq)
        ax_qq.set_title('Q-Q Plot')
        ax_qq.grid(True, alpha=0.3)
        
        plt.suptitle(f'Evaluation Summary - {self.meta["project_name"]}', fontsize=16)
        plt.savefig(os.path.join(self.output_dir, 'summary.png'), dpi=150, bbox_inches='tight')
        plt.close()
