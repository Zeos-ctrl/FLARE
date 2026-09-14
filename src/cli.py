# src/cli.py
import argparse
import logging
from src.core.config import ProjectConfig
from src.core.trainer import Trainer
from src.core.tuner import Tuner

def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[logging.StreamHandler()]
    )

def main():
    parser = argparse.ArgumentParser(description='Gravitational Wave Surrogate Training')
    parser.add_argument('action', choices=['train', 'tune', 'both', 'estimate'],
                       help='Action to perform')
    parser.add_argument('--project', default='default_project',
                       help='Project name for checkpointing')

    # Estimation arguments
    parser.add_argument('--event', default='GW150914',
                       help='GWOSC event name for parameter estimation')
    parser.add_argument('--detectors', nargs='+', default=['H1', 'L1'],
                       help='Detectors to use for estimation')
    parser.add_argument('--nwalkers', type=int, default=32,
                       help='Number of MCMC walkers')
    parser.add_argument('--nsteps', type=int, default=2000,
                       help='Number of MCMC steps')
    parser.add_argument('--bank-size', type=int, default=2000,
                       help='Waveform bank size for MCMC warm-start')
    
    # Data arguments
    parser.add_argument('--samples', type=int, default=1000,
                       help='Number of training samples')
    parser.add_argument('--waveform', default='SEOBNRv4',
                       choices=['SEOBNRv4', 'IMRPhenomD'],
                       help='Waveform approximant')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=200,
                       help='Maximum training epochs')
    parser.add_argument('--batch-size', type=int, default=32768,
                       help='Batch size for training')
    parser.add_argument('--amp-lr', type=float, default=0.0005,
                       help='Amplitude model learning rate')
    parser.add_argument('--phase-lr', type=float, default=0.0005,
                       help='Phase model learning rate')
    
    # HPO arguments
    parser.add_argument('--hpo-trials', type=int, default=50,
                       help='Number of HPO trials')
    parser.add_argument('--hpo-samples', type=int, default=1000,
                       help='Samples to use during HPO')
    
    # General arguments
    parser.add_argument('--device', default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose logging')
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    # Create config from arguments
    config = ProjectConfig(
        project_name=args.project,
        num_samples=args.samples,
        waveform=args.waveform,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        amp_lr=args.amp_lr,
        phase_lr=args.phase_lr,
        hpo_trials=args.hpo_trials,
        hpo_samples=args.hpo_samples,
        device=args.device
    )
    
    if args.action == 'train':
        trainer = Trainer(config)
        trainer.run_training()
    elif args.action == 'tune':
        tuner = Tuner(config)
        tuner.run_tuning()
    elif args.action == 'both':
        tuner = Tuner(config)
        tuner.run_tuning()
        trainer = Trainer(config)
        trainer.run_training()
    elif args.action == 'estimate':
        run_estimate(config, args)


def run_estimate(config, args):
    """Estimate parameters for a GWOSC event using a trained surrogate."""
    import os
    import json
    from src.inference.predictor import WaveformPredictor
    from src.inference.estimator import GWEventEstimator, WaveformBank

    predictor = WaveformPredictor(config.project_path, device=config.device)
    estimator = GWEventEstimator(predictor, args.event, detectors=args.detectors)

    bank = WaveformBank(predictor, estimator.prior)
    bank.build(n_samples=args.bank_size)

    result = estimator.run_mcmc(
        nwalkers=args.nwalkers, nsteps=args.nsteps, bank=bank,
    )

    out_dir = os.path.join(config.project_path, 'estimation', args.event)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'posterior_summary.json'), 'w') as f:
        json.dump(result.to_dict(), f, indent=2)
    result.corner_plot(path=os.path.join(out_dir, 'corner.png'))
    try:
        estimator.reconstruction_plot(result, path=os.path.join(out_dir, 'reconstruction.png'))
    except Exception as exc:  # noqa: BLE001 - never fail the run on a plot
        logging.getLogger('flare').warning("Reconstruction plot skipped: %s", exc)

    logging.getLogger('flare').info("Estimation complete. Results in %s", out_dir)
    for name, stats in result.summary().items():
        logging.getLogger('flare').info(
            "  %s: %.3f (+%.3f / -%.3f)", name, stats['median'],
            stats['upper_90'] - stats['median'], stats['median'] - stats['lower_90'])


if __name__ == '__main__':
    main()
