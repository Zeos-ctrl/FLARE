# src/core/tuner.py
import optuna
import json
import os
import logging
from typing import Optional

from src.core.config import ProjectConfig
from src.core.trainer import Trainer
from src.data.generator import WaveformGenerator

logger = logging.getLogger(__name__)

class Tuner:
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.checkpoint_path = config.project_path
        os.makedirs(self.checkpoint_path, exist_ok=True)
        
        # Generate HPO dataset once
        self.generator = WaveformGenerator(
            waveform_length=config.waveform_length,
            delta_t=config.delta_t,
            f_lower=config.f_lower,
            approximant=config.waveform
        )

        logger.info(f"Generating {config.hpo_samples} samples for HPO...")
        self.hpo_dataset = self.generator.generate_dataset(
            n_samples=config.hpo_samples,
            clean=config.clean_data,
            feature_names=config.feature_names,
            sampling_ranges=config.sampling_ranges
        )
    
    def run_tuning(self, model_type: str = "both", n_trials: Optional[int] = None,
                   callback=None):
        """Run hyperparameter optimization.

        Args:
            callback: Optional Optuna callback ``fn(study, trial)`` invoked after
                each trial; used by the dashboard to stream trial progress.
        """
        n_trials = n_trials or self.config.hpo_trials

        # Create study storage
        storage = f'sqlite:///{self.checkpoint_path}/optuna.db'

        if model_type in ["amp", "both"]:
            self._tune_amplitude(n_trials, storage, callback=callback)

        if model_type in ["phase", "both"]:
            self._tune_phase(n_trials, storage, callback=callback)

    def _tune_amplitude(self, n_trials: int, storage: str, callback=None):
        """Tune amplitude model"""
        study_name = f'{self.config.project_name}_amp'

        study = optuna.create_study(
            study_name=study_name,
            direction='minimize',
            storage=storage,
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
            load_if_exists=True
        )

        study.optimize(
            lambda trial: self._amp_objective(trial),
            n_trials=n_trials,
            callbacks=[callback] if callback else None,
        )
        
        # Save best params
        best_params = study.best_params
        param_file = os.path.join(self.checkpoint_path, 'amp_params.json')
        with open(param_file, 'w') as f:
            json.dump(best_params, f, indent=2)
        
        logger.info(f"Best amp params: {best_params}")
        logger.info(f"Best amp value: {study.best_value:.6f}")
    
    def _amp_objective(self, trial: optuna.Trial) -> float:
        """Objective function for amplitude model"""
        # Sample hyperparameters
        params = {
            'amp_hidden_size': trial.suggest_categorical('amp_hidden_size', [64, 128, 256, 512]),
            'layers': trial.suggest_int('layers', 2, 6),
            'banks': trial.suggest_int('banks', 1, 6),
            'dropout': trial.suggest_float('dropout', 0.0, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-8, 1e-2, log=True),
            'grad_clip': trial.suggest_float('grad_clip', 0.5, 5.0)
        }
        # Carry custom-model source so the factory builds the right network.
        params['model_kind'] = self.config.model_kind
        params['amp_module'] = self.config.amp_module
        params['phase_module'] = self.config.phase_module

        # Create config for this trial
        trial_config = ProjectConfig(**self.config.__dict__)
        trial_config.amp_lr = params['learning_rate']
        trial_config.amp_weight_decay = params['weight_decay']
        trial_config.amp_clip = params['grad_clip']
        trial_config.amp_dropout = params['dropout']
        trial_config.amp_hidden_layers = [params['amp_hidden_size']] * params['layers']
        trial_config.amp_banks = params['banks']
        trial_config.num_epochs = 50  # Shorter for HPO
        
        # Train with trial config
        trainer = Trainer(trial_config)
        loaders = trainer.create_dataloaders(self.hpo_dataset)
        
        # Create and train model
        from src.models.model_factory import make_amp_model
        model = make_amp_model(len(self.config.feature_names), params).to(trainer.device)
        
        # Train with early stopping and pruning
        best_val_loss = float('inf')
        
        for epoch in range(trial_config.num_epochs):
            # Training step
            val_loss = trainer._train_epoch(model, loaders['amp'], 'amp')
            
            # Report to Optuna
            trial.report(val_loss, epoch)
            
            # Pruning
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
            best_val_loss = min(best_val_loss, val_loss)

        return best_val_loss

    def _tune_phase(self, n_trials: int, storage: str, callback=None):
        """Tune phase model"""
        study_name = f'{self.config.project_name}_phase'

        study = optuna.create_study(
            study_name=study_name,
            direction='minimize',
            storage=storage,
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
            load_if_exists=True
        )

        study.optimize(
            lambda trial: self._phase_objective(trial),
            n_trials=n_trials,
            callbacks=[callback] if callback else None,
        )

        best_params = study.best_params
        param_file = os.path.join(self.checkpoint_path, 'phase_params.json')
        with open(param_file, 'w') as f:
            json.dump(best_params, f, indent=2)

        logger.info(f"Best phase params: {best_params}")
        logger.info(f"Best phase value: {study.best_value:.6f}")

    def _phase_objective(self, trial: optuna.Trial) -> float:
        """Objective function for phase model"""
        params = {
            'phase_hidden_size': trial.suggest_categorical('phase_hidden_size', [64, 128, 256, 512]),
            'layers': trial.suggest_int('layers', 2, 6),
            'banks': trial.suggest_int('banks', 1, 8),
            'dropout': trial.suggest_float('dropout', 0.0, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-8, 1e-2, log=True),
            'grad_clip': trial.suggest_float('grad_clip', 0.5, 5.0)
        }
        # Carry custom-model source so the factory builds the right network.
        params['model_kind'] = self.config.model_kind
        params['amp_module'] = self.config.amp_module
        params['phase_module'] = self.config.phase_module

        # Create config for this trial
        trial_config = ProjectConfig(**self.config.__dict__)
        trial_config.phase_lr = params['learning_rate']
        trial_config.phase_weight_decay = params['weight_decay']
        trial_config.phase_clip = params['grad_clip']
        trial_config.phase_dropout = params['dropout']
        trial_config.phase_hidden_layers = [params['phase_hidden_size']] * params['layers']
        trial_config.phase_banks = params['banks']
        trial_config.num_epochs = 50  # Shorter for HPO

        trainer = Trainer(trial_config)
        loaders = trainer.create_dataloaders(self.hpo_dataset)

        from src.models.model_factory import make_phase_model
        model = make_phase_model(len(self.config.feature_names), params).to(trainer.device)

        best_val_loss = float('inf')

        for epoch in range(trial_config.num_epochs):
            val_loss = trainer._train_epoch(model, loaders['phase'], 'phase')
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            best_val_loss = min(best_val_loss, val_loss)

        return best_val_loss
