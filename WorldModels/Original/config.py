from dataclasses import dataclass

@dataclass
class Config:
    # 共通
    learning_rate: float = 0.001
    batch_size: int = 32
    
    # VAE
    z_size: int = 32
    kl_tolerance: float = 0.5

    # MDN-RNN
    action_num: int = 3
    sequence_length: int = 10
    rnn_units: int = 256
    num_mixture: int = 5  # number of mixtures in MDN
    temperature: float = 1.15

    # Controller（CMA-ES）
    num_individual: int = 16
    num_generations: int = 2    # initial value 100
    cma_sigma: float = 0.5

