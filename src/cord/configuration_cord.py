"""Configuration for the CoRD Transformers model."""

from transformers import PretrainedConfig


class CordConfig(PretrainedConfig):
    """Configuration for the prefix-causal CoRD language model.

    The recurrent macroblock is shared across loop iterations. The default
    values describe the research-scale blueprint; the checked-in
    ``configs/cord-50m.json`` is the development configuration.
    """

    model_type = "cord"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=48_000,
        hidden_size=2_048,
        intermediate_size=8_192,
        num_hidden_layers=32,
        num_decoder_layers=None,
        num_attention_heads=16,
        num_key_value_heads=None,
        concept_slots=128,
        concept_num_attention_heads=16,
        concept_latent_size=512,
        num_recurrent_loops=2,
        max_recurrent_loops=32,
        minimum_recurrent_loops=None,
        num_kda_layers=3,
        num_mla_layers=1,
        kda_backend="auto",
        kda_chunk_size=64,
        kda_conv_kernel_size=4,
        latent_kv_size=768,
        qk_rope_head_dim=None,
        qk_nope_head_dim=None,
        routed_latent_width=1_024,
        expert_intermediate_size=2_048,
        num_experts=32,
        num_experts_per_token=2,
        num_shared_experts=1,
        use_moe=True,
        use_situ_glu=True,
        use_block_attn_res=True,
        block_attn_res_slots=4,
        hot_frontier_size=8,
        hot_frontier_max_size=16,
        gpu_checkpoint_slots=4,
        archive_checkpoint_interval=4,
        local_bptt_loops=4,
        num_reasoning_operators=5,
        controller_hidden_size=None,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        initializer_range=0.02,
        rope_theta=10_000.0,
        max_position_embeddings=32_768,
        use_cache=True,
        tie_word_embeddings=True,
        loop_dropout=0.0,
        attention_dropout=0.0,
        halting_threshold=0.5,
        budget_penalty=0.0,
        router_bias_update_rate=0.001,
        router_activation="softmax",
        router_renormalize=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        **kwargs,
    ):
        if num_decoder_layers is None:
            num_decoder_layers = num_hidden_layers
        if minimum_recurrent_loops is None:
            minimum_recurrent_loops = num_recurrent_loops
        if vocab_size < 1 or hidden_size < 1 or intermediate_size < 1:
            raise ValueError("vocab_size, hidden_size and intermediate_size must be positive")
        if num_hidden_layers < 1 or num_decoder_layers < 1:
            raise ValueError("num_hidden_layers and num_decoder_layers must be positive")
        if num_attention_heads < 1:
            raise ValueError("num_attention_heads must be positive")
        if num_recurrent_loops < 1 or max_recurrent_loops < num_recurrent_loops:
            raise ValueError("max_recurrent_loops must be >= num_recurrent_loops >= 1")
        if minimum_recurrent_loops < 1 or minimum_recurrent_loops > max_recurrent_loops:
            raise ValueError("minimum_recurrent_loops must be within the recurrent loop range")
        if hidden_size % num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        if not 1 <= num_key_value_heads <= num_attention_heads:
            raise ValueError("num_key_value_heads must be in [1, num_attention_heads]")
        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if concept_slots < 1 or concept_latent_size < 1:
            raise ValueError("concept_slots and concept_latent_size must be positive")
        if concept_num_attention_heads < 1:
            raise ValueError("concept_num_attention_heads must be positive")
        if hidden_size % concept_num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by concept_num_attention_heads")
        if concept_latent_size % concept_num_attention_heads != 0:
            raise ValueError("concept_latent_size must be divisible by concept_num_attention_heads")
        if num_kda_layers < 0 or num_mla_layers < 0 or num_kda_layers + num_mla_layers < 1:
            raise ValueError("the macroblock must contain at least one attention layer")
        if num_experts < 0 or num_shared_experts not in {0, 1}:
            raise ValueError("num_experts must be non-negative and num_shared_experts must be 0 or 1")
        if use_moe and (num_experts < 1 or not 1 <= num_experts_per_token <= num_experts):
            raise ValueError("num_experts_per_token must be between 1 and num_experts")
        if hot_frontier_size < 1 or hot_frontier_max_size < hot_frontier_size:
            raise ValueError("hot_frontier_max_size must be >= hot_frontier_size >= 1")
        if block_attn_res_slots < 1 or gpu_checkpoint_slots < 1:
            raise ValueError("residual and checkpoint capacities must be positive")
        if archive_checkpoint_interval < 1:
            raise ValueError("archive_checkpoint_interval must be positive")
        if local_bptt_loops < 1:
            raise ValueError("local_bptt_loops must be positive")
        if any(
            value < 1
            for value in (kda_chunk_size, kda_conv_kernel_size, latent_kv_size, routed_latent_width,
                          expert_intermediate_size, max_position_embeddings)
        ):
            raise ValueError("kernel, latent, expert and position sizes must be positive")
        if num_reasoning_operators != 5:
            raise ValueError("CoRD v1 defines exactly five structural operators")
        if controller_hidden_size is not None and controller_hidden_size < 1:
            raise ValueError("controller_hidden_size must be positive")
        if not 0.0 <= halting_threshold <= 1.0:
            raise ValueError("halting_threshold must be between 0 and 1")
        if not 0.0 <= loop_dropout <= 1.0 or not 0.0 <= attention_dropout <= 1.0:
            raise ValueError("loop_dropout and attention_dropout must be between 0 and 1")
        if router_bias_update_rate < 0.0:
            raise ValueError("router_bias_update_rate must be non-negative")
        if kda_backend not in {"auto", "eager", "fla"}:
            raise ValueError("kda_backend must be one of: auto, eager, fla")
        if router_activation not in {"softmax", "sigmoid"}:
            raise ValueError("router_activation must be softmax or sigmoid")

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.concept_slots = concept_slots
        self.concept_num_attention_heads = concept_num_attention_heads
        self.concept_latent_size = concept_latent_size
        self.num_recurrent_loops = num_recurrent_loops
        self.max_recurrent_loops = max_recurrent_loops
        self.minimum_recurrent_loops = minimum_recurrent_loops
        self.num_kda_layers = num_kda_layers
        self.num_mla_layers = num_mla_layers
        self.kda_backend = kda_backend
        self.kda_chunk_size = kda_chunk_size
        self.kda_conv_kernel_size = kda_conv_kernel_size
        self.latent_kv_size = latent_kv_size
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.routed_latent_width = routed_latent_width
        self.expert_intermediate_size = expert_intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.num_shared_experts = num_shared_experts
        self.use_moe = use_moe
        self.use_situ_glu = use_situ_glu
        self.use_block_attn_res = use_block_attn_res
        self.block_attn_res_slots = block_attn_res_slots
        self.hot_frontier_size = hot_frontier_size
        self.hot_frontier_max_size = hot_frontier_max_size
        self.gpu_checkpoint_slots = gpu_checkpoint_slots
        self.archive_checkpoint_interval = archive_checkpoint_interval
        self.local_bptt_loops = local_bptt_loops
        self.num_reasoning_operators = num_reasoning_operators
        self.controller_hidden_size = controller_hidden_size
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.use_cache = use_cache
        self.loop_dropout = loop_dropout
        self.attention_dropout = attention_dropout
        self.halting_threshold = halting_threshold
        self.budget_penalty = budget_penalty
        self.router_bias_update_rate = router_bias_update_rate
        self.router_activation = router_activation
        self.router_renormalize = router_renormalize
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
