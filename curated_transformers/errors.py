class ErrorsWithCodes(type):
    def __getattribute__(self, code):
        msg = super().__getattribute__(code)
        if code.startswith("__"):  # python system attributes like __class__
            return msg
        else:
            return "(curated-transformers) [{code}] {msg}".format(code=code, msg=msg)


# fmt: off

class Warnings(metaclass=ErrorsWithCodes):
    W001 = ("Skipping transfomer pipe with TorchScript model - Model "
            "is likely already quantized")


class Errors(metaclass=ErrorsWithCodes):
    E001 = ("Attempting to quantize a transformer pipe with a "
            "non-PyTorch model ('{model_name}'). Quantization "
            "is only supported by PyTorch-specific curated "
            "transformer models")
    E002 = ("The number of hidden layers ({num_hidden_layers}) in the "
            "ALBERT encoder must be divisable by number of hidden groups "
            "({num_hidden_groups})")
    E003 = ("The hidden width of the transformer ({hidden_width}) must be "
            "divisible by the number of self-attention heads ({num_heads})")
    E004 = ("The point-wise feed-forward network in the transformer only "
            "supports the following activation functions: {activation_funcs}")
    E007 = ("Attempting to load the weights of an unsupported Hugging "
            "Face `transformers` model ({unsupported_model}). Currently "
            "supported models: {supported_models}")
    E008 = ("The number of layers in the scalar weighting listener model ({num_layers_scalar_weight})"
            "needs to be the same as the number of hidden layers in the "
            "transformer ({num_layers_transformer}). You can use the config's "
            "interpolation functionality to link the former's `num_layers` "
            "parameter with the latter")
    E009 = ("At least one sequence in the transformer's input has a length "
            "of {seq_len}, which is larger than the model's maximum sequence "
            "length of {max_seq_len} tokens")
    E010 = ("Curated transformers do not currently support listener replacement")
    E011 = ("`{loader_name}` requires the Hugging Face `transformers` package to be installed")
    E012 = ("`{listener_name}` requires the upstream transformer pipe to output "
            "all hidden layer outputs. This can be enabled by setting the pipe's "
            "`all_layer_outputs` parameter to `True` in the pipeline config")
    E013 = ("The target pipe names for gradual transformer unfreezing contain both the "
            "wild-card operator ('*') and individual names. Use either of the two but not both")
    E014 = ("Model '{model_name}' received an unexpected input of type '{input_type}'. "
            "It can only wrap/be chained with models whose outputs are of type  "
            "`TransformerModelOutput` (in almost all cases, models of type `TorchTransformerModelT`)")
    E015 = ("Input passed to the `ScalarWeight` model do not have the same number "
            "layers. Distinct layer counts: {layer_counts}")
    E016 = ("Input passed to the `ScalarWeight` model do not have the same width. "
            "Distinct widths: {hidden_widths}")
    E017 = ("Span extractor stride ({stride}) must be within [window_size / 2, "
            "window_size] ([{half_window_size}, {window_size}])")
    E018 = ("Span extractor batch size must be greater than zero")
    E019 = ("Byte-BPE piece encoder vocabulary doesn't contain '{piece}' piece")
    E020 = ("Character piece encoder vocabulary is not available. Use a loader "
            "to initialize the encoder.")
    E021 = ("Attempting to use the `CharEncoderLoader` piece encoder loader with an "
            "incompatible model ('{model_name}'). It can only be used with the "
            "`CharEncoder` piece encoder")
    E022 = ("Attempting to load an unsupported Hugging Face tokenizer "
            "({unsupported_tokenizer}). Currently supported tokenizers: "
            "{supported_tokenizers}")
    E023 = ("Japanese BERT models currently only support character subword encoding")
    E024 = ("Attempting to initialize an incompatible piece encoder ('{model_name}') "
            "with the Hugging Face Japanese BERT tokenizer. It can only be used with the "
            "`CharEncoder` piece encoder")
    E025 = ("Attempting to perform gradual unfreezing of a non-transformer pipe "
            "('{pipe_name}'}. Only transformer pipes support this feature")
    E026 = ("Attempting to load the weights of a Hugging Face `transformers` model "
            "into an unsupported curated encoder ({unsupported_encoder}). Currently "
            "supported encoders: {supported_encoders}")

# fmt: on
