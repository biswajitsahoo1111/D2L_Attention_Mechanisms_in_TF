from saved_class_Decoder import Decoder

class AttentionDecoder(Decoder):
    """The base attention-based decoder interfae."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    @property
    def attention_weights(self):
        raise NotImplementedError