from networks.net_32 import EncoderVqResnet32, DecoderVqResnet32, DecoderResnet32_multi


class EncoderVq_resnet(EncoderVqResnet32):
    def __init__(self, dim_z, cfgs, flg_bn, flg_var_q):
        super(EncoderVq_resnet, self).__init__(dim_z, cfgs, flg_bn, flg_var_q)
        self.dataset = "MVtecAD"


class DecoderVq_resnet(DecoderVqResnet32):
    def __init__(self, dim_z, cfgs, flg_bn):
        super(DecoderVq_resnet, self).__init__(dim_z, cfgs, flg_bn)
        self.dataset = "MVtecAD"



class Decoder_Multi_resnet(DecoderResnet32_multi):
    def __init__(self, scales, channels, cfgs, flg_bn):
        super(Decoder_Multi_resnet, self).__init__(scales, channels, cfgs, flg_bn)
        self.dataset = "MVtecAD"

