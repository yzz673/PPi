from ..ssl_tasks.ssl_model import SSLModel, LSTMDecoder, ChannelDiscriminator, ContextSwapDiscriminator, LSTMFE


def model_setting(ssl_config):
    fe = LSTMFE(input_size=ssl_config.d_model * 2, hidden_dim=ssl_config.d_model, num_layers=ssl_config.n_enc_layers, device=ssl_config.device)
    decoder = LSTMDecoder(input_size=ssl_config.d_model, hidden_dim=ssl_config.d_model * 2,
                          num_layers=ssl_config.n_enc_layers,
                          device=ssl_config.device)
    judgeDiscriminator = ChannelDiscriminator(indim=ssl_config.d_model)
    ctxtSwpDiscriminator = ContextSwapDiscriminator(indim=ssl_config.d_model * 2)
    model = SSLModel(FE=fe, decoder=decoder, judgeDiscriminator=judgeDiscriminator,
                     ctxtSwpDiscriminator=ctxtSwpDiscriminator).to(ssl_config.device)

    return model
