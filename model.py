from nussl.ml.networks.modules import AmplitudeToDB, BatchNorm, RecurrentStack, Embedding
from torch import nn
import nussl


class MaskInference(nn.Module):
    def __init__(self, num_features, num_audio_channels, hidden_size,
                 num_layers, bidirectional, dropout, num_sources,
                 activation='sigmoid'):
        super().__init__()
        # Converting spectrogram to log spectrogram. ps: human perception of volume is logarithmic
        self.amplitude_to_db = AmplitudeToDB()
        self.input_normalization = BatchNorm(num_features)
        # Stack of Layers of LSTMs, using GRU is not common for source separation.
        # Setting bidirectional parameter to True will turn LSTM layers to bidirectional LSTM layers
        # Process of data in realtime is impossible if we set bidirectional to true
        self.recurrent_stack = RecurrentStack(
            num_features * num_audio_channels, hidden_size,
            num_layers, bool(bidirectional), dropout,
        )
        hidden_size = hidden_size * (int(bidirectional) + 1)
        self.embedding = Embedding(num_features, hidden_size,
                                   num_sources, activation,
                                   num_audio_channels)

    def forward(self, data):
        mix_magnitude = data

        data = self.amplitude_to_db(mix_magnitude)
        data = self.input_normalization(data)
        data = self.recurrent_stack(data)
        mask = self.embedding(data)
        estimates = mix_magnitude.unsqueeze(-1) * mask

        output = {
            'mask': mask,
            'estimates': estimates
        }
        return output

    @classmethod
    def build(cls, num_features, num_audio_channels, hidden_size,
              num_layers, bidirectional, dropout, num_sources,
              activation='sigmoid'):
        nussl.ml.register_module(cls)

        modules = {
            'model': {
                'class': 'MaskInference',
                'args': {
                    'num_features': num_features,
                    'num_audio_channels': num_audio_channels,
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'bidirectional': bidirectional,
                    'dropout': dropout,
                    'num_sources': num_sources,
                    'activation': activation
                }
            }
        }

        connections = [
            ['model', ['mix_magnitude']]
        ]
        for key in ['mask', 'estimates']:
            modules[key] = {'class': 'Alias'}
            connections.append([key, [f'model:{key}']])

        output = ['estimates', 'mask', ]
        config = {
            'name': cls.__name__,
            'modules': modules,
            'connections': connections,
            'output': output
        }
        return nussl.ml.SeparationModel(config)
