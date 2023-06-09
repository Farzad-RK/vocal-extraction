from data_augmentation import create_mix_closure
from nussl.datasets import transforms as nussl_transforms
from model import MaskInference
import nussl
import torch
from pathlib import Path
import json
import numpy as np
import glob

processing_device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_model(maximum_number_of_mixtures=1000):
    print("Using {} for pytorch training".format(processing_device))

    # Transform pipline for compatibility with pytorch
    composed_transforms = nussl_transforms.Compose([
        nussl_transforms.SumSources([['bass', 'drums', 'other']]),
        nussl_transforms.MagnitudeSpectrumApproximation(),
        nussl_transforms.IndexSources('source_magnitudes', 1),
        nussl_transforms.ToSeparationModel(),
    ])
    test_composed_transforms = nussl_transforms.Compose([nussl_transforms.SumSources([['bass', 'drums', 'other']])])

    train_mix_closure = create_mix_closure('./dataset/train', 'train')
    valid_mix_closure = create_mix_closure('./dataset/valid', 'valid')
    test_mix_closure = create_mix_closure('./dataset/test', 'test')

    # STFT parameters
    stft_params = nussl.STFTParams(window_length=512, hop_length=128, window_type='sqrt_hann')

    # onTheFly mixer parameters
    number_of_channels = 1
    sample_rate = 44100

    # Training parameters
    learning_rate = 1e-3
    batch_size = 10
    number_of_workers = 1

    train_data = nussl.datasets.OnTheFly(transform=composed_transforms,
                                         stft_params=stft_params,
                                         num_channels=number_of_channels,
                                         sample_rate=sample_rate,
                                         num_mixtures=maximum_number_of_mixtures,
                                         mix_closure=train_mix_closure)
    train_dataloader = torch.utils.data.DataLoader(train_data, num_workers=number_of_workers, batch_size=batch_size)

    valid_data = nussl.datasets.OnTheFly(transform=composed_transforms,
                                         stft_params=stft_params,
                                         num_channels=number_of_channels,
                                         sample_rate=sample_rate,
                                         num_mixtures=maximum_number_of_mixtures,
                                         mix_closure=valid_mix_closure)

    valid_dataloader = torch.utils.data.DataLoader(valid_data, num_workers=number_of_workers, batch_size=batch_size)

    test_data = nussl.datasets.OnTheFly(transform=test_composed_transforms,
                                        stft_params=stft_params,
                                        num_channels=number_of_channels,
                                        sample_rate=sample_rate,
                                        num_mixtures=100,
                                        mix_closure=test_mix_closure)

    # Initializing model
    model = create_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Absolute differences between the true value and the predicted value.
    loss_function = nussl.ml.train.loss.L1Loss()

    model.verbose = True

    # Using pytorch ignite engine
    def train_step(engine, batch):
        optimizer.zero_grad()
        output = model(batch)
        loss = loss_function(
            output['estimates'],
            batch['source_magnitudes']
        )

        loss.backward()
        optimizer.step()

        loss_vals = {
            'L1Loss': loss.item(),
            'loss': loss.item()
        }

        return loss_vals

    def val_step(engine, batch):
        with torch.no_grad():
            output = model(batch)
        loss = loss_function(
            output['estimates'],
            batch['source_magnitudes']
        )
        loss_vals = {
            'L1Loss': loss.item(),
            'loss': loss.item()
        }
        return loss_vals

    trainer, validator = nussl.ml.train.create_train_and_validation_engines(
        train_step, val_step, device=processing_device
    )
    output_folder = Path('./').absolute()

    nussl.ml.train.add_stdout_handler(trainer, validator)
    nussl.ml.train.add_validate_and_checkpoint(output_folder, model,
                                               optimizer, train_data, trainer, valid_dataloader, validator)
    trainer.run(
        train_dataloader,
        epoch_length=32,
        max_epochs=25
    )


def create_model():
    stft_params = nussl.STFTParams(window_length=512, hop_length=128, window_type='sqrt_hann')
    number_of_features = stft_params.window_length // 2 + 1
    number_of_channels = 1
    hidden_size = 50
    number_of_layers = 2
    is_bidirectional = True
    dropout = 0.3
    number_of_sources = 1
    activation_function = 'sigmoid'
    model = MaskInference.build(num_features=number_of_features,
                                num_audio_channels=number_of_channels,
                                hidden_size=hidden_size,
                                num_layers=number_of_layers,
                                bidirectional=is_bidirectional,
                                dropout=dropout,
                                num_sources=number_of_sources,
                                activation=activation_function
                                ).to(processing_device)

    return model


def evaluate_model(path_to_trained_model='checkpoints/best.model.pth'):
    loaded_model = torch.load(path_to_trained_model, map_location=processing_device)
    model = create_model()
    model.load_state_dict(loaded_model['state_dict'])
    separator = nussl.separation.deep.DeepMaskEstimation(
        nussl.AudioSignal(), model_path=None,
        device=processing_device,
    )

    # We have to load the model manually, it's a workaround for saved pytorch models
    separator.device = processing_device
    separator.model = model
    separator.config = loaded_model['metadata']['config']
    separator.metadata.update(loaded_model['metadata'])
    separator.transform = separator._get_transforms(loaded_model['metadata']['train_dataset']['transforms'])

    tfm = nussl_transforms.Compose([
        nussl_transforms.SumSources([['bass', 'drums', 'other']]),
    ])
    test_dataset = nussl.datasets.MUSDB18(subsets=['test'], transform=tfm)

    # Iterate over test dataset for evaluation
    # In the augmentation module we have used the second half of the test set as validation set
    # therefor, we only use the first half of the test set for the evaluation of the model.
    split_index = (len(test_dataset) // 2) - 1
    test_dataset = list(test_dataset)[0:split_index]
    output_folder = './evaluation_results'
    for i in range(len(test_dataset)):
        item = test_dataset[i]
        print("Processing {}".format(item['mix'].file_name))
        separator.audio_signal = item['mix']
        estimates = separator()

        source_keys = list(item['sources'].keys())
        estimates = {
            'vocals': estimates[0],
            'bass+drums+other': item['mix'] - estimates[0]
        }
        sources = [item['sources'][k] for k in source_keys]
        estimates = [estimates[k] for k in source_keys]

        evaluator = nussl.evaluation.BSSEvalScale(
            sources, estimates, source_labels=source_keys
        )
        scores = evaluator.evaluate()
        output_folder = Path(output_folder).absolute()
        output_folder.mkdir(exist_ok=True)
        output_file = output_folder / sources[0].file_name.replace('wav', 'json')
        with open(output_file, 'w') as f:
            json.dump(scores, f, indent=4)

    print('Results are stored in {}'.format(output_folder))
    json_files = glob.glob(f"./evaluation_results/*.json")
    df = nussl.evaluation.aggregate_score_files(
        json_files, aggregator=np.nanmedian)
    report_card = nussl.evaluation.report_card(
        df, report_each_source=True)
    print(report_card)


def main():
    is_running = True
    option = None
    while is_running:
        try:
            print("Make sure that Sox and FFMPEG environment variables are set\n"
                  "Select an operation\n"
                  "\n"
                  "1.Train MaskInference model \n"
                  "2.Evaluate the model with test set \n"
                  "3.Evaluate with a pretrained model (1000 mixtures, 25 epochs, 10 hours of training) \n"
                  "4.Exit \n"
                  )
            option = int(input('Please select an operation: '))
        except:
            print("Invalid option")
            continue
        if 1 <= option <= 4:
            if option == 1:
                train_model()
            elif option == 2:
                evaluate_model()
            elif option == 3:
                evaluate_model(path_to_trained_model='pretrained_model/best.model.pth')
            else:
                exit(0)
        else:
            print("Invalid option")
            continue


if __name__ == '__main__':
    main()
