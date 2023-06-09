import warnings
import nussl
import scaper
from pathlib import Path
import numpy as np

# Template defines how augmentation mechanism should produce a new random mix
scaper_event_template = {
    'label': ('const', 'vocals'),
    'source_file': ('choose', []),
    'source_time': ('uniform', 0, 7),
    'event_time': ('const', 0),
    'event_duration': ('const', 5.0),
    'snr': ('uniform', -5, 5),
    'pitch_shift': ('uniform', -2, 2),
    'time_stretch': ('uniform', 0.8, 1.2)
}

scaper_object_configuration = {
    'sample_rate': 44100,
    'reference_db': -20,
    'number_of_channels': 1,
    'generated_track_duration': 5
}

# Stem labels in MUSDB18
stem_labels = ['vocals', 'drums', 'bass', 'other']


def create_scaper_object(foreground_dir, background_dir, seed):
    scaper_object = scaper.Scaper(
        duration=scaper_object_configuration['generated_track_duration'],
        fg_path=str(foreground_dir),
        bg_path=str(background_dir),
        random_state=seed
    )
    scaper_object.sr = scaper_object_configuration['sample_rate']
    scaper_object.ref_db = scaper_object_configuration['reference_db']
    scaper_object.n_channels = scaper_object_configuration['number_of_channels']
    return scaper_object


def generate_incoherent_mixture(foreground_dir, background_dir, template, seed):
    scaper_object = create_scaper_object(foreground_dir, background_dir, seed)
    event_parameters = template.copy()
    for stem_label in stem_labels:
        event_parameters['label'] = ('const', stem_label)
        scaper_object.add_event(**event_parameters)
    return scaper_object.generate(fix_clipping=True)


def generate_coherent_mixture(foreground_dir, background_dir, template, seed):
    scaper_object = create_scaper_object(foreground_dir, background_dir, seed)
    event_parameters = template.copy()

    # To generate a coherent mixture, the stems should come from the same song
    scaper_object.add_event(**event_parameters)
    event = scaper_object._instantiate_event(scaper_object.fg_spec[0])
    scaper_object.reset_fg_event_spec()
    event_parameters['source_time'] = ('const', event.source_time)
    event_parameters['pitch_shift'] = ('const', event.pitch_shift)
    event_parameters['time_stretch'] = ('const', event.time_stretch)
    for label in stem_labels:
        event_parameters['label'] = ('const', label)
        coherent_source_file = event.source_file.replace('vocals', label)
        event_parameters['source_file'] = ('const', coherent_source_file)
        scaper_object.add_event(**event_parameters)
    return scaper_object.generate(fix_clipping=True)


def prepare_dataset(dataset, path):

    if not Path('./dataset').is_dir():
        Path('./dataset').mkdir()
    if not Path(path).is_dir():
        # Scaper uses this structure to generate audio mixtures
        # We put all the stems in the foreground folder

        Path(path).mkdir()
        Path('{}/foreground'.format(path)).mkdir()
        Path('{}/background'.format(path)).mkdir()
        print("Generating {} structure for Scaper".format(path))
        for song in dataset:
            title = song['mix'].file_name
            for stem_type_name, data in song['sources'].items():
                path_to_store_stem = Path('{}/foreground/{}'.format(path, stem_type_name))
                path_to_store_stem.mkdir(exist_ok=True)
                data.write_audio_to_file("{}/{}.wav".format(path_to_store_stem, title))
        print("Data is ready at {} Number of items:{}".format(path, len(dataset)))


def generate_audio_mixture(dataset, foreground_folder, background_folder, event_template, seed):
    # Suppress warning
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')

        # Decide whether it should be a coherent or incoherent mixture
        random_state = np.random.RandomState(seed)

        if random_state.rand() > 0.5:
            audio_data = generate_coherent_mixture(foreground_folder, background_folder, event_template, seed)
        else:
            audio_data = generate_incoherent_mixture(foreground_folder, background_folder, event_template, seed)

    mixture_audio, mixture_jam, annotation_list, stem_audio_list = audio_data
    mixture = dataset._load_audio_from_array(
        audio_data=mixture_audio, sample_rate=dataset.sample_rate
    )
    # Convert stems to nussl format
    sources = {}
    jam_annotation = mixture_jam.annotations.search(namespace='scaper')[0]
    for observation, stem_audio in zip(jam_annotation.data, stem_audio_list):
        key = observation.value['label']
        sources[key] = dataset._load_audio_from_array(
            audio_data=stem_audio, sample_rate=dataset.sample_rate
        )
    result = {
        'mix': mixture,
        'sources': sources,
        'metadata': mixture_jam
    }
    return result


class MixClosure:
    def __init__(self, foreground_folder, background_folder, event_template):
        self.foreground_folder = foreground_folder
        self.background_folder = background_folder
        self.event_template = event_template

    def __call__(self, dataset, seed):
        return generate_audio_mixture(dataset,
                                      self.foreground_folder,
                                      self.background_folder,
                                      self.event_template,
                                      seed)


def create_mix_closure(path, subset):
    if subset == "test":
        dataset = nussl.datasets.MUSDB18(subsets=['test'], download=True)
        split_index = (len(dataset)//2) - 1
        dataset = list(dataset)[0:split_index]
    elif subset == "valid":
        dataset = nussl.datasets.MUSDB18(subsets=['test'], download=True)
        split_index = (len(dataset) // 2)
        dataset = list(dataset)[split_index:]
    else:
        dataset = nussl.datasets.MUSDB18(subsets=['train'], download=True)
    prepare_dataset(dataset, path)
    mix_closure = MixClosure('{}/foreground'.format(path), '{}/background'.format(path), scaper_event_template)
    return mix_closure
