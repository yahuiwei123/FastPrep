"""SynthSeg 2.0 standard (non-robust, non-fast) stages with explicit file ports.

Parcellation consumes raw cortex support, QC consumes raw channel indices.
These must be captured BEFORE topological postprocessing. No MNI registration
is performed: preprocessing resamples to 1 mm and temporarily reorients to RAS.
"""
import argparse
import base64
import json
import os
from pathlib import Path
import sys
import time

import nibabel as nib
import numpy as np

SOURCE = Path(__file__).resolve().parent / 'source'
sys.path.insert(0, str(SOURCE))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='operation', required=True)
    definitions = {
        'preprocess': ['image', 'output-image', 'output-geometry'],
        'segment': ['image', 'checkpoint', 'labels', 'denoiser-labels', 'output'],
        'segment-base': ['image', 'checkpoint', 'labels', 'denoiser-labels', 'geometry',
                         'topology', 'output-segmentation', 'output-volumes'],
        'select-cortex': ['posteriors', 'labels', 'output'],
        'select-qc-indices': ['posteriors', 'output'],
        'cortex-parcellate': ['image', 'cortex', 'checkpoint', 'parcellation-labels', 'output'],
        'qc': ['indices', 'checkpoint', 'labels', 'qc-labels', 'output'],
        'postprocess': ['posteriors', 'geometry', 'labels', 'topology', 'output-segmentation', 'output-volumes'],
    }
    for op, flags in definitions.items():
        s = sub.add_parser(op)
        for flag in flags:
            s.add_argument('--' + flag, required=True, type=Path)
        if op in ('segment', 'segment-base'):
            s.add_argument('--sequential-flip', action='store_true',
                           help='Run the unchanged left/right ensemble sequentially to reduce GPU memory')
        if op == 'cortex-parcellate':
            s.add_argument('--smooth', action='store_true', help='Match upstream --parc WITHOUT --qc; with QC upstream returns unsmoothed parcellation')
        if op == 'postprocess':
            s.add_argument('--output-posteriors', type=Path,
                           help='Explicitly export posteriors; omitted in the FastPrep pipeline')
            s.add_argument('--parcellation', type=Path)
            s.add_argument('--parcellation-labels', type=Path)
        s.add_argument('--threads', type=int, default=4)
        s.add_argument('--device', default='cpu')
        s.add_argument('--runtime-info', type=Path)
    return p


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as f:
        np.save(f, value, allow_pickle=False)


def label_info(path):
    from SynthSeg.predict import get_flip_indices
    return get_flip_indices(np.load(path, allow_pickle=False), 19)


def predict_sequential_flip(net, image, flip_indices):
    posteriors = net.predict(image, verbose=0)
    flipped = net.predict(np.flip(image, axis=1), verbose=0)
    restored = np.flip(flipped, axis=1)
    if posteriors.shape != restored.shape or posteriors.shape[-1] != len(flip_indices):
        raise ValueError('Invalid posterior shape for the left/right ensemble')
    for channel, flipped_channel in enumerate(flip_indices):
        posteriors[..., channel] += restored[..., flipped_channel]
    posteriors *= 0.5
    return posteriors


def configure_tensorflow(requested_device, threads):
    if threads < 1:
        raise ValueError('threads must be positive')
    device_index = 0
    if requested_device not in ('cpu', 'cuda', 'auto'):
        index_text = requested_device.removeprefix('cuda:')
        if not index_text.isdecimal():
            raise ValueError('device must be cpu, cuda, auto, or a non-negative visible GPU index')
        device_index = int(index_text)
    if requested_device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import tensorflow as tf
    tf.config.threading.set_inter_op_parallelism_threads(threads)
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    runtime = {'requested_device': requested_device, 'device': 'cpu',
               'tensorflow_version': tf.__version__,
               'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}
    devices = tf.config.list_physical_devices('GPU') if requested_device != 'cpu' else []
    if requested_device == 'cpu' or requested_device == 'auto' and not devices:
        tf.config.set_visible_devices([], 'GPU')
        return tf, runtime
    if device_index >= len(devices):
        raise RuntimeError(f'SynthSeg GPU {device_index} requested, but TensorFlow sees {len(devices)} GPUs; no CPU fallback')
    selected = devices[device_index]
    tf.config.set_visible_devices([selected], 'GPU')
    tf.config.experimental.set_memory_growth(selected, True)
    with tf.device('/GPU:0'):
        probe = tf.nn.conv3d(tf.ones((1, 4, 4, 4, 1)), tf.ones((3, 3, 3, 1, 1)),
                             strides=[1, 1, 1, 1, 1], padding='VALID')
        probe.numpy()
    if 'GPU:0' not in probe.device.upper():
        raise RuntimeError('SynthSeg convolution did not execute on GPU; no CPU fallback')
    runtime.update(device=f'cuda:{device_index}', tensorflow_device=probe.device,
                   gpu_name=tf.config.experimental.get_device_details(selected).get('device_name'))
    print('SynthSeg GPU execution verified: ' + json.dumps(runtime), flush=True)
    return tf, runtime


def postprocess_base(a, posteriors, upstream, utils):
    """Apply the unchanged upstream postprocessing, exporting labels and volumes."""
    start = time.monotonic()
    parcellation = getattr(a, 'parcellation', None)
    parcellation_labels = getattr(a, 'parcellation_labels', None)
    if (parcellation is None) != (parcellation_labels is None):
        raise ValueError('parcellation and parcellation-labels must be supplied together')
    labels, _, indices = label_info(a.labels)
    info = json.loads(a.geometry.read_text())
    parc_labels = np.unique(np.load(parcellation_labels)) if parcellation else None
    seg, post, volumes = upstream.postprocess(posteriors,
        np.load(parcellation) if parcellation else None, info['shape'], np.array(info['pad_idx']),
        info['crop_idx'], labels, parc_labels, np.array(info['affine']), np.array(info['resolution']),
        False, np.load(a.topology)[indices], False)
    processed = time.monotonic()
    h = nib.Nifti1Header(binaryblock=base64.b64decode(info['header']))
    outputs = [(a.output_segmentation, seg, 'int32')]
    if getattr(a, 'output_posteriors', None) is not None:
        outputs.append((a.output_posteriors, post, 'float32'))
    for path, data, dtype in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        utils.save_volume(data, np.array(info['affine']), h, str(path), dtype=dtype)
    save(a.output_volumes, volumes)
    return {'postprocess': processed - start, 'save_outputs': time.monotonic() - processed}


def main(argv=None):
    a = parser().parse_args(argv)
    start = time.monotonic()
    tf, runtime = configure_tensorflow(a.device, a.threads)
    from SynthSeg import predict_synthseg as upstream
    from ext.lab2im import utils
    runtime['timing_seconds'] = {'initialization': time.monotonic() - start}
    if a.operation == 'preprocess':
        image = nib.load(a.image)
        if not isinstance(image, nib.Nifti1Image) or len(image.shape) != 3:
            raise ValueError('This atomic geometry serialization requires a 3D NIfTI-1 image')
        im, aff, h, res, shape, pad, crop = upstream.preprocess(str(a.image), ct=False, min_pad=128)
        save(a.output_image, im)
        info = {'affine': aff.tolist(), 'resolution': np.asarray(res).tolist(), 'shape': shape,
                'pad_idx': np.asarray(pad).tolist(), 'crop_idx': None if crop is None else np.asarray(crop).tolist(),
                'header': base64.b64encode(h.binaryblock).decode(), 'source': str(a.image.resolve())}
        a.output_geometry.parent.mkdir(parents=True, exist_ok=True)
        a.output_geometry.write_text(json.dumps(info, indent=2) + '\n')
    elif a.operation in ('segment', 'segment-base'):
        start = time.monotonic()
        labels, flip, _ = label_info(a.labels)
        net = upstream.build_model(str(a.checkpoint), None, None, 224, labels,
            np.unique(np.load(a.denoiser_labels)), None, None, 0.5,
            None if a.sequential_flip else flip, False, False, False)
        runtime['model_weight_devices'] = sorted({weight.device for weight in net.weights})
        if runtime['device'] != 'cpu' and any('GPU:' not in weight.device.upper() for weight in net.weights):
            raise RuntimeError('SynthSeg model weights are not on GPU; no CPU fallback')
        runtime['flip_ensemble_execution'] = 'sequential' if a.sequential_flip else 'combined_graph'
        image = np.load(a.image, allow_pickle=False)
        runtime['timing_seconds']['model_and_input_loading'] = time.monotonic() - start
        start = time.monotonic()
        posteriors = predict_sequential_flip(net, image, flip) if a.sequential_flip else net.predict(image, verbose=0)
        runtime['timing_seconds']['inference_and_ensemble'] = time.monotonic() - start
        if a.operation == 'segment-base':
            del net, image
            runtime['timing_seconds'].update(postprocess_base(a, posteriors, upstream, utils))
            runtime['posterior_storage'] = 'memory_only'
        else:
            save(a.output, posteriors)
    elif a.operation == 'select-cortex':
        labels, _, _ = label_info(a.labels)
        post = np.load(a.posteriors)
        if post.shape[-1] != len(labels) or not np.isfinite(post).all():
            raise ValueError('Invalid posterior channels')
        raw_labels = labels[post.argmax(-1)]
        save(a.output, np.isin(raw_labels, [3, 42]).astype(np.int32))
    elif a.operation == 'select-qc-indices':
        post = np.load(a.posteriors)
        if not np.isfinite(post).all() or post.ndim != 5:
            raise ValueError('Expected finite batch/spatial/channel posteriors')
        save(a.output, post.argmax(-1).astype(np.int32))
    elif a.operation == 'cortex-parcellate':
        net = parcellation_model(a.checkpoint, len(np.unique(np.load(a.parcellation_labels))), a.smooth)
        im, cortex = np.load(a.image), np.load(a.cortex)
        if cortex.shape != im.shape[:-1] or not np.isin(cortex, [0, 1]).all():
            raise ValueError('Cortex must be raw argmax support on the padded model grid')
        save(a.output, net.predict([im, cortex], verbose=0))
    elif a.operation == 'qc':
        labels, _, indices = label_info(a.labels)
        qc_labels = np.load(a.qc_labels)[indices]
        raw = np.load(a.indices)
        if raw.ndim != 4 or not np.issubdtype(raw.dtype, np.integer) or raw.min() < 0 or raw.max() >= len(labels):
            raise ValueError('QC consumes raw channel INDICES, not FreeSurfer label IDs')
        net = qc_model(a.checkpoint, labels, qc_labels, upstream.MakeShape)
        scores = net.predict([raw, np.array(raw.shape[1:])[None]], verbose=0)
        save(a.output, np.around(np.clip(np.squeeze(scores)[1:], 0, 1), 4))
    else:
        runtime['timing_seconds'].update(postprocess_base(a, np.load(a.posteriors), upstream, utils))
    if a.runtime_info is not None:
        a.runtime_info.parent.mkdir(parents=True, exist_ok=True)
        a.runtime_info.write_text(json.dumps(runtime, indent=2) + '\n')


def parcellation_model(checkpoint, channels, smooth):
    import tensorflow as tf
    from keras import layers as KL
    from keras.models import Model
    from ext.neuron import models
    from ext.lab2im.layers import GaussianBlur
    image = KL.Input([None, None, None, 1], name='atomic_image')
    cortex = KL.Input([None, None, None], dtype='int32', name='atomic_cortex')
    onehot = KL.Lambda(lambda x: tf.one_hot(x, depth=2, axis=-1))(cortex)
    combined = KL.Lambda(lambda x: tf.cast(tf.concat(x, axis=-1), 'float32'))([image, onehot])
    net = models.unet(input_model=Model([image, cortex], combined), input_shape=[None, None, None, 3],
        nb_labels=channels, nb_levels=5, nb_conv_per_level=2, conv_size=3, nb_features=24,
        feat_mult=2, activation='elu', batch_norm=-1, name='unet_parc')
    net.load_weights(str(checkpoint), by_name=True)
    if smooth:
        last = net.output
        last._keras_shape = tuple(last.get_shape().as_list())
        net = Model(net.inputs, GaussianBlur(sigma=0.5)(last))
    return net


def qc_model(checkpoint, labels, qc_labels, make_shape):
    import tensorflow as tf
    from keras import layers as KL
    from keras.models import Model
    from ext.neuron import models
    from ext.lab2im.layers import ConvertLabels
    n = len(np.unique(qc_labels))
    indices = KL.Input([None, None, None], dtype='int32', name='atomic_indices')
    shape = KL.Input([3], dtype='int32', name='atomic_shape')
    last = make_shape(224)([indices, shape])
    last = ConvertLabels(np.arange(len(labels)), labels)(last)
    last = ConvertLabels(labels, qc_labels)(last)
    last = KL.Lambda(lambda x: tf.one_hot(tf.cast(x, 'int32'), depth=n, axis=-1))(last)
    net = models.conv_enc(input_model=Model([indices, shape], last), input_shape=[None, None, None, 1],
        nb_levels=4, nb_conv_per_level=2, conv_size=5, nb_features=24, feat_mult=2,
        activation='relu', batch_norm=-1, use_residuals=True, name='qc')
    last = KL.MaxPool3D(pool_size=(2, 2, 2), name='qc_maxpool_3', padding='same')(net.outputs[0])
    for i in range(2):
        last = KL.Conv3D(n, kernel_size=5, padding='same', activation='relu',
                        data_format='channels_last', name='qc_final_conv_' + str(i))(last)
    last = KL.Lambda(lambda x: tf.reduce_mean(x, axis=[1, 2, 3]), name='qc_final_pred')(last)
    net = Model(net.inputs, last)
    net.load_weights(str(checkpoint), by_name=True)
    return net


if __name__ == '__main__':
    main()
