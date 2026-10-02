"""
LeafOCR — end-to-end demo app
====================================
Pipeline (all in memory, single image in -> text out):

  1. Take one uploaded palm-leaf image.
  2. Resize it up to a multiple of PATCH_SIZE and cut it into 512x512
     patches (row-major), mirroring how your training patches were made.
  3. Run the DeepLabv3+ segmentation model on every patch to get a
     foreground (leaf/text) mask.
  4. Reconstruct the full-size image + mask, filling the background with
     the manuscript's own average leaf color.
  5. Split the reconstructed mask into text lines (peak/valley detection
     on the row ink profile).
  6. Split each line into ~200px-wide chunks, cutting at low-ink columns.
  7. Run the multi-scale CRNN model on every chunk, CTC-decode, and
     re-assemble chunks -> lines -> full manuscript text.

Before running, fill in the three paths in the CONFIG section below:
  - DEEPLAB_MODEL_PATH : your trained DeepLabv3+ .h5 weights file
  - CRNN_MODEL_PATH    : your trained CRNN inference_model.keras file
  - VOCAB_PATH         : the vocabulary.json saved during CRNN training

Run with:  python app.py
"""

import os
import math
import json
import numpy as np
import cv2

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow import keras

from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

import gradio as gr

# ============================================================
# CONFIG — UPDATE THESE THREE PATHS FOR YOUR MACHINE
# ============================================================

DEEPLAB_MODEL_PATH = r"D:\sfr-system\compariosn_best_deeplabv3plus_model_model.h5"
CRNN_MODEL_PATH     = r"D:\sfr-system\newds_aug\training_model_best.keras"
VOCAB_PATH           = r"D:\sfr-system\newds_aug\vocabulary (6) (1).json"


# Segmentation
PATCH_SIZE = 512  # must match what the DeepLab model was trained on

# Line splitting (row profile of the mask)
SMOOTH_WINDOW   = 9
MIN_PEAK_DIST   = 15
PEAK_PROMINENCE = 0.08
MIN_LINE_ROWS   = 30
ROW_PAD         = 3
TRIM_COLUMNS    = True

# Line -> chunk splitting
TARGET_WIDTH   = 200
TOLERANCE      = 30
INK_MARGIN     = 20
COL_SMOOTH     = 5
MIN_LAST_CHUNK = 50

# CRNN input size (must match training)
CRNN_IMG_WIDTH  = 200
CRNN_IMG_HEIGHT = 50


# ============================================================
# 1. DEEPLABV3+ ARCHITECTURE (must match training exactly so the
#    .h5 weights load correctly)
# ============================================================

def depthwise_separable_conv(x, filters, kernel_size=3, strides=1, dilation_rate=1, name_prefix=''):
    x = layers.DepthwiseConv2D(
        kernel_size=kernel_size, strides=strides, dilation_rate=dilation_rate,
        padding='same', use_bias=False, name=f'{name_prefix}_depthwise'
    )(x)
    x = BatchNormalization(name=f'{name_prefix}_depthwise_bn')(x)
    x = Activation('relu', name=f'{name_prefix}_depthwise_relu')(x)

    x = Conv2D(filters, kernel_size=1, padding='same', use_bias=False, name=f'{name_prefix}_pointwise')(x)
    x = BatchNormalization(name=f'{name_prefix}_pointwise_bn')(x)
    x = Activation('relu', name=f'{name_prefix}_pointwise_relu')(x)
    return x


def atrous_spatial_pyramid_pooling(x, output_stride=16):
    b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
    b0 = BatchNormalization(name='aspp0_bn')(b0)
    b0 = Activation('relu', name='aspp0_activation')(b0)

    atrous_rates = (6, 12, 18) if output_stride != 8 else (12, 24, 36)

    b1 = Conv2D(256, (3, 3), padding='same', dilation_rate=atrous_rates[0], use_bias=False, name='aspp1')(x)
    b1 = BatchNormalization(name='aspp1_bn')(b1)
    b1 = Activation('relu', name='aspp1_activation')(b1)

    b2 = Conv2D(256, (3, 3), padding='same', dilation_rate=atrous_rates[1], use_bias=False, name='aspp2')(x)
    b2 = BatchNormalization(name='aspp2_bn')(b2)
    b2 = Activation('relu', name='aspp2_activation')(b2)

    b3 = Conv2D(256, (3, 3), padding='same', dilation_rate=atrous_rates[2], use_bias=False, name='aspp3')(x)
    b3 = BatchNormalization(name='aspp3_bn')(b3)
    b3 = Activation('relu', name='aspp3_activation')(b3)

    b4 = GlobalAveragePooling2D()(x)
    b4 = tf.keras.layers.Lambda(lambda t: tf.expand_dims(t, 1))(b4)
    b4 = tf.keras.layers.Lambda(lambda t: tf.expand_dims(t, 1))(b4)
    b4 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='image_pooling')(b4)
    b4 = BatchNormalization(name='image_pooling_bn')(b4)
    b4 = Activation('relu', name='image_pooling_activation')(b4)

    def resize_to_feature_map(inputs):
        feature_map, pooled_features = inputs
        shape = tf.shape(feature_map)
        return tf.image.resize(pooled_features, [shape[1], shape[2]])

    b4 = tf.keras.layers.Lambda(resize_to_feature_map)([x, b4])

    x = layers.Concatenate()([b0, b1, b2, b3, b4])
    x = Conv2D(256, (1, 1), padding='same', use_bias=False, name='concat_projection')(x)
    x = BatchNormalization(name='concat_projection_bn')(x)
    x = Activation('relu', name='concat_projection_activation')(x)
    x = Dropout(0.1)(x)
    return x


def deeplabv3plus_model(input_size=(512, 512, 3), output_stride=16):
    inputs = Input(input_size)

    x = Conv2D(32, (3, 3), strides=2, padding='same', use_bias=False, name='entry_flow_conv1_1')(inputs)
    x = BatchNormalization(name='entry_flow_conv1_1_bn')(x)
    x = Activation('relu', name='entry_flow_conv1_1_relu')(x)

    x = Conv2D(64, (3, 3), padding='same', use_bias=False, name='entry_flow_conv1_2')(x)
    x = BatchNormalization(name='entry_flow_conv1_2_bn')(x)
    x = Activation('relu', name='entry_flow_conv1_2_relu')(x)

    # Block 1
    residual = Conv2D(128, (1, 1), strides=2, padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = depthwise_separable_conv(x, 128, name_prefix='entry_flow_block1_1')
    x = depthwise_separable_conv(x, 128, name_prefix='entry_flow_block1_2')
    x = MaxPooling2D((3, 3), strides=2, padding='same')(x)
    x = layers.Add()([x, residual])

    # Block 2
    residual = Conv2D(256, (1, 1), strides=2, padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = Activation('relu')(x)
    x = depthwise_separable_conv(x, 256, name_prefix='entry_flow_block2_1')
    x = depthwise_separable_conv(x, 256, name_prefix='entry_flow_block2_2')
    x = MaxPooling2D((3, 3), strides=2, padding='same')(x)
    x = layers.Add()([x, residual])

    low_level_features = x

    # Block 3
    residual = Conv2D(728, (1, 1), strides=2, padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = Activation('relu')(x)
    x = depthwise_separable_conv(x, 728, name_prefix='entry_flow_block3_1')
    x = depthwise_separable_conv(x, 728, name_prefix='entry_flow_block3_2')
    x = MaxPooling2D((3, 3), strides=2, padding='same')(x)
    x = layers.Add()([x, residual])

    # Middle flow
    for i in range(8):
        residual = x
        prefix = f'middle_flow_unit_{i + 1}'
        x = Activation('relu')(x)
        x = depthwise_separable_conv(x, 728, name_prefix=f'{prefix}_1')
        x = depthwise_separable_conv(x, 728, name_prefix=f'{prefix}_2')
        x = depthwise_separable_conv(x, 728, name_prefix=f'{prefix}_3')
        x = layers.Add()([x, residual])

    # Exit flow
    residual = Conv2D(1024, (1, 1), strides=1, padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = Activation('relu')(x)
    x = depthwise_separable_conv(x, 728, name_prefix='exit_flow_block1_1')
    x = depthwise_separable_conv(x, 1024, name_prefix='exit_flow_block1_2')
    x = layers.Add()([x, residual])

    x = depthwise_separable_conv(x, 1536, name_prefix='exit_flow_block2_1')
    x = depthwise_separable_conv(x, 1536, name_prefix='exit_flow_block2_2')
    x = depthwise_separable_conv(x, 2048, name_prefix='exit_flow_block2_3')

    x = atrous_spatial_pyramid_pooling(x, output_stride)

    low_level_features = Conv2D(48, (1, 1), padding='same', use_bias=False, name='feature_projection0')(low_level_features)
    low_level_features = BatchNormalization(name='feature_projection0_bn')(low_level_features)
    low_level_features = Activation('relu', name='feature_projection0_relu')(low_level_features)

    def upsample_to_match(inputs):
        encoder_features, low_level_features = inputs
        low_level_shape = tf.shape(low_level_features)
        return tf.image.resize(encoder_features, [low_level_shape[1], low_level_shape[2]])

    x = tf.keras.layers.Lambda(upsample_to_match)([x, low_level_features])
    x = layers.Concatenate()([x, low_level_features])

    x = depthwise_separable_conv(x, 256, name_prefix='decoder_conv0')
    x = depthwise_separable_conv(x, 256, name_prefix='decoder_conv1')

    def upsample_to_input_size(t):
        return tf.image.resize(t, [512, 512])

    x = tf.keras.layers.Lambda(upsample_to_input_size)(x)
    outputs = Conv2D(1, kernel_size=1, activation='sigmoid', name='output')(x)

    return Model(inputs=[inputs], outputs=[outputs])


# ============================================================
# 2. CRNN CUSTOM LAYERS (needed to load the saved .keras model)
# ============================================================

class ChannelMeanPool(layers.Layer):
    def call(self, x):
        return tf.reduce_mean(x, axis=-1, keepdims=True)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (1,)


class ChannelMaxPool(layers.Layer):
    def call(self, x):
        return tf.reduce_max(x, axis=-1, keepdims=True)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (1,)


class MainCTCLayer(layers.Layer):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, inputs):
        y_true, y_pred = inputs
        bl = tf.cast(tf.shape(y_true)[0], dtype='int64')
        il = tf.cast(tf.shape(y_pred)[1], dtype='int64') * tf.ones((bl, 1), dtype='int64')
        ll = tf.cast(tf.shape(y_true)[1], dtype='int64') * tf.ones((bl, 1), dtype='int64')
        self.add_loss(self.loss_fn(y_true, y_pred, il, ll))
        return y_pred

    def get_config(self):
        return super().get_config()


class PathCTCLayer(layers.Layer):
    def __init__(self, path_name, initial_weight=0.1, min_weight=0.02, decay_epochs=15, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.path_name = path_name
        self.initial_weight = initial_weight
        self.min_weight = min_weight
        self.decay_epochs = decay_epochs
        self.loss_fn = keras.backend.ctc_batch_cost
        self.epoch = tf.Variable(0.0, trainable=False, dtype=tf.float32)

    def call(self, inputs):
        y_true, y_pred = inputs
        bl = tf.cast(tf.shape(y_true)[0], dtype='int64')
        il = tf.cast(tf.shape(y_pred)[1], dtype='int64') * tf.ones((bl, 1), dtype='int64')
        ll = tf.cast(tf.shape(y_true)[1], dtype='int64') * tf.ones((bl, 1), dtype='int64')
        loss = self.loss_fn(y_true, y_pred, il, ll)
        ne = tf.minimum(self.epoch / self.decay_epochs, 1.0)
        weight = self.min_weight + (self.initial_weight - self.min_weight) * ((1.0 - ne) ** 2)
        self.add_loss(weight * loss)
        return y_pred

    def get_config(self):
        c = super().get_config()
        c.update({'path_name': self.path_name, 'initial_weight': self.initial_weight,
                  'min_weight': self.min_weight, 'decay_epochs': self.decay_epochs})
        return c


# ============================================================
# 3. MODEL LOADING (lazy — happens once, on first request)
# ============================================================

_deeplab_model = None
_crnn_model = None
_num_to_char = None


def get_deeplab_model():
    global _deeplab_model
    if _deeplab_model is None:
        print("Loading DeepLabv3+ segmentation model...")
        m = deeplabv3plus_model(input_size=(PATCH_SIZE, PATCH_SIZE, 3))
        m.load_weights(DEEPLAB_MODEL_PATH)
        _deeplab_model = m
        print("✅ DeepLabv3+ model loaded.")
    return _deeplab_model


# ============================================================
# 3. CRNN MODEL LOADING
# ============================================================

_crnn_training_model = None
_crnn_inference_model = None
_num_to_char = None


def get_crnn_model():

    global _crnn_training_model
    global _crnn_inference_model
    global _num_to_char

    # --------------------------------------------------------
    # Load only once
    # --------------------------------------------------------

    if _crnn_inference_model is None:

        print("Loading CRNN training model...")

        # ----------------------------------------------------
        # Load vocabulary
        # ----------------------------------------------------

        with open(
            VOCAB_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            vocabulary = json.load(f)

        _num_to_char = layers.StringLookup(
            vocabulary=vocabulary,
            mask_token=None,
            invert=True
        )

        # ----------------------------------------------------
        # Allow Lambda/custom objects if required
        # ----------------------------------------------------

        keras.config.enable_unsafe_deserialization()

        # ----------------------------------------------------
        # Load the COMPLETE 5-output training model
        # ----------------------------------------------------

        _crnn_training_model = keras.models.load_model(

            CRNN_MODEL_PATH,

            custom_objects={

                "MainCTCLayer": MainCTCLayer,

                "PathCTCLayer": PathCTCLayer,

                "ChannelMeanPool": ChannelMeanPool,

                "ChannelMaxPool": ChannelMaxPool,

            },

            safe_mode=False
        )

        print(
            "✅ Training model loaded."
        )

        print(
            "Training model inputs:"
        )

        for x in _crnn_training_model.inputs:

            print(
                "  ",
                x,
                x.shape
            )

        print(
            "\nTraining model outputs:"
        )

        for y in _crnn_training_model.outputs:

            print(
                "  ",
                y,
                y.shape
            )

        # ====================================================
        # BUILD INFERENCE MODEL
        # ====================================================

        # IMPORTANT:
        #
        # We take ONLY main_softmax.
        #
        # But we KEEP BOTH inputs:
        #
        # image
        # label
        #
        # because your new training graph has both inputs.
        # ====================================================

        softmax_layer = (
            _crnn_training_model
            .get_layer("main_softmax")
        )

        _crnn_inference_model = keras.Model(

            inputs=_crnn_training_model.inputs,

            outputs=softmax_layer.output,

            name="CRNN_Inference_Model"
        )

        print(
            "\n✅ Inference model created."
        )

        print(
            "Inference inputs:"
        )

        for x in _crnn_inference_model.inputs:

            print(
                "  ",
                x.name,
                x.shape
            )

        print(
            "\nInference output:"
        )

        print(
            _crnn_inference_model.output_shape
        )

    return (
        _crnn_inference_model,
        _num_to_char
    )

# ============================================================
# 4. PATCHING — resize the input image up to a multiple of
#    PATCH_SIZE and cut it into a row-major grid, mirroring how
#    the training patches were generated.
# ============================================================

def resize_to_multiple(img, patch_size=PATCH_SIZE):
    h, w = img.shape[:2]
    new_h = math.ceil(h / patch_size) * patch_size
    new_w = math.ceil(w / patch_size) * patch_size
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    rows = new_h // patch_size
    cols = new_w // patch_size
    return resized, rows, cols


def cut_patches(resized_img, rows, cols, patch_size=PATCH_SIZE):
    """Row-major patches: patch_idx increments across the whole grid."""
    patches = {}
    idx = 0
    for r in range(rows):
        for c in range(cols):
            patch = resized_img[r*patch_size:(r+1)*patch_size, c*patch_size:(c+1)*patch_size]
            patches[idx] = patch
            idx += 1
    return patches


# ============================================================
# 5. SEGMENTATION + RECONSTRUCTION
# ============================================================

def predict_patches_batch(model, patches_bgr):
    """patches_bgr: list of PATCH_SIZExPATCH_SIZEx3 uint8 BGR arrays.
    Runs ALL patches through the model in a single batched call instead of
    one at a time — looping single-image model.predict() calls is very slow
    on CPU because of per-call overhead. Returns list of (patch_rgb, mask)."""
    rgb_list = [cv2.cvtColor(p, cv2.COLOR_BGR2RGB) for p in patches_bgr]
    batch = np.stack([rgb.astype(np.float32) / 255.0 for rgb in rgb_list], axis=0)
    preds = model.predict(batch, verbose=0)
    masks = (preds.squeeze(-1) > 0.5).astype(np.uint8)
    return [(rgb_list[i], masks[i]) for i in range(len(rgb_list))]


def reconstruct(model, orig_bgr):
    h, w = orig_bgr.shape[:2]
    resized, rows, cols = resize_to_multiple(orig_bgr)
    patches = cut_patches(resized, rows, cols)

    indices = sorted(patches.keys())
    batch_results = predict_patches_batch(model, [patches[i] for i in indices])
    patch_results = dict(zip(indices, batch_results))

    fg_sum = np.zeros(3, dtype=np.float64)
    fg_count = 0
    for patch_rgb, patch_mask in patch_results.values():
        fg_pixels = patch_rgb[patch_mask == 1]
        if fg_pixels.size > 0:
            fg_sum += fg_pixels.sum(axis=0)
            fg_count += fg_pixels.shape[0]

    avg_color = (fg_sum / fg_count).astype(np.uint8) if fg_count > 0 else np.array([255, 255, 255], dtype=np.uint8)

    rgb_canvas = np.empty((rows * PATCH_SIZE, cols * PATCH_SIZE, 3), dtype=np.uint8)
    rgb_canvas[:] = avg_color
    mask_canvas = np.zeros((rows * PATCH_SIZE, cols * PATCH_SIZE), dtype=np.uint8)

    for idx, (patch_rgb, patch_mask) in patch_results.items():
        r, c = idx // cols, idx % cols
        masked_patch = np.where(patch_mask[..., None] == 1, patch_rgb, avg_color)
        rgb_canvas[r*PATCH_SIZE:(r+1)*PATCH_SIZE, c*PATCH_SIZE:(c+1)*PATCH_SIZE] = masked_patch
        mask_canvas[r*PATCH_SIZE:(r+1)*PATCH_SIZE, c*PATCH_SIZE:(c+1)*PATCH_SIZE] = patch_mask

    rgb_canvas = cv2.resize(rgb_canvas, (w, h), interpolation=cv2.INTER_LINEAR)
    mask_canvas = cv2.resize(mask_canvas, (w, h), interpolation=cv2.INTER_NEAREST)

    bgr_canvas = cv2.cvtColor(rgb_canvas, cv2.COLOR_RGB2BGR)
    return bgr_canvas, mask_canvas


# ============================================================
# 6. LINE SPLITTING
# ============================================================

def _tighten_band(row_sum, top, bot):
    band = row_sum[top:bot]
    nonzero = np.where(band > 0)[0]
    if len(nonzero) == 0:
        return top, bot
    return top + int(nonzero[0]), top + int(nonzero[-1]) + 1


def split_into_lines(mask):
    row_sum = mask.sum(axis=1).astype(float)
    if row_sum.max() == 0:
        return []

    smooth = uniform_filter1d(row_sum, size=SMOOTH_WINDOW)
    peaks, _ = find_peaks(smooth, distance=MIN_PEAK_DIST, prominence=smooth.max() * PEAK_PROMINENCE)

    if len(peaks) == 0:
        if row_sum.max() < MIN_LINE_ROWS:
            return []
        return [_tighten_band(row_sum, 0, len(row_sum))]

    h = len(row_sum)
    bounds = [0]
    for i in range(len(peaks) - 1):
        a, b = peaks[i], peaks[i + 1]
        valley = a + int(np.argmin(smooth[a:b + 1]))
        bounds.append(valley)
    bounds.append(h)

    lines = []
    for i in range(len(bounds) - 1):
        top, bot = bounds[i], bounds[i + 1]
        if row_sum[top:bot].max() < MIN_LINE_ROWS:
            continue
        lines.append(_tighten_band(row_sum, top, bot))
    return lines


def crop_line(img, mask, top, bot):
    """Returns (crop, top_abs, bot_abs, left_abs, right_abs) — the offsets let
    callers map anything found inside `crop` back to absolute coordinates on
    the full reconstructed image (used for drawing the annotated overlay)."""
    h, w = img.shape[0], img.shape[1]
    top_p = max(0, top - ROW_PAD)
    bot_p = min(h, bot + ROW_PAD)
    crop = img[top_p:bot_p, :]
    mask_band = mask[top_p:bot_p, :]

    left, right = 0, w
    if TRIM_COLUMNS:
        col_fg = mask_band.any(axis=0)
        if col_fg.any():
            left = int(np.argmax(col_fg))
            right = len(col_fg) - int(np.argmax(col_fg[::-1]))
            crop = crop[:, left:right]
    return crop, top_p, bot_p, left, right


# ============================================================
# 7. LINE -> CHUNK SPLITTING
# ============================================================

def ink_column_profile(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    counts = np.bincount(gray.flatten(), minlength=256)
    bg_val = int(np.argmax(counts))
    thresh = bg_val - INK_MARGIN
    ink = (gray < thresh).astype(np.uint8)
    col_sum = ink.sum(axis=0).astype(float)
    return uniform_filter1d(col_sum, size=COL_SMOOTH)


def find_chunk_bounds(col_profile, target=TARGET_WIDTH, tol=TOLERANCE, min_last=MIN_LAST_CHUNK):
    w = len(col_profile)
    if w <= target + tol:
        return [(0, w)]

    bounds = [0]
    pos = 0
    while True:
        remaining = w - pos
        if remaining <= target + tol:
            bounds.append(w)
            break
        lo = max(pos + target - tol, pos + 1)
        hi = min(pos + target + tol, w - 1)
        window = col_profile[lo:hi + 1]
        min_val = window.min()
        candidates = np.where(window == min_val)[0]
        ideal = (pos + target) - lo
        best = candidates[np.argmin(np.abs(candidates - ideal))]
        cut = lo + int(best)
        if cut <= pos:
            cut = pos + target
        bounds.append(cut)
        pos = cut

    if len(bounds) >= 3 and (bounds[-1] - bounds[-2]) < min_last:
        bounds.pop(-2)
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def split_line_into_chunks(line_img):
    if line_img.shape[1] == 0 or line_img.shape[0] == 0:
        return []
    profile = ink_column_profile(line_img)
    bounds = find_chunk_bounds(profile)
    return [line_img[:, left:right] for left, right in bounds]


def split_line_into_chunks_with_bounds(line_img):
    """Same as split_line_into_chunks, but also returns each chunk's
    (left, right) column bounds *within line_img* — needed to compute the
    chunk's absolute rectangle on the full manuscript for the annotated
    overlay."""
    if line_img.shape[1] == 0 or line_img.shape[0] == 0:
        return [], []
    profile = ink_column_profile(line_img)
    bounds = find_chunk_bounds(profile)
    imgs = [line_img[:, left:right] for left, right in bounds]
    return bounds, imgs


# ============================================================
# 8. CRNN RECOGNITION (operates directly on in-memory BGR arrays)
# ============================================================

def _preprocess_chunk(chunk_bgr):
    img_rgb = cv2.cvtColor(chunk_bgr, cv2.COLOR_BGR2RGB)
    img = img_rgb.astype(np.float32) / 255.0
    img = tf.image.resize(img, (CRNN_IMG_HEIGHT, CRNN_IMG_WIDTH))
    img = tf.transpose(img, perm=[1, 0, 2])   # match training: (width, height, channels)
    return img


# ============================================================
# CRNN RECOGNITION
# ============================================================

def _preprocess_chunk(chunk_bgr):

    # --------------------------------------------------------
    # BGR → RGB
    # --------------------------------------------------------

    img_rgb = cv2.cvtColor(
        chunk_bgr,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------------
    # Normalize to [0, 1]
    # --------------------------------------------------------

    img = (
        img_rgb.astype(np.float32)
        / 255.0
    )

    # --------------------------------------------------------
    # Resize exactly as training
    # --------------------------------------------------------

    img = tf.image.resize(
        img,
        (
            CRNN_IMG_HEIGHT,
            CRNN_IMG_WIDTH
        )
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Training:
    # (50, 200, 3)
    #
    # transpose:
    # (200, 50, 3)
    # --------------------------------------------------------

    img = tf.transpose(
        img,
        perm=[1, 0, 2]
    )

    return img


def predict_chunks_batch(
        model,
        num_to_char,
        chunks_bgr
    ):

    """
    Predict all chunks using the NEW
    two-input inference model.

    Model inputs:

        image : (None, 200, 50, 3)
        label : (None, None)

    Model output:

        main_softmax : (None, 300, 81)
    """

    if not chunks_bgr:

        return []


    # ========================================================
    # PREPROCESS ALL CHUNKS
    # ========================================================

    batch = tf.stack(
        [
            _preprocess_chunk(c)
            for c in chunks_bgr
        ],
        axis=0
    )

    print(
        f"CRNN batch shape: {batch.shape}"
    )


    # ========================================================
    # CREATE DUMMY LABEL
    # ========================================================

    batch_size = tf.shape(
        batch
    )[0]

    # The actual label is NOT needed for inference.
    #
    # It is supplied only because the saved training model
    # has two inputs.
    #
    # One dummy value per image is sufficient for this
    # inference graph because we are extracting main_softmax.
    # ========================================================

    dummy_label = tf.zeros(
        (
            batch_size,
            1
        ),
        dtype=tf.float32
    )


    # ========================================================
    # PREDICT
    # ========================================================

    preds = model.predict(

        [
            batch,
            dummy_label
        ],

        verbose=0
    )


    # ========================================================
    # SAFETY CHECK
    # ========================================================

    if isinstance(
        preds,
        list
    ):

        print(
            "WARNING: inference model returned a list."
        )

        preds = preds[0]


    print(
        f"CRNN prediction shape: {preds.shape}"
    )


    # Expected:

    # (batch_size, 300, 81)


    # ========================================================
    # CTC DECODE
    # ========================================================

    input_len = (
        np.ones(
            preds.shape[0]
        )
        * preds.shape[1]
    )


    decoded = keras.backend.ctc_decode(

        preds,

        input_length=input_len,

        greedy=True

    )[0][0].numpy()


    # ========================================================
    # CONVERT INDICES → CHARACTERS
    # ========================================================

    texts = []


    for result in decoded:

        text = ""

        for idx in result:

            if idx != -1:

                char = num_to_char(

                    tf.constant(
                        [idx]
                    )

                )

                text += (
                    char.numpy()[0]
                    .decode("utf-8")
                )

        texts.append(text)


    return texts

def draw_one_chunk(annotated_bgr, rect, in_place=True):
    """Draws a single chunk's highlight box onto annotated_bgr (BGR, mutated
    in place by default). Kept separate from draw_annotations so the live
    pipeline can update the overlay incrementally, one chunk at a time."""
    x1, y1, x2, y2 = rect
    if x2 <= x1 or y2 <= y1:
        return annotated_bgr
    target = annotated_bgr if in_place else annotated_bgr.copy()
    region = target[y1:y2, x1:x2].astype(np.float32)
    tint = region * 0.6 + np.array([0, 200, 0], dtype=np.float32) * 0.4  # BGR green
    target[y1:y2, x1:x2] = tint.clip(0, 255).astype(np.uint8)
    cv2.rectangle(target, (x1, y1), (x2, y2), (0, 200, 0), 2)
    return target


def draw_annotations(recon_bgr, chunk_rects_and_texts):
    """Draws every chunk's highlight in one pass. Returns an RGB image."""
    annotated = recon_bgr.copy()
    for rect, _text in chunk_rects_and_texts:
        draw_one_chunk(annotated, rect, in_place=True)
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)


# ============================================================
# 9. FULL PIPELINE — single image in, everything out.
#    This is a GENERATOR: it yields progressively as each line/chunk is
#    processed, so the UI updates live (highlight box + text appearing one
#    chunk at a time) instead of only showing a result at the very end.
#    The CRNN is still called once per LINE (batched across that line's
#    chunks) for speed — only the on-screen update happens per chunk.
# ============================================================

def run_pipeline(image):
    """image: numpy array (RGB, as given by Gradio) or None."""
    if image is None:
        yield None, None, [], [], "Please upload a palm leaf image first."
        return

    orig_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    deeplab = get_deeplab_model()
    crnn, num_to_char = get_crnn_model()

    # 1. Segment + reconstruct
    recon_bgr, mask = reconstruct(deeplab, orig_bgr)

    # Mask overlay for display (red tint over detected foreground)
    overlay = recon_bgr.copy()
    overlay[mask == 1] = (0.5 * overlay[mask == 1] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    # 2. Split into lines — keep each line's absolute (top, bottom, left)
    #    offset in the full image so chunk positions can be mapped back later.
    line_bounds = split_into_lines(mask)
    if not line_bounds:
        yield overlay_rgb, None, [], [], "No text lines were detected in this image."
        return

    line_crops = []
    line_offsets = []
    for top, bot in line_bounds:
        crop, top_p, bot_p, left, _right = crop_line(recon_bgr, mask, top, bot)
        line_crops.append(crop)
        line_offsets.append((top_p, bot_p, left))

    line_gallery = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in line_crops if c.size > 0]

    # Show the plain reconstructed image (no highlights yet) plus the
    # detected lines, before recognition starts.
    annotated_bgr = recon_bgr.copy()
    yield overlay_rgb, cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB), line_gallery, [], ""

    # 3. Walk line by line. Within each line, batch all of that line's
    #    chunks through the CRNN in ONE call (fast), then reveal them on
    #    screen one at a time (live).
    lines_by_idx = {}
    chunk_gallery = []

    for line_idx, (line_img, (top_p, bot_p, left)) in enumerate(zip(line_crops, line_offsets), start=1):
        bounds, imgs = split_line_into_chunks_with_bounds(line_img)
        if not imgs:
            continue

        texts = predict_chunks_batch(crnn, num_to_char, imgs)

        for chunk_idx, ((cl, cr), chunk_img, text) in enumerate(zip(bounds, imgs, texts), start=1):
            if chunk_img.size == 0:
                continue
            rect = (left + cl, top_p, left + cr, bot_p)

            lines_by_idx.setdefault(line_idx, []).append(text)
            caption = f"L{line_idx:02d} C{chunk_idx:02d}: {text}"
            chunk_gallery.append((cv2.cvtColor(chunk_img, cv2.COLOR_BGR2RGB), caption))
            draw_one_chunk(annotated_bgr, rect, in_place=True)

            current_text = '\n'.join(
                ' '.join(lines_by_idx[i]) for i in sorted(lines_by_idx.keys())
            )
            yield (
                overlay_rgb,
                cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB),
                line_gallery,
                chunk_gallery,
                current_text,
            )


# ============================================================
# 10. GRADIO UI
# ============================================================

with gr.Blocks(title="Palm Leaf OCR") as demo:
    gr.Markdown(
        "# LeafOCR \n"
        "Upload a palm leaf  image. The pipeline segments text lines with "
        "DeepLabv3+, splits each line into chunks, recognizes each chunk with the "
        "multi-scale CRNN, and reassembles the final text."
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="Palm Leaf Image", type="numpy")
            run_btn = gr.Button("Run OCR", variant="primary")
            annotated_output = gr.Image(label="Live: each chunk highlighted as it's recognized")
        with gr.Column(scale=1):
            output_text = gr.Textbox(label="Recognized Text", lines=12)

    with gr.Accordion("Intermediate steps", open=False):
        mask_output = gr.Image(label="Segmentation mask (overlay)")
        line_output = gr.Gallery(label="Detected lines", columns=1, object_fit="contain")
        chunk_output = gr.Gallery(label="Line chunks (with predicted text)", columns=4, object_fit="contain")

    run_btn.click(
        fn=run_pipeline,
        inputs=[input_image],
        outputs=[mask_output, annotated_output, line_output, chunk_output, output_text],
    )

if __name__ == "__main__":
    demo.launch()
