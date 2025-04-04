"""Fiddle config for the dataloader of APC models."""

import functools
from collections.abc import Callable, Sequence

from fiddle.experimental import auto_config

from wlt import wpath
from wlt.conf.dataloaders import data_config_utils
from wlt.diffusion import constants
from wlt.diffusion.patisserie import (
    abm_transforms,
    aic_transforms,
    apc_transforms,
    bucket_transforms,
)
from wlt.patisserie import (
    dag,
    data_sources,
    pipeline,
    timed_decoders,
    timestamp_sampling,
    transformations,
)
from wlt.patisserie.transformations import video_transforms

APC_TRAIN_LANCE_PATH = "gs://wlt-data-internal-us-west4/lance_snapshots/wan_realestate10k_train_250323-0528.lance"
APC_TEST_LANCE_PATH = "gs://wlt-data-internal-us-west4/lance_snapshots/wan_realestate10k_val_250323-0500.lance"

CAPTION_LANCE_COLUMN = "caption"
TEXT_EMBED_LANCE_COLUMN = "caption_embedding_path"
TEXT_EMBED_NUM_TOKENS_LANCE_COLUMN = "caption_embedding_num_tokens"
APC_LANCE_COLUMNS = [
    "dataset_id",
    "wlt_id",
    "shot_id",
    "subshot_id",
    "start_pts_ms",
    "duration_ms",
    "rgb_480min",
    "video_index_bytes",
    "camera_capture",
    "disparity_90percentile",
    "posing_keyframe_ts",
    CAPTION_LANCE_COLUMN,
    TEXT_EMBED_LANCE_COLUMN,
    TEXT_EMBED_NUM_TOKENS_LANCE_COLUMN,
]


def build_apc_input_graph(
    is_training: bool, lance_filter: str | None = None, version: int | str | None = None
) -> dag.Node:
    """Returns the input graph for APC TIP2V patisserie dataloder."""
    data_dir = APC_TRAIN_LANCE_PATH if is_training else APC_TEST_LANCE_PATH
    data_source = data_sources.LanceBatchedDataSource(
        data_dir=wpath.ensure_pathlike(data_dir),
        batch_size=64,  # This is the batch size for reading lance, not for train/test.
        columns=APC_LANCE_COLUMNS,
        filter=lance_filter,
        version=version,
        drop_last=False,
    )
    sampler = data_config_utils.build_sampler(data_source, is_training=is_training)

    return pipeline.create_source(data_source, sampler)


def build_apc_data_graph(
    input_graph: dag.Node | Callable[[], dag.Node],
    batch_size: int,
    num_frames: int,
    num_pixel_threshold: int,
    hw_bucket_list: Sequence[tuple[int, int]],
    image_size_multiple_of: int,
    frame_skip: int,
) -> list[transformations.Transform]:
    """
    Helper function to build data graph for APC TIP2V patisserie dataloder.

    Args:
        input_graph: The input graph to extend upon for the data pipeline.
        batch_size: The per-gpu batch size used for training/evaluation.
        num_frames: The number of frames to sample from each shot.
        num_pixel_threshold: The number of pixels that the target size should be below.
            e.g., `192*192` means that the final processed image should have pixels
            less than `192*192`.
        vae_spatial_compression: The spatial compression factor for VAE, used to
            downsample the torchcam before computing pose embeddings.
        min_num_cond_frames: The minimum number of conditional frames.
        max_num_cond_frames: The maximum number of conditional frames.
        hw_bucket_list: A sequence of (height, width) that is used to define all
            aspect ratio buckets. e.g. `((9, 16), (16, 9), (1, 1))` means that the
            dataloader can yield batches of aspect ratio 9:16, 16:9, or 1:1.
        image_size_multiple_of: The height and width of the final processed images
            should be divisible by this number. Default to 16, considering the 8x VAE
            downsample size and 2x patchify size.
        camera_require_exact_timestamps: Whether to require exact timestamps for
            cameras. Set this flag to True can make sure camera and rgb are perfectly
            aligned.
        min_framerate: The minimum framerate for random framerate sampling.
        max_framerate: The maximum framerate for random framerate sampling.
        use_dummy_timestamps: If True, use dummy (all zero) timestamps.
        remove_cond_frames_from_input_frames: Whether to remove conditional frames
            from input frames such that the input frames and conditional frames are
            different.
        interpolate_poses: Whether to interpolate poses from posing keyframes.
    Returns:
        A list of data transforms for the APC TIP2V patisserie dataloder.
    """
    timestamp_tranform = timestamp_sampling.TimestampSamplerTransform(
        timestamp_sampler=timestamp_sampling.StridedTimestampSampler(
            order_mode="ascending",
            use_posing_keyframes=False,
            sample_size=num_frames,
            stride=frame_skip,
            use_keyframes=False,
        ),
        timestamp_key=constants.TIMESTAMP_KEY,
    )
    return pipeline.chain(
        input_graph() if callable(input_graph) else input_graph,
        transformations.FlattenBatchTransform(),
        timestamp_tranform,
        video_transforms.VideoDecodeTransformV2(
            decoders={
                constants.CAPTION_KEY: timed_decoders.IdentityDecoder(
                    column=CAPTION_LANCE_COLUMN,
                ),
                constants.UMT5XXL_TXT_EMBED_KEY: timed_decoders.TextEmbedDecoder(
                    text_embed_column=TEXT_EMBED_LANCE_COLUMN,
                ),
                constants.NUM_TXT_TOKENS_KEY: timed_decoders.IdentityDecoder(
                    column=TEXT_EMBED_NUM_TOKENS_LANCE_COLUMN,
                ),
                constants.POSED_IMAGE_KEY: timed_decoders.PosedImageDecoder(
                    rgb_decoder=timed_decoders.RGBFrameDecoder(),
                    camera_decoder=timed_decoders.CameraDecoder(
                        scale_by_disparity=True,
                        require_exact_timestamps=False,
                        interpolate_poses=True,
                    ),
                ),
            },
            add_timestamps=True,
            timestamp_key=constants.TIMESTAMP_KEY,
        ),
        aic_transforms.NormalizeVideoTimestampsTransform(relative_timestamps=True),
        aic_transforms.AddValidFrameMaskTransform(
            image_key=constants.POSED_IMAGE_KEY,
            valid_frame_mask_key=constants.VALID_FRAME_MASK_KEY,
            mode="ones",
        ),
        apc_transforms.CameraRelativeTransform(
            posed_image_key_list=(constants.POSED_IMAGE_KEY,),
            ref_posed_image_key=constants.POSED_IMAGE_KEY,
            relative_mode="random",
            # Following Dec2 setting.
            max_radius=20.0,
        ),
        bucket_transforms.FindAspectRatioBucketTransform(
            hw_bucket_list=hw_bucket_list,
            num_pixel_threshold=num_pixel_threshold,
            image_size_multiple_of=image_size_multiple_of,
            image_key=constants.POSED_IMAGE_KEY,
        ),
        apc_transforms.ResizeCropPosedImageTransform(
            posed_image_key=constants.POSED_IMAGE_KEY,
            resize_method="lanczos3",
            clamp_rgb=True,
        ),
        apc_transforms.DisassemblePosedImageTransform(
            posed_image_key=constants.POSED_IMAGE_KEY,
            rgb_key=constants.RGB_KEY,
            camera_key=constants.CAMERA_KEY,
        ),
        aic_transforms.SampleConditionalFrameTransform(
            video_frame_to_cond_frame_mapping={
                constants.RGB_KEY: constants.COND_FRAMES_KEY,
                constants.TIMESTAMP_KEY: constants.COND_FRAMES_TIMESTAMP_KEY,
                constants.VALID_FRAME_MASK_KEY: constants.COND_FRAMES_MASK_KEY,
                # constants.POSE_EMBED_KEY: constants.COND_FRAMES_POSE_EMBED_KEY,
            },
            min_num_cond_frames=1,
            max_num_cond_frames=1,
            remove_cond_frames_from_input_frames=False,
            use_first_frame_as_cond=True,
        ),
        abm_transforms.FilterNanOrInfDataTransform(),
        bucket_transforms.BucketBatchingTransform(
            batch_size=batch_size, num_buckets=len(hw_bucket_list), drop_last=False
        ),
    )


@functools.partial(auto_config.auto_config, experimental_allow_control_flow=True)
def build_apc_dataloader_fn(
    batch_size: int = 16,
    num_frames: int = 8,
    num_pixel_threshold: int = 384 * 384,
    frame_skip: int = 1,
    is_training: bool = True,
    version: int | str | None = None,
):
    """Returns partial function to build a APC dataloader."""
    # Plus 1 because we sampling 1 frame within each keyframe interval. So
    # num_frames=num_keyframes_intervals=num_keyframes+1.
    # lance_filter = f"num_posing_keyframe >= {num_frames+1}"
    lance_filter = f"num_frames >= {num_frames * frame_skip}"
    return functools.partial(
        data_config_utils.build_graph_dataloader,
        graph=build_apc_data_graph(
            input_graph=build_apc_input_graph(
                is_training=is_training,
                lance_filter=lance_filter,
                version=version,
            ),
            batch_size=batch_size,
            num_frames=num_frames,
            num_pixel_threshold=num_pixel_threshold,
            hw_bucket_list=((15, 26), (26, 15), (3, 4), (4, 3), (1, 1)),
            image_size_multiple_of=16,
            frame_skip=frame_skip,
        ),
    )
