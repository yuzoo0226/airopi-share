import numpy as np

import openpi.training.weight_loaders as weight_loaders


def test_merge_params_copies_shared_image_encoder_weights_into_split_paths() -> None:
    loaded_params = {
        "PaliGemma": {
            "img": {
                "embedding": {
                    "kernel": np.ones((2,), dtype=np.float32),
                }
            }
        }
    }
    ref_params = {
        "PaliGemma": {
            "img": {
                "embedding": {
                    "kernel": np.zeros((2,), dtype=np.float32),
                }
            },
            "img_left_wrist_0_rgb": {
                "embedding": {
                    "kernel": np.zeros((2,), dtype=np.float32),
                }
            },
            "img_right_wrist_0_rgb": {
                "embedding": {
                    "kernel": np.zeros((2,), dtype=np.float32),
                }
            },
        }
    }

    merged = weight_loaders._merge_params(loaded_params, ref_params, missing_regex=".*")  # noqa: SLF001

    np.testing.assert_array_equal(merged["PaliGemma"]["img"]["embedding"]["kernel"], np.ones((2,), dtype=np.float32))
    np.testing.assert_array_equal(
        merged["PaliGemma"]["img_left_wrist_0_rgb"]["embedding"]["kernel"],
        np.ones((2,), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        merged["PaliGemma"]["img_right_wrist_0_rgb"]["embedding"]["kernel"],
        np.ones((2,), dtype=np.float32),
    )
