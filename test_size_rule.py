"""Unit tests for the pure bottle-size classification rule."""

import unittest

from vision import classify_bottle_size


class ClassifyBottleSizeTests(unittest.TestCase):
    def test_bottle_within_both_limits_is_normal(self) -> None:
        self.assertEqual(
            classify_bottle_size((25, 20, 75, 180), 200, 200, 0.25, 0.8),
            "NORMAL",
        )

    def test_area_above_calibrated_limit_is_oversized(self) -> None:
        self.assertEqual(
            classify_bottle_size((0, 0, 100, 100), 200, 200, 0.20, 0.8),
            "OVERSIZED_BOTTLE",
        )

    def test_height_above_calibrated_limit_is_oversized(self) -> None:
        self.assertEqual(
            classify_bottle_size((10, 0, 30, 190), 200, 200, 0.50, 0.8),
            "OVERSIZED_BOTTLE",
        )

    def test_box_touching_limit_is_not_oversized(self) -> None:
        self.assertEqual(
            classify_bottle_size((0, 0, 50, 100), 200, 200, 0.125, 0.5),
            "NORMAL",
        )

    def test_bottle_inside_calibrated_one_liter_range_is_normal(self) -> None:
        self.assertEqual(
            classify_bottle_size(
                (40, 20, 160, 180), 200, 200,
                max_area_ratio=0.50, max_height_ratio=0.90,
                min_area_ratio=0.40, min_height_ratio=0.70,
            ),
            "NORMAL",
        )

    def test_bottle_below_both_calibrated_limits_is_undersized(self) -> None:
        self.assertEqual(
            classify_bottle_size(
                (75, 75, 125, 125), 200, 200,
                max_area_ratio=0.50, max_height_ratio=0.90,
                min_area_ratio=0.10, min_height_ratio=0.30,
            ),
            "UNDERSIZED_BOTTLE",
        )

    def test_only_one_dimension_below_range_is_not_undersized(self) -> None:
        self.assertEqual(
            classify_bottle_size(
                (80, 10, 120, 190), 200, 200,
                max_area_ratio=0.50, max_height_ratio=0.95,
                min_area_ratio=0.15, min_height_ratio=0.50,
            ),
            "NORMAL",
        )

    def test_inverted_size_range_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            classify_bottle_size(
                (0, 0, 10, 10), 200, 200,
                max_area_ratio=0.1, max_height_ratio=0.1,
                min_area_ratio=0.2, min_height_ratio=0.2,
            )

    def test_invalid_box_length_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            classify_bottle_size((1, 2, 3), 200, 200, 0.2, 0.5)

    def test_invalid_frame_size_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            classify_bottle_size((0, 0, 10, 10), 0, 200, 0.2, 0.5)


if __name__ == "__main__":
    unittest.main()
