import numpy as np
from PIL import Image, ImageDraw
import pytest

from tasks.business_finance.saas_onepager_brand_refresh_instance_1.scripts import (
    score_onepager_v2 as scorer,
)


class PixelComparisonJudge:
    def __init__(self, regions):
        self.regions = regions

    def judge(self, submission, reference, question):
        assert submission.size == reference.size
        if question == scorer.REGION_RUBRIC["headline_subheadline"]:
            left, top, width, height = self.regions["headline_subheadline"]
            bounds = (left, top, left + width, top + height)
            submission = submission.crop(bounds)
            reference = reference.crop(bounds)
        error = np.abs(
            np.asarray(submission, dtype=float) - np.asarray(reference, dtype=float)
        ).mean()
        return bool(error < 3)


@pytest.fixture
def region_images(tmp_path):
    reference_size = (1530, 2721)
    image = Image.new("RGB", reference_size, "white")
    drawing = ImageDraw.Draw(image)
    regions = {}
    for index, label in enumerate(scorer.REGION_RUBRIC):
        left, top, width, height = 90, 100 + 245 * index, 1250, 180
        regions[label] = (left, top, width, height)
        drawing.rectangle(
            (left, top, left + width, top + height),
            fill=(25 + index * 17, 180 - index * 12, 40 + index * 9),
        )
        drawing.rectangle((left + 100, top + 30, left + 600, top + 120), fill="black")
    reference = tmp_path / "reference.png"
    submission = tmp_path / "submission.png"
    image.save(reference)
    return image, submission, reference, regions


@pytest.mark.parametrize("size", [(1071, 1905), (1152, 2048), (1530, 2721), (2295, 4082)])
def test_equivalent_exports_compare_the_same_regions_at_allowed_resolutions(region_images, size):
    image, submission, reference, regions = region_images
    image.resize(size, Image.Resampling.LANCZOS).save(submission)
    score, detail = scorer.vlm_region_score(
        PixelComparisonJudge(regions), submission, reference, regions
    )
    assert score == 1
    assert all(detail.values())


def test_missing_footer_still_loses_visual_credit_after_rescaling(region_images):
    image, submission, reference, regions = region_images
    left, top, width, height = regions["footer_band"]
    ImageDraw.Draw(image).rectangle((left, top, left + width, top + height), fill="white")
    image.resize((1152, 2048), Image.Resampling.LANCZOS).save(submission)
    score, detail = scorer.vlm_region_score(
        PixelComparisonJudge(regions), submission, reference, regions
    )
    assert score == 0.9
    assert detail["footer_band"] is False


@pytest.mark.parametrize("size", [(1071, 1905), (1152, 2048), (2295, 4082)])
def test_rescaled_input_copy_still_fails_anti_copy_gate(region_images, size):
    image, submission, reference, regions = region_images
    image.resize(size, Image.Resampling.LANCZOS).save(submission)
    violations = scorer.anti_copy_violations(submission, reference, regions, image.size)
    assert set(violations) == set(scorer.ANTI_COPY_REGIONS)


def test_changed_regions_are_not_mistaken_for_rescaled_input_copies(region_images):
    image, submission, reference, regions = region_images
    image = Image.fromarray(255 - np.asarray(image))
    image.resize((1152, 2048), Image.Resampling.LANCZOS).save(submission)
    assert scorer.anti_copy_violations(submission, reference, regions, image.size) == {}


def test_anti_copy_regions_use_reference_coordinates_when_input_size_also_differs(region_images):
    image, submission, reference, regions = region_images
    image.resize((1152, 2048), Image.Resampling.LANCZOS).save(submission)
    image.resize((1071, 1905), Image.Resampling.LANCZOS).save(reference)
    violations = scorer.anti_copy_violations(submission, reference, regions, image.size)
    assert set(violations) == set(scorer.ANTI_COPY_REGIONS)
