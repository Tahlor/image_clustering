"""One-time patch helper; removed by the workflow that invokes it."""

from pathlib import Path


config_path = Path("src/image_clustering/clustering/config.py")
config = config_path.read_text(encoding="utf-8")
config = config.replace(
    "    ecc_coarse_dimension: int = 192\n",
    "    ecc_coarse_dimension: int = 192\n"
    "    ecc_working_dimension: int = 384\n",
)
config = config.replace(
    '        if self.ecc_coarse_dimension < 64:\n'
    '            raise ValueError("ecc_coarse_dimension must be at least 64")\n',
    '        if self.ecc_coarse_dimension < 64:\n'
    '            raise ValueError("ecc_coarse_dimension must be at least 64")\n'
    '        if self.ecc_working_dimension < 128:\n'
    '            raise ValueError("ecc_working_dimension must be at least 128")\n',
)
config_path.write_text(config, encoding="utf-8")

path = Path("src/image_clustering/clustering/registration.py")
text = path.read_text(encoding="utf-8")
text = text.replace("import math\n", "import math\nimport threading\n")
text = text.replace(
    '_AFFINE_MODELS = {"affine", "ecc_euclidean"}\n',
    '_AFFINE_MODELS = {"affine", "ecc_euclidean"}\n'
    "_ECC_LOCK = threading.Lock()\n",
)
old = '''    previous_canvas, previous_offset = _center_on_canvas(
        previous.gray,
        canvas_shape,
    )
    current_canvas, current_offset = _center_on_canvas(
        current.gray,
        canvas_shape,
    )
    template = _ecc_preprocess(previous_canvas)
    input_image = _ecc_preprocess(current_canvas)

    current_to_canvas = np.array(
        [
            [1.0, 0.0, current_offset[0]],
            [0.0, 1.0, current_offset[1]],
            [0.0, 0.0, 1.0],
        ]
    )
    previous_to_canvas = np.array(
        [
            [1.0, 0.0, previous_offset[0]],
            [0.0, 1.0, previous_offset[1]],
            [0.0, 0.0, 1.0],
        ]
    )
'''
new = '''    previous_canvas, previous_offset = _center_on_canvas(
        previous.gray,
        canvas_shape,
    )
    current_canvas, current_offset = _center_on_canvas(
        current.gray,
        canvas_shape,
    )

    # Full-resolution ECC can spend minutes on repetitive or unrelated forms.
    # Estimate motion on a bounded canvas, then map it back for full-resolution
    # content scoring.
    scale = min(1.0, config.ecc_working_dimension / max(canvas_shape))
    if scale < 1.0:
        working_size = (
            max(64, round(canvas_shape[1] * scale)),
            max(64, round(canvas_shape[0] * scale)),
        )
        previous_working = cv2.resize(
            previous_canvas,
            working_size,
            interpolation=cv2.INTER_AREA,
        )
        current_working = cv2.resize(
            current_canvas,
            working_size,
            interpolation=cv2.INTER_AREA,
        )
    else:
        previous_working = previous_canvas
        current_working = current_canvas
    scale_x = previous_working.shape[1] / canvas_shape[1]
    scale_y = previous_working.shape[0] / canvas_shape[0]
    canvas_to_working = np.diag([scale_x, scale_y, 1.0])

    template = _ecc_preprocess(previous_working)
    input_image = _ecc_preprocess(current_working)

    current_to_canvas = np.array(
        [
            [1.0, 0.0, current_offset[0]],
            [0.0, 1.0, current_offset[1]],
            [0.0, 0.0, 1.0],
        ]
    )
    previous_to_canvas = np.array(
        [
            [1.0, 0.0, previous_offset[0]],
            [0.0, 1.0, previous_offset[1]],
            [0.0, 0.0, 1.0],
        ]
    )
    current_to_working = canvas_to_working @ current_to_canvas
    previous_to_working = canvas_to_working @ previous_to_canvas
'''
if old not in text:
    raise RuntimeError("registration canvas block not found")
text = text.replace(old, new)
text = text.replace(
    '''        current_to_previous_canvas = (
            previous_to_canvas @ initial @ np.linalg.inv(current_to_canvas)
        )
        warp = cv2.invertAffineTransform(
            current_to_previous_canvas[:2].astype(np.float32)
        )
''',
    '''        current_to_previous_working = (
            previous_to_working @ initial @ np.linalg.inv(current_to_working)
        )
        warp = cv2.invertAffineTransform(
            current_to_previous_working[:2].astype(np.float32)
        )
''',
)
text = text.replace(
    '''        window = cv2.createHanningWindow(
            (canvas_shape[1], canvas_shape[0]),
            cv2.CV_32F,
        )
''',
    '''        window = cv2.createHanningWindow(
            (template.shape[1], template.shape[0]),
            cv2.CV_32F,
        )
''',
    1,
)
text = text.replace(
    '''    try:
        correlation, template_to_current = cv2.findTransformECC(
            template,
            input_image,
            warp,
            cv2.MOTION_EUCLIDEAN,
            criteria,
            None,
            config.ecc_gaussian_filter_size,
        )
    except cv2.error:
''',
    '''    try:
        # OpenCV ECC is not reliably efficient when several calls run at once.
        # Keep the ordinary pair pipeline parallel and serialize this rare step.
        with _ECC_LOCK:
            correlation, template_to_current = cv2.findTransformECC(
                template,
                input_image,
                warp,
                cv2.MOTION_EUCLIDEAN,
                criteria,
                None,
                config.ecc_gaussian_filter_size,
            )
    except cv2.error:
''',
)
text = text.replace(
    '''    current_to_previous_canvas = cv2.invertAffineTransform(template_to_current)
    canvas_matrix = np.vstack(
        [current_to_previous_canvas, np.array([0.0, 0.0, 1.0])]
    )
    source_matrix = (
        np.linalg.inv(previous_to_canvas) @ canvas_matrix @ current_to_canvas
    )
''',
    '''    current_to_previous_working = cv2.invertAffineTransform(
        template_to_current
    )
    working_matrix = np.vstack(
        [current_to_previous_working, np.array([0.0, 0.0, 1.0])]
    )
    source_matrix = (
        np.linalg.inv(previous_to_working)
        @ working_matrix
        @ current_to_working
    )
''',
)
path.write_text(text, encoding="utf-8")

test_path = Path("tests/test_recall_first_occlusion.py")
tests = test_path.read_text(encoding="utf-8")
if "def test_ecc_uses_bounded_working_canvas(" not in tests:
    tests += '''


def test_ecc_uses_bounded_working_canvas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from image_clustering.clustering import registration as registration_module

    captured: dict[str, tuple[int, int]] = {}

    def fake_ecc(
        template: np.ndarray,
        input_image: np.ndarray,
        warp: np.ndarray,
        motion_type: int,
        criteria: tuple[int, int, float],
        input_mask: np.ndarray | None,
        gaussian_filter_size: int,
    ) -> tuple[float, np.ndarray]:
        captured["template"] = template.shape
        captured["input"] = input_image.shape
        return 0.99, warp

    monkeypatch.setattr(cv2, "findTransformECC", fake_ecc)
    image = cv2.resize(_document(), (900, 900))
    config = ClusterConfig(
        ecc_working_dimension=256,
        min_inliers=1000,
        max_features=2500,
    )
    result = registration_module._small_motion_ecc_registration(
        previous=_features(image, "previous.jpg"),
        current=_features(image, "current.jpg"),
        config=config,
    )

    assert result.accepted
    assert max(captured["template"]) <= 256
    assert captured["template"] == captured["input"]
'''
test_path.write_text(tests, encoding="utf-8")
