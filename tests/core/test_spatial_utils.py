from lm_deluge.util.spatial import (
    Box2D,
    CoordinateSpace,
    Point,
    SpatialResult,
    denormalize_point,
    denormalize_point_1k,
    normalize_point,
    normalize_point_1k,
    parse_gemini_boxes,
    parse_gemini_points,
    parse_moondream,
    parse_perceptron,
)


def test_point_and_box_coordinate_conversions():
    point = Point(0.5, 0.25)
    pixel_point = point.to_pixels(200, 100)
    assert pixel_point == Point(100, 25, space=CoordinateSpace.PIXELS)

    box = Box2D(0.1, 0.2, 0.3, 0.4)
    pixel_box = box.to_pixels(1000, 500)
    assert pixel_box == Box2D(100, 100, 300, 200, space=CoordinateSpace.PIXELS)
    assert pixel_box.center() == Point(200, 150, space=CoordinateSpace.PIXELS)

    box_1k = box.to_1000()
    assert box_1k == Box2D(
        100,
        200,
        300,
        400,
        space=CoordinateSpace.NORMALIZED_1000,
    )
    assert box_1k.to_normalized() == box


def test_legacy_normalization_helpers_still_work():
    assert normalize_point((50, 25), 100, 100) == Point(
        0.5, 0.25, space=CoordinateSpace.NORMALIZED
    )
    assert denormalize_point((0.5, 0.25), 100, 100) == Point(
        50, 25, space=CoordinateSpace.PIXELS
    )
    assert normalize_point_1k((50, 25), 100, 100) == Point(
        500, 250, space=CoordinateSpace.NORMALIZED_1000
    )
    assert denormalize_point_1k((500, 250), 100, 100) == Point(
        50.0, 25.0, space=CoordinateSpace.PIXELS
    )


def test_parse_moondream_detect_point_and_segment_shapes():
    detect = parse_moondream(
        {"objects": [{"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4}]}
    )
    assert detect.boxes == [Box2D(0.1, 0.2, 0.3, 0.4, raw=detect.raw["objects"][0])]

    point = parse_moondream({"points": [{"x": 0.5, "y": 0.75}]})
    assert point.points == [Point(0.5, 0.75, raw=point.raw["points"][0])]

    segment = parse_moondream(
        {
            "path": "M 0.1 0.2 L 0.3 0.4 Z",
            "bbox": {"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4},
        }
    )
    assert len(segment.masks) == 1
    assert segment.masks[0].bbox == Box2D(
        0.1,
        0.2,
        0.3,
        0.4,
        raw=segment.raw["bbox"],
    )


def test_parse_perceptron_tags():
    text = (
        "Here are results. "
        '<point mention="ball">(871,179)</point> '
        '<point_box mention="kitten">(172,127) (586,967)</point_box> '
        '<polygon mention="racket">(1,2) (3,4) (5,6)</polygon>'
    )
    result = parse_perceptron(text)

    assert result.source == "perceptron"
    assert result.points[0] == Point(
        871,
        179,
        label="ball",
        description="ball",
        source="perceptron",
        raw='<point mention="ball">(871,179)</point>',
        extra={"mention": "ball"},
        space=CoordinateSpace.NORMALIZED_1000,
    )
    assert result.boxes[0] == Box2D(
        172,
        127,
        586,
        967,
        label="kitten",
        description="kitten",
        source="perceptron",
        raw='<point_box mention="kitten">(172,127) (586,967)</point_box>',
        extra={"mention": "kitten"},
        space=CoordinateSpace.NORMALIZED_1000,
    )
    assert len(result.polygons) == 1
    assert [tuple(p) for p in result.polygons[0].points] == [
        (1, 2),
        (3, 4),
        (5, 6),
    ]
    assert result.text == "Here are results."


def test_parse_perceptron_collection_tags():
    result = parse_perceptron(
        '<collection mention="balls">'
        '<point mention="ball one">(100,200)</point>'
        '<point mention="ball two">(300,400)</point>'
        "</collection>"
    )

    assert [point.label for point in result.points] == ["ball one", "ball two"]
    assert [tuple(point) for point in result.points] == [(100, 200), (300, 400)]


def test_parse_gemini_boxes_and_points():
    boxes = parse_gemini_boxes([{"label": "kitten", "box_2d": [127, 172, 967, 586]}])
    assert boxes.boxes == [
        Box2D(172, 127, 586, 967, space=CoordinateSpace.NORMALIZED_1000)
    ]

    points = parse_gemini_points(
        [{"label": "ball", "point": [179, 871], "description": "tennis ball"}]
    )
    assert points.points == [
        Point(
            871,
            179,
            label="ball",
            description="tennis ball",
            raw=points.raw[0],
            source="gemini",
            space=CoordinateSpace.NORMALIZED_1000,
        )
    ]


def test_spatial_result_conversion():
    result = SpatialResult(
        points=[Point(500, 250, space=CoordinateSpace.NORMALIZED_1000)],
        boxes=[Box2D(100, 200, 300, 400, space=CoordinateSpace.NORMALIZED_1000)],
    )

    pixels = result.to_pixels(100, 200)
    assert pixels.points == [Point(50, 50, space=CoordinateSpace.PIXELS)]
    assert pixels.boxes == [Box2D(10, 40, 30, 80, space=CoordinateSpace.PIXELS)]


if __name__ == "__main__":
    test_point_and_box_coordinate_conversions()
    test_legacy_normalization_helpers_still_work()
    test_parse_moondream_detect_point_and_segment_shapes()
    test_parse_perceptron_tags()
    test_parse_perceptron_collection_tags()
    test_parse_gemini_boxes_and_points()
    test_spatial_result_conversion()
    print("All spatial utility tests passed!")
