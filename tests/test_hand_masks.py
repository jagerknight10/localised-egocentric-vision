import numpy as np

from egovision.data.hand_masks import crop_frame, parse_hand_mask


def test_parse_mask_and_crop(tmp_path) -> None:
    path = tmp_path / "s1_cheese_0000000020.xml"
    path.write_text(
        "<annotation><imagesize><nrows>10</nrows><ncols>12</ncols></imagesize>"
        "<object><polygon><pt><x>3</x><y>4</y></pt><pt><x>6</x><y>7</y></pt>"
        "</polygon></object></annotation>"
    )
    mask = parse_hand_mask(path, margin=0)
    assert mask.frame_index == 20
    assert mask.box == (3, 4, 7, 8)
    image = np.zeros((10, 12, 3), dtype=np.uint8)
    assert crop_frame(image, mask).shape == (4, 4, 3)
