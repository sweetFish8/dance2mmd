import numpy as np
from dance2mmd import quat
from dance2mmd.synthetic import make
from dance2mmd.motion import Motion
from dance2mmd.retarget import retarget, motion_to_vmd
from dance2mmd.vmd import read_vmd_header
from dance2mmd.postprocess import smooth


def test_quat_two_vectors():
    q = quat.from_two_vectors([1, 0, 0], [0, 1, 0])
    # rotating x by q should land on y
    v = np.array([1.0, 0, 0])
    qv = quat.mul(quat.mul(q, [v[0], v[1], v[2], 0]), quat.conj(q))[:3]
    assert np.allclose(qv, [0, 1, 0], atol=1e-6)


def test_json_roundtrip(tmp_path):
    m = make(30)
    p = tmp_path / "m.json"
    m.to_json(str(p))
    m2 = Motion.from_json(str(p))
    assert m2.num_frames == 30
    assert np.allclose(m.positions, m2.positions, atol=1e-4)


def test_retarget_counts():
    m = make(10)
    frames = retarget(m)
    # 12 driven bones + センター + 左足ＩＫ + 右足ＩＫ per frame
    assert len(frames) == 10 * 15
    for bf in frames:
        assert abs(np.linalg.norm(bf.quat) - 1.0) < 1e-3 or bf.name == "センター"


def test_vmd_header(tmp_path):
    m = make(20)
    out = tmp_path / "d.vmd"
    n = motion_to_vmd(m, str(out), model_name="test")
    sig, model, count = read_vmd_header(str(out))
    assert sig == b"Vocaloid Motion Data 0002"
    assert model == "test"
    assert count == n == 20 * 15


def test_smooth_shape():
    m = smooth(make(40), window=5)
    assert m.num_frames == 40
