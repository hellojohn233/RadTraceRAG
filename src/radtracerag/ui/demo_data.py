"""演示数据。

真实链路还没接上，界面先要能跑起来看效果。这里用一组贴近 SLAKE 风格的
假数据把三组对照跑通，影像用程序生成的合成腹部切片。

接上真实链路后，把 load_demo() 换成实际的 pipeline 调用即可，界面不用动。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from radtracerag.contracts import Answer, Evidence, ImageObservation, Metric, RetrievalStep
from radtracerag.fixtures import MODES

ASSETS = Path(__file__).resolve().parents[3] / "assets"


@dataclass(frozen=True)
class DemoCase:
    image: Path
    question: str
    answer: Answer
    observation: ImageObservation
    evidence: list[Evidence]
    trace: list[RetrievalStep]
    metrics: list[Metric]
    desc: str


def make_demo_image(path: Path | None = None) -> Path:
    """生成一张合成的腹部断层影像。

    不是真片子，只是有 CT 观感的灰阶结构：黑背景、分层的体壁轮廓、
    右上腹一片均匀高密度的肝、两侧肾影、中后方的椎体亮点和椎前主动脉。
    用软边椭圆遮罩而不是高斯叠加，才能得到 CT 那种边界清晰的结构。
    """
    path = path or (ASSETS / "demo" / "abdominal-ct.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path

    h = w = 512
    yy, xx = np.mgrid[0:h, 0:w]

    def soft_ellipse(cy_: float, cx_: float, ry: float, rx: float, edge: float = 0.10) -> np.ndarray:
        d = ((yy - h * cy_) / (h * ry)) ** 2 + ((xx - w * cx_) / (w * rx)) ** 2
        return np.clip((1.0 - d) / edge, 0, 1).astype(np.float32)

    def layer(mask: np.ndarray, value: float) -> np.ndarray:
        return mask * value

    img = np.zeros((h, w), dtype=np.float32)

    # 腹内软组织底
    body = soft_ellipse(0.50, 0.50, 0.45, 0.41, 0.30)
    img = np.maximum(img, layer(body, 0.20))
    # 皮下脂肪层：身体轮廓再往外一圈
    fat = soft_ellipse(0.50, 0.50, 0.485, 0.445, 0.20) - body
    img = np.maximum(img, layer(np.clip(fat, 0, 1), 0.34))
    # 肝：右上腹大片均匀高密度
    img = np.maximum(img, layer(soft_ellipse(0.40, 0.60, 0.285, 0.235, 0.42), 0.30))
    # 脾：左上腹
    img = np.maximum(img, layer(soft_ellipse(0.44, 0.36, 0.195, 0.155, 0.42), 0.26))
    # 两侧肾影
    for kx in (0.255, 0.745):
        img = np.maximum(img, layer(soft_ellipse(0.505, kx, 0.115, 0.078, 0.44), 0.40))
    # 椎体：骨，最亮
    img = np.maximum(img, layer(soft_ellipse(0.600, 0.50, 0.062, 0.055, 0.34), 0.70))
    # 椎体棘突
    img = np.maximum(img, layer(soft_ellipse(0.655, 0.50, 0.045, 0.016, 0.40), 0.55))
    # 腹主动脉：椎体前方的亮点
    img = np.maximum(img, layer(soft_ellipse(0.508, 0.50, 0.034, 0.029, 0.45), 0.50))
    # 胃肠气腔：低密度黑斑
    gas = soft_ellipse(0.455, 0.425, 0.115, 0.085, 0.50)
    img = np.clip(img - gas * 0.17, 0, 1)

    rng = np.random.default_rng(7)
    img += rng.normal(0, 0.016, img.shape).astype(np.float32)

    img = np.clip(img, 0, 1)
    # CT 重建影像不是锐利的，轻度模糊
    blurred = Image.fromarray((img * 255).astype(np.uint8), mode="L").filter(
        ImageFilter.GaussianBlur(radius=1.6)
    )
    img = np.asarray(blurred, dtype=np.float32) / 255.0
    # 窗宽窗位：提中间调，让软组织层次出来
    img = (img - 0.30) * 1.22 + 0.26
    img = np.clip(img, 0, 1)

    Image.fromarray((img * 255).astype(np.uint8), mode="L").save(path)
    return path


def load_demo(mode: str) -> DemoCase:
    spec = MODES.get(mode, MODES["C · 图增强 RAG"])
    image = make_demo_image()
    return DemoCase(
        image=image,
        question="图中主要器官的作用是什么？",
        answer=spec["answer"],
        observation=spec["observation"],
        evidence=spec["evidence"],
        trace=spec["trace"],
        metrics=spec["metrics"],
        desc=spec["desc"],
    )


__all__ = ["DemoCase", "MODES", "load_demo", "make_demo_image"]
