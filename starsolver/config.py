from dataclasses import dataclass, field


@dataclass
class DrawConfig:
    star_radius: int          = 25
    constellation_color: tuple = (255, 180, 0)
    constellation_thickness: int = 4
    match_color: tuple        = (255, 180, 0)
    unknown_color: tuple      = (200, 50, 50)
    circle_thickness: int     = 2
    detection_color: tuple    = (0, 255, 0)
    detection_thickness: int  = 5
    max_draw: int             = 2000


@dataclass
class DetectConfig:
    ratio_threshold: float = 20.0


@dataclass
class SolveConfig:
    timeout_ms: int = 10000


@dataclass
class Config:
    draw:   DrawConfig   = field(default_factory=DrawConfig)
    detect: DetectConfig = field(default_factory=DetectConfig)
    solve:  SolveConfig  = field(default_factory=SolveConfig)
