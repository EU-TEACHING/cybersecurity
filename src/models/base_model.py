# -*- coding: utf-8 -*-
"""Abstract base model"""

from abc import ABC, abstractmethod
from src.utils.config import Config


class BaseModel(ABC):
    """Abstract Model class that is inherited to all models"""
    def __init__(self, cfg):
        self.config = Config.from_json(cfg)

    @abstractmethod
    def _load_data(self):
        pass

    @abstractmethod
    def _build(self, hyperparams):
        pass

    @abstractmethod
    def _train(self, hyperparams):
        pass

    @abstractmethod
    def _eval(self):
        pass
