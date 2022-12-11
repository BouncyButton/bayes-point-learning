from ._template import TemplateEstimator
from ._template import TemplateClassifier
from ._template import TemplateTransformer
from ._bpl import BplClassifier
# import pyximport; pyximport.install()
# from ._bpl_v2 import BplClassifierV2
from ._bpl_v5 import BplClassifierV5
from ._bpl_split import BplClassifierSplit
from ._bpl_optimization import BplClassifierOptimization
from ._aq import AqClassifier
from ._definitions import ROOT_DIR
from ._dataset import get_dataset
from ._version import __version__

__all__ = ['TemplateEstimator', 'TemplateClassifier', 'TemplateTransformer',
           'BplClassifier',
           'AqClassifier',
           'get_dataset',
           # 'BplClassifierV2',
           'BplClassifierV5',
           'BplClassifierOptimization',
           'BplClassifierSplit',
           'ROOT_DIR',
           '__version__']
