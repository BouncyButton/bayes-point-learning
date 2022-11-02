import pytest

from sklearn.utils.estimator_checks import check_estimator

from bpllib import TemplateEstimator
from bpllib import TemplateClassifier
from bpllib import TemplateTransformer
from bpllib._bpl import BplClassifier


@pytest.mark.parametrize(
    "estimator",
    [BplClassifier()]
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
