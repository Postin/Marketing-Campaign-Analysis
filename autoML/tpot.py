import time

from imblearn.pipeline import make_pipeline
from tpot import TPOTClassifier
from tpot.builtins import StackingEstimator

from constants import seed


def tpot_pipeline_optimizer(X_train, y_train, X_valid, y_valid, generations=5, population_size=50, cv=5, scoring='f1'):
    pipeline_optimizer = TPOTClassifier(generations=generations, population_size=population_size, cv=cv,
                                        random_state=seed, verbosity=2, scoring=scoring)
    pipeline_optimizer.fit(X_train, y_train)
    print(pipeline_optimizer.score(X_valid, y_valid))
    # Save pipeline
    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y_%H-%M-%S', t)
    pipeline_optimizer.export(data_file_path='pipelines', output_file_name='../autoML/pipelines/tpot_{}.py'.format(timestamp))
    return pipeline_optimizer