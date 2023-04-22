import numpy as np


def evaluate_measures(sample):
    """Calculate measure of split quality (each node separately).

    Please use natural logarithm (e.g. np.log) to evaluate value of entropy measure.

    Parameters
    ----------
    sample : a list of integers. The size of the sample equals to the number of objects in the current node. The integer
    values are equal to the class labels of the objects in the node.

    Returns
    -------
    measures - a dictionary which contains three values of the split quality.
    Example of output:

    {
        'gini': 0.1,
        'entropy': 1.0,
        'error': 0.6
    }

    """
    measures = {'gini': float(len(sample)), 'entropy': float(sum(sample)), 'error': float(max(sample))}

    vals = np.unique(sample)
    probs = {u: sample.count(u) / len(sample) for u in vals}
    probs_lst = np.array(list(probs.values()))

    measures['error'] = 1 - np.max(probs_lst)

    measures['entropy'] = -np.sum(probs_lst*np.log(probs_lst))

    measures['gini'] = 1 - np.sum(np.square(probs_lst))

    return measures
