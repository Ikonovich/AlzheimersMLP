import numpy as np


# This class contains methods to assist with calculating metrics


# Calculate TP, FP, TN, and FN for each class, and
# uses those to calculate standard metrics.
# This returns four baseMetrics dictionaries
# that can be passed into get_recall, etc
# to get more complex metrics.
def get_class_base_metrics(results, expected, labels):

    # Generate the list of TP/FP/TN/FN dictionaries
    classMetrics = list()
    for i in range(len(results[0])):
        classMetrics.append({"Label": labels[i], "TP": 0, "FP": 0, "TN": 0, "FN": 0})

    # Now calculate the metrics
    for actual, predicted in zip(results, expected):
        actualIndex = np.argmax(actual)
        predictedIndex = np.argmax(predicted)

        # Set metrics for a correct result
        if actualIndex == predictedIndex:
            classMetrics[predictedIndex]["TP"] += 1
            for i in range(len(classMetrics)):
                if i != predictedIndex:
                    classMetrics[i]["TN"] += 1

        # Set metrics for an incorrect result
        else:
            classMetrics[predictedIndex]["FN"] += 1
            classMetrics[actualIndex]["FP"] += 1
            for i in range(len(classMetrics)):
                if i != predictedIndex and i != actualIndex:
                    classMetrics[actualIndex]["TN"] += 1

    return classMetrics

# Uses the output from get_class_base_metrics
# to calculate the base metrics (TP, FP, FN, and TN)
# for the entire network.
def get_network_base_metrics(base_metric_list):
    tp, tn, fp, fn = (0, 0, 0, 0)

    for metrics in base_metric_list:
        tp += metrics["TP"]
        fp += metrics["FP"]
        tn += metrics["TN"]
        fn += metrics["FN"]

    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}



def get_recall(base_metric_dict):
    tp = base_metric_dict["TP"]
    fn = base_metric_dict["FN"]
    if tp == 0:
        return 0
    return tp / (tp + fn)


def get_precision(base_metric_dict):
    tp = base_metric_dict["TP"]
    fp = base_metric_dict["FP"]
    if tp == 0:
        return 0
    return tp / (tp + fp)


def get_f1score(base_metric_dict):
    tp = base_metric_dict["TP"]
    fp = base_metric_dict["FP"]
    fn = base_metric_dict["FN"]
    # alternative f1-score formula
    if tp == 0:
        return 0
    f1 = (2 * tp) / ((2 * tp) + fp + fn)
    return f1


def get_specificity(base_metric_dict):
    fp = base_metric_dict["FP"]
    tn = base_metric_dict["TN"]
    if tn == 0:
        return 0
    return tn / (tn + fp)


def get_accuracy(base_metric_dict):
    tp = base_metric_dict["TP"]
    fp = base_metric_dict["FP"]
    tn = base_metric_dict["TN"]
    fn = base_metric_dict["FN"]
    if tp + tn == 0:
        return 0
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    return accuracy

def get_total_accuracy(expected, results):
    correct = 0
    wrong = 0
    for actual_val_array,expected_val_array in zip(results,expected):
        actual_val = np.argmax(actual_val_array)
        expected_val = np.argmax(expected_val_array)
        if actual_val == expected_val:
            correct += 1
        else:
            wrong += 1
    return correct / (correct + wrong)

def get_macro_averaged_f1(base_metric_list):
    totalF1 = 0
    for metrics in base_metric_list:
        totalF1 += get_f1score(metrics)

    if totalF1 == 0:
        return 0
    return totalF1 / len(base_metric_list)

# Given a dictionary of results, expected values, and labels,
# creates a dictionary containing all available metrics
# on a per-class basis.
# Labels should be a list with the same size and order
# as one entry in results or expected
def get_metrics(results, expected, labels):
    class_base_metrics = get_class_base_metrics(results, expected, labels)

    results = np.asarray(results)
    expected = np.asarray(expected)
    returnList = []

    for i in range(len(class_base_metrics)):
        metrics = class_base_metrics[i]

        returnList.append({
            "Label: ": labels[i],
            "Accuracy": get_accuracy(metrics),
            "Precision": get_precision(metrics),
            "Recall": get_recall(metrics),
            "F1 Score": get_f1score(metrics),
            "Specificity": get_specificity(metrics)})

    network_base_metrics = get_network_base_metrics(class_base_metrics)

    returnList.insert(0, {
        "Overall Accuracy": get_total_accuracy(expected, results),
        "Precision": get_precision(network_base_metrics),
        "Recall": get_recall(network_base_metrics),
        "Micro-Averaged F1 Score": get_f1score(network_base_metrics),
        "Macro-Averaged F1 Score": get_macro_averaged_f1(class_base_metrics),
        "Specificity:": get_specificity(network_base_metrics)
    })

    return returnList
