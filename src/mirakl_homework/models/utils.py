import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def get_ancestors(graph, node):
    # Initialize an empty list to collect ancestors
    ancestors = []

    # Use a queue for breadth-first traversal
    queue = [node]  # Start from the given node
    visited = set()  # To avoid cycles

    while queue:
        current = queue.pop(0)  # Dequeue the first element

        # Get the parents of the current node
        parents = graph.predecessors(current)

        for parent in parents:
            if parent not in visited:  # Check if already visited
                visited.add(parent)  # Mark the parent as visited
                ancestors.append(parent)  # Collect the parent
                queue.append(parent)  # Add parent to the queue for further exploration

    # Return the ancestors in the order from root to the direct parent
    # return [np.int64(0) if ancestor == new_root_id else uniques[ancestor] for ancestor in ancestors]
    return ancestors


def result_analysis(y_true, y_pred):
    global_accuracy = accuracy_score(y_true, y_pred)
    global_precision = precision_score(
        y_true,
        y_pred,
        average='macro',
        zero_division=0
    )
    weighted_global_precision = precision_score(
        y_true,
        y_pred,
        average='weighted',
        zero_division=0
    )

    global_recall = recall_score(
        y_true,
        y_pred,
        average='macro',
        zero_division=0
    )
    weighted_global_recall = recall_score(
        y_true,
        y_pred,
        average='weighted',
        zero_division=0
    )

    print(f"{global_accuracy:.2%} of global accuracy")
    print(f"{global_precision:.2%} of global precision")
    print(f"{weighted_global_precision:.2%} of weighted global precision")
    print(f"{global_recall:.2%} of global recall")
    print(f"{weighted_global_recall:.2%} of weighted global recall")

    precision_per_class = precision_score(
        y_true,
        y_pred,
        average=None,
        zero_division=0
    )
    recall_per_class = recall_score(
        y_true,
        y_pred,
        average=None,
        zero_division=0
    )
    class_counts = pd.Series(y_true).value_counts().sort_index()

    # Identify best and worst class based on precision
    best_class_idx = np.argmax(precision_per_class)
    worst_class_idx = np.argmin(precision_per_class)

    # Best and worst class precision and recall values
    best_class_precision = precision_per_class[best_class_idx]
    worst_class_precision = precision_per_class[worst_class_idx]

    best_class_recall = recall_per_class[best_class_idx]
    worst_class_recall = recall_per_class[worst_class_idx]

    best_class_count = class_counts[best_class_idx]
    worst_class_count = class_counts[worst_class_idx]

    # Print best and worst class details
    print(f"Best classified category: Class {best_class_idx}, Precision: {best_class_precision:.2%}, Recall: {best_class_recall:.2%}, Number of elements: {best_class_count}")
    print(f"Worst classified category: Class {worst_class_idx}, Precision: {worst_class_precision:.2%}, Recall: {worst_class_recall:.2%}, Number of elements: {worst_class_count}")
