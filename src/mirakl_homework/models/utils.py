from sklearn.metrics import accuracy_score


def dataframe_to_dict(df, features_cols):
    return {col: df[col].values for col in features_cols}


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
    print(f"{accuracy_score(y_true, y_pred):.2%} of global accuracy")
