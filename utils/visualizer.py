# try:
#     import graphviz
# except ImportError:
#     graphviz = None
import os
import graphviz


def export_tree_to_image(
    root_node, output_dir="output", filename="tree_viz", feature_names=None
):
    """
    Generates a PNG image of the decision tree using Graphviz.
    """
    if graphviz is None:
        print("\n[Warning] Graphviz library not installed. Skipping visualization.")
        print("Please run: pip install graphviz")
        return

    if root_node is None:
        print("Tree has not been trained yet. Nothing to visualize.")
        return

    # Initialize graphviz object
    dot = graphviz.Digraph(comment="C4.5 Decision Tree")

    # A mutable list counter to assign unique IDs to every node in the image
    node_counter = [0]

    def _traverse_and_draw(node, parent_id=None, edge_label=""):
        """Recursive helper to draw nodes and edges."""
        # 1. Assign a unique ID for the graph image
        current_id = str(node_counter[0])
        node_counter[0] += 1

        # 2. Determine Node Label and Appearance
        if node.is_leaf:
            # It's an answer node
            label = f"Prediction:\nClass {node.value}"
            shape = "box"
            color = "#90EE90"  # Light green
        else:
            # It's a question node
            # Figure out the name of the feature being split
            f_name = f"Column {node.feature_index}"
            if feature_names and node.feature_index < len(feature_names):
                f_name = feature_names[node.feature_index]

            label = f"{f_name}\n<= {node.threshold:.3f}?"
            shape = "ellipse"
            color = "#ADD8E6"  # Light blue

        # Add node to graph
        dot.node(current_id, label, shape=shape, style="filled", fillcolor=color)

        # 3. Draw Connection from Parent (if exists)
        if parent_id is not None:
            # Connection from parent to current node
            dot.edge(parent_id, current_id, label=edge_label)

        # 4. Recurse to Childern
        if not node.is_leaf:
            # Left child implies True (<= threshold)
            _traverse_and_draw(node.childern["left"], current_id, "True (<=)")
            # Right child implies False (> threshold)
            _traverse_and_draw(node.childern["right"], current_id, "False (>)")

    # --- Start the recursive drawing process ---
    _traverse_and_draw(root_node)

    # --- Save the file ---
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, filename)

    try:
        # This generates both a source file and a .png image
        dot.render(full_path, view=False, format="png", cleanup=True)
        print(f"\n[Success] Tree visualization saved to: {full_path}.png")
    except graphviz.backend.ExecutableNotFound:
        print("\n[Error] Graphviz system executable not found.")
        print("The Python library is installed, but the Graphviz tool is not.")
        print(
            "Please install it from https://graphviz.org/download/ and add it to your PATH."
        )


def export_tree_to_dot(root_node, filepath, feature_names=None):
    if root_node is None:
        return
    # generates dot
    dot_lines = [
        "digraph Tree {",
        'node [shape=box, fontname="helvetica"];',
        'edge [fontname="helvetica"];',
    ]
    node_counter = [0]

    def _traverse(node, parent_id=None, edge_label=""):
        my_id = node_counter[0]
        node_counter[0] += 1

        # Label logic
        if node.is_leaf:
            label = f"Prediction:\\nClass {node.value}"
            color = "#90EE90"  # Green
        else:
            fname = f"Col {node.feature_index}"
            if feature_names and node.feature_index < len(feature_names):
                fname = feature_names[node.feature_index]
            label = f"{fname}\\n<= {node.threshold:.3f}?"
            color = "#ADD8E6"  # Blue

        dot_lines.append(
            f'{my_id} [label="{label}", style="filled", fillcolor="{color}"];'
        )

        if parent_id is not None:
            dot_lines.append(f'{parent_id} -> {my_id} [label="{edge_label}"];')

        if not node.is_leaf:
            _traverse(node.childern["left"], my_id, "True")
            _traverse(node.childern["right"], my_id, "False")

    _traverse(root_node)
    dot_lines.append("}")

    # generate rules
    rules = ["\n\n--- RULES RESULT ---"]

    def _print_rules(node, spacing=""):
        if node.is_leaf:
            rules.append(f"{spacing}THEN Prediction = Class {node.value}")
            return

        fname = f"Column {node.feature_index}"
        if feature_names and node.feature_index < len(feature_names):
            fname = feature_names[node.feature_index]

        # Left Child (True)
        rules.append(f"{spacing}IF {fname} <= {node.threshold:.3f}:")
        _print_rules(node.childern["left"], spacing + "    ")

        # Right Child (False)
        rules.append(f"{spacing}IF {fname} > {node.threshold:.3f}:")
        _print_rules(node.childern["right"], spacing + "    ")

    _print_rules(root_node)

    # save as file
    with open(filepath, "w") as f:
        f.write("\n".join(dot_lines))
        f.write("\n".join(rules))
