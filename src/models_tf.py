"""Optional Keras models for the grouped ANN comparison."""

import tensorflow as tf


def create_tf_ann(n_features, hidden_layers=(10,), activation="tanh",
                  learning_rate=0.001, l2=0.0001, seed=42):
    """Build a fresh network; inputs and targets are scaled by the caller."""
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    model = tf.keras.Sequential([tf.keras.Input(shape=(n_features,))])
    for units in hidden_layers:
        model.add(tf.keras.layers.Dense(
            units, activation=activation,
            kernel_regularizer=tf.keras.regularizers.l2(l2),
        ))
    model.add(tf.keras.layers.Dense(1))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse", metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model
