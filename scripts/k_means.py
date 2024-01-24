from functools import partial
import jax
import jax.numpy as jnp

@jax.jit
def vector_quantize(points, codebook):
    assignment = jax.vmap(
        lambda point: jnp.argmin(jax.vmap(jnp.linalg.norm)(codebook - point))
    )(points)
    distns = jax.vmap(jnp.linalg.norm)(codebook[assignment,:] - points)
    return assignment, distns

@partial(jax.jit, static_argnums=(2,))
def kmeans_run(key, points, k, thresh=1e-5):

    def improve_centroids(val):
        prev_centroids, prev_distn, _ = val
        assignment, distortions = vector_quantize(points, prev_centroids)

        # Count number of points assigned per centroid
        # (Thanks to Jean-Baptiste Cordonnier for pointing this way out that is
        # much faster and let's be honest more readable!)
        counts = (
            (assignment[jnp.newaxis, :] == jnp.arange(k)[:, jnp.newaxis])
            .sum(axis=1, keepdims=True)
            .clip(min=1.)  # clip to change 0/0 later to 0/1
        )

        # Sum over points in a centroid by zeroing others out
        new_centroids = jnp.sum(
            jnp.where(
                # axes: (data points, clusters, data dimension)
                assignment[:, jnp.newaxis, jnp.newaxis] \
                    == jnp.arange(k)[jnp.newaxis, :, jnp.newaxis],
                points[:, jnp.newaxis, :],
                0.,
            ),
            axis=0,
        ) / counts

        return new_centroids, jnp.mean(distortions), prev_distn

    # Run one iteration to initialize distortions and cause it'll never hurt...
    initial_indices = jax.random.shuffle(key, jnp.arange(points.shape[0]))[:k]
    initial_val = improve_centroids((points[initial_indices, :], jnp.inf, None))
    # ...then iterate until convergence!
    centroids, distortion, _ = jax.lax.while_loop(
        lambda val: (val[2] - val[1]) > thresh,
        improve_centroids,
        initial_val,
    )
    return centroids, distortion

@partial(jax.jit, static_argnums=(2,3))
def kmeans(key, points, k, restarts, **kwargs):
    all_centroids, all_distortions = jax.vmap(
        lambda key: kmeans_run(key, points, k, **kwargs)
    )(jax.random.split(key, restarts))
    i = jnp.argmin(all_distortions)
    return all_centroids[i], all_distortions[i]


def kmeans_init(model,emissions,key,ar):
    key1, key2 = jax.random.split(key, 2)
    centers, _ = kmeans(key1, emissions.reshape(-1, model.emission_dim), model.num_states, 1)
    if ar:
        params, props = model.initialize(
            key=key2, method="prior", emissions=emissions, emission_biases=jnp.array(centers),
        )
    else:
        params, props = model.initialize(
            key=key2, method="prior", emissions=emissions, emission_means=jnp.array(centers)
        )
    return params, props
