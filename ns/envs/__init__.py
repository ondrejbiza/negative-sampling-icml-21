from gym.envs.registration import register
from ns.envs.block_pushing import BlockPushing

register(
    'ShapesTrain-v0',
    entry_point='ns.envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes'},
)

register(
    'ShapesEval-v0',
    entry_point='ns.envs.block_pushing:BlockPushing',
    max_episode_steps=10,
    kwargs={'render_type': 'shapes'},
)

register(
    'ShapesBgRandomTrain-v0',
    entry_point='ns.envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes', 'background': BlockPushing.BACKGROUND_RANDOM},
)

register(
    'ShapesBgRandomEval-v0',
    entry_point='ns.envs.block_pushing:BlockPushing',
    max_episode_steps=10,
    kwargs={'render_type': 'shapes', 'background': BlockPushing.BACKGROUND_RANDOM},
)

register(
    'ShapesBgSameEpTrain-v0',
    entry_point='ns.envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes', 'background': BlockPushing.BACKGROUND_RANDOM_SAME_EP},
)

register(
    'ShapesBgSameEpEval-v0',
    entry_point='ns.envs.block_pushing:BlockPushing',
    max_episode_steps=10,
    kwargs={'render_type': 'shapes', 'background': BlockPushing.BACKGROUND_RANDOM_SAME_EP},
)

register(
    'ShapesBgDeterministicTrain-v0',
    entry_point='ns.envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes', 'background': BlockPushing.BACKGROUND_DETERMINISTIC},
)

register(
    'ShapesBgDeterministicEval-v0',
    entry_point='ns.envs.block_pushing:BlockPushing',
    max_episode_steps=10,
    kwargs={'render_type': 'shapes', 'background': BlockPushing.BACKGROUND_DETERMINISTIC},
)

register(
    'ShapesCursorTrain-v0',
    entry_point='ns.envs.block_pushing_cursor:BlockPushingCursor',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes'},
)

register(
    'ShapesCursorEval-v0',
    entry_point='ns.envs.block_pushing_cursor:BlockPushingCursor',
    max_episode_steps=10,
    kwargs={'render_type': 'shapes'},
)

register(
    'ShapesImmovableTrain-v0',
    entry_point='ns.envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes', 'immovable': True},
)

register(
    'ShapesImmovableEval-v0',
    entry_point='ns.envs.block_pushing:BlockPushing',
    max_episode_steps=10,
    kwargs={'render_type': 'shapes', 'immovable': True},
)

register(
    'ShapesOppositeTrain-v0',
    entry_point='ns.envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes', 'opposite_direction': True},
)

register(
    'ShapesOppositeEval-v0',
    entry_point='ns.envs.block_pushing:BlockPushing',
    max_episode_steps=10,
    kwargs={'render_type': 'shapes', 'opposite_direction': True},
)

register(
    'ShapesImmovableFixedTrain-v0',
    entry_point='ns.envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes', 'immovable': True, 'immovable_fixed': True},
)

register(
    'ShapesImmovableFixedEval-v0',
    entry_point='ns.envs.block_pushing:BlockPushing',
    max_episode_steps=10,
    kwargs={'render_type': 'shapes', 'immovable': True, 'immovable_fixed': True},
)

register(
    'ShapesCursorImmovableTrain-v0',
    entry_point='ns.envs.block_pushing_cursor:BlockPushingCursor',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes', 'immovable': True},
)

register(
    'ShapesCursorImmovableEval-v0',
    entry_point='ns.envs.block_pushing_cursor:BlockPushingCursor',
    max_episode_steps=10,
    kwargs={'render_type': 'shapes', 'immovable': True},
)

register(
    'ShapesMetricO3C1Train-v0',
    entry_point='ns.envs.block_pushing_metric:BlockPushingMetric',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes', 'num_objects': 3, 'num_colors': 1},
)

register(
    'ShapesMetricO3C1Eval-v0',
    entry_point='ns.envs.block_pushing_metric:BlockPushingMetric',
    max_episode_steps=10,
    kwargs={'render_type': 'shapes', 'num_objects': 3, 'num_colors': 1},
)

register(
    'ShapesMetricO3C2Train-v0',
    entry_point='ns.envs.block_pushing_metric:BlockPushingMetric',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes', 'num_objects': 3, 'num_colors': 2,
            'background': BlockPushing.BACKGROUND_DETERMINISTIC},
)

register(
    'ShapesMetricO3C2Eval-v0',
    entry_point='ns.envs.block_pushing_metric:BlockPushingMetric',
    max_episode_steps=10,
    kwargs={'render_type': 'shapes', 'num_objects': 3, 'num_colors': 2,
            'background': BlockPushing.BACKGROUND_DETERMINISTIC},
)

register(
    'ShapesMetricO2C1Train-v0',
    entry_point='ns.envs.block_pushing_metric:BlockPushingMetric',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes', 'num_objects': 2, 'num_colors': 1},
)

register(
    'ShapesMetricO2C1Eval-v0',
    entry_point='ns.envs.block_pushing_metric:BlockPushingMetric',
    max_episode_steps=10,
    kwargs={'render_type': 'shapes', 'num_objects': 2, 'num_colors': 1},
)

register(
    'ShapesMetricO2C5Train-v0',
    entry_point='ns.envs.block_pushing_metric:BlockPushingMetric',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes', 'num_objects': 2, 'num_colors': 5,
            'background': BlockPushing.BACKGROUND_DETERMINISTIC},
)

register(
    'ShapesMetricO2C5Eval-v0',
    entry_point='ns.envs.block_pushing_metric:BlockPushingMetric',
    max_episode_steps=10,
    kwargs={'render_type': 'shapes', 'num_objects': 2, 'num_colors': 5,
            'background': BlockPushing.BACKGROUND_DETERMINISTIC},
)

register(
    'ShapesSameCursorTrain-v0',
    entry_point='ns.envs.block_pushing_cursor:BlockPushingCursor',
    max_episode_steps=100,
    kwargs={'render_type': 'shapes', 'same_shape_and_color': True},
)

register(
    'ShapesSameCursorEval-v0',
    entry_point='ns.envs.block_pushing_cursor:BlockPushingCursor',
    max_episode_steps=10,
    kwargs={'render_type': 'shapes', 'same_shape_and_color': True},
)

register(
    'CubesTrain-v0',
    entry_point='ns.envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'cubes'},
)

register(
    'CubesEval-v0',
    entry_point='ns.envs.block_pushing:BlockPushing',
    max_episode_steps=10,
    kwargs={'render_type': 'cubes'},
)

register(
    'CubesBgRandomTrain-v0',
    entry_point='ns.envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'cubes', 'background': BlockPushing.BACKGROUND_RANDOM},
)

register(
    'CubesBgRandomEval-v0',
    entry_point='ns.envs.block_pushing:BlockPushing',
    max_episode_steps=10,
    kwargs={'render_type': 'cubes', 'background': BlockPushing.BACKGROUND_RANDOM},
)

register(
    'CubesBgSameEpTrain-v0',
    entry_point='ns.envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'cubes', 'background': BlockPushing.BACKGROUND_RANDOM_SAME_EP},
)

register(
    'CubesBgSameEpEval-v0',
    entry_point='ns.envs.block_pushing:BlockPushing',
    max_episode_steps=10,
    kwargs={'render_type': 'cubes', 'background': BlockPushing.BACKGROUND_RANDOM_SAME_EP},
)

register(
    'CubesBgDeterministicTrain-v0',
    entry_point='ns.envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'cubes', 'background': BlockPushing.BACKGROUND_DETERMINISTIC},
)

register(
    'CubesBgDeterministicEval-v0',
    entry_point='ns.envs.block_pushing:BlockPushing',
    max_episode_steps=10,
    kwargs={'render_type': 'cubes', 'background': BlockPushing.BACKGROUND_DETERMINISTIC},
)

register(
    'CubesCursorTrain-v0',
    entry_point='ns.envs.block_pushing_cursor:BlockPushingCursor',
    max_episode_steps=100,
    kwargs={'render_type': 'cubes'},
)

register(
    'CubesCursorEval-v0',
    entry_point='ns.envs.block_pushing_cursor:BlockPushingCursor',
    max_episode_steps=10,
    kwargs={'render_type': 'cubes'},
)

register(
    'CubesImmovableTrain-v0',
    entry_point='ns.envs.block_pushing:BlockPushing',
    max_episode_steps=100,
    kwargs={'render_type': 'cubes', 'immovable': True},
)

register(
    'CubesImmovableEval-v0',
    entry_point='ns.envs.block_pushing:BlockPushing',
    max_episode_steps=10,
    kwargs={'render_type': 'cubes', 'immovable': True},
)
