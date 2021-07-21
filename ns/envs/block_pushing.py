"""Gym environment for block pushing tasks (2D Shapes and 3D Cubes)."""

import numpy as np

import ns.utils as utils
import gym
from gym import spaces
from gym.utils import seeding

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image


import skimage


def square(r0, c0, width, im_size):
    rr, cc = [r0, r0 + width, r0 + width, r0], [c0, c0, c0 + width, c0 + width]
    return skimage.draw.polygon(rr, cc, im_size)


def triangle(r0, c0, width, im_size):
    rr, cc = [r0, r0 + width, r0 + width], [c0 + width//2, c0, c0 + width]
    return skimage.draw.polygon(rr, cc, im_size)


def fig2rgb_array(fig):
    fig.canvas.draw()
    buffer = fig.canvas.tostring_rgb()
    width, height = fig.canvas.get_width_height()
    return np.fromstring(buffer, dtype=np.uint8).reshape(height, width, 3)


def render_cubes(positions, width, background_color=None):
    voxels = np.zeros((width, width, width), dtype=np.bool)
    colors = np.empty(voxels.shape, dtype=object)

    cols = ['purple', 'green', 'orange', 'blue', 'brown']

    for i, pos in enumerate(positions):
        voxels[pos[0], pos[1], 0] = True
        colors[pos[0], pos[1], 0] = cols[i]

    if background_color is None:
        background_color = (0.5, 0.5, 0.5, 1.0)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.w_zaxis.set_pane_color(background_color)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.line.set_lw(0.)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.voxels(voxels, facecolors=colors, edgecolor='k')

    im = fig2rgb_array(fig)
    plt.close(fig)
    im = np.array(  # Crop and resize
        Image.fromarray(im[215:455, 80:570]).resize((50, 50), Image.ANTIALIAS))
    return im / 255.


class BlockPushing(gym.Env):
    """Gym environment for block pushing task."""

    DIRECTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    BACKGROUND_WHITE = 0
    BACKGROUND_RANDOM = 1
    BACKGROUND_RANDOM_SAME_EP = 2
    BACKGROUND_DETERMINISTIC = 3

    def __init__(self, width=5, height=5, render_type='cubes', num_objects=5,
                 seed=None, immovable=False, immovable_fixed=False, opposite_direction=False,
                 background=BACKGROUND_WHITE, num_colors=5, same_shape_and_color=False):
        self.width = width
        self.height = height
        self.render_type = render_type
        self.immovable = immovable
        self.immovable_fixed = immovable_fixed
        self.opposite_direction = opposite_direction
        self.same_shape_and_color = same_shape_and_color

        self.num_objects = num_objects
        self.num_actions = 4 * self.num_objects  # Move NESW
        self.num_colors = num_colors

        self.colors = utils.get_colors(num_colors=max(9, self.num_objects))
        self.background = background
        # used only if background != BACKGROUND_WHITE
        self.background_colors = utils.get_colors(cmap="Set2", num_colors=num_colors)
        # used only if background in [BACKGROUND_RANDOM_SAME_EP, BACKGROUND_DETERMINISTIC]
        self.background_index = 0

        self.np_random = None
        self.game = None
        self.target = None

        # Initialize to pos outside of env for easier collision resolution.
        self.objects = [[-1, -1] for _ in range(self.num_objects)]

        # If True, then check for collisions and don't allow two
        #   objects to occupy the same position.
        self.collisions = True

        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(3, self.width, self.height),
            dtype=np.float32
        )

        self.seed(seed)
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self):
        # background color is only implemented for some render types
        assert self.background == self.BACKGROUND_WHITE or self.render_type in ["shapes", "cubes"]
        # same shapes only implemented for shapes
        assert not self.same_shape_and_color or self.render_type == "shapes"

        if self.render_type == 'grid':
            im = np.zeros((3, self.width, self.height))
            for idx, pos in enumerate(self.objects):
                im[:, pos[0], pos[1]] = self.colors[idx][:3]
            return im
        elif self.render_type == 'circles':
            im = np.zeros((self.width*10, self.height*10, 3), dtype=np.float32)
            for idx, pos in enumerate(self.objects):
                rr, cc = skimage.draw.circle(
                    pos[0]*10 + 5, pos[1]*10 + 5, 5, im.shape)
                im[rr, cc, :] = self.colors[idx][:3]
            return im.transpose([2, 0, 1])
        elif self.render_type == 'shapes':
            im = np.zeros((self.width*10, self.height*10, 3), dtype=np.float32)
            im = self.set_background_color_shapes_(im)

            if self.same_shape_and_color:

                for pos in self.objects:

                    rr, cc = skimage.draw.circle(
                        pos[0] * 10 + 5, pos[1] * 10 + 5, 5, im.shape)
                    im[rr, cc, :] = self.colors[0][:3]

            else:

                for idx, pos in enumerate(self.objects):
                    if idx % 3 == 0:
                        rr, cc = skimage.draw.circle(
                            pos[0]*10 + 5, pos[1]*10 + 5, 5, im.shape)
                        im[rr, cc, :] = self.colors[idx][:3]
                    elif idx % 3 == 1:
                        rr, cc = triangle(
                            pos[0]*10, pos[1]*10, 10, im.shape)
                        im[rr, cc, :] = self.colors[idx][:3]
                    else:
                        rr, cc = square(
                            pos[0]*10, pos[1]*10, 10, im.shape)
                        im[rr, cc, :] = self.colors[idx][:3]

            return im.transpose([2, 0, 1])
        elif self.render_type == 'cubes':
            im = render_cubes(self.objects, self.width, background_color=self.get_background_color_())
            return im.transpose([2, 0, 1])

    def get_background_color_(self):

        if self.background == self.BACKGROUND_WHITE:
            # on a second thought, the default background is actually black or gray
            return None
        elif self.background == self.BACKGROUND_RANDOM:
            idx = np.random.randint(len(self.background_colors))
            return self.background_colors[idx]
        elif self.background in [self.BACKGROUND_RANDOM_SAME_EP, self.BACKGROUND_DETERMINISTIC]:
            return self.background_colors[self.background_index]
        else:
            assert False

    def set_background_color_shapes_(self, im):

        color = self.get_background_color_()

        if color is not None:
            for i in range(3):
                im[:, :, i] = color[i]

        return im

    def get_state(self):
        im = np.zeros(
            (self.num_objects, self.width, self.height), dtype=np.int32)
        for idx, pos in enumerate(self.objects):
            im[idx, pos[0], pos[1]] = 1
        return im

    def reset(self):

        self.reset_background_color_()
        self.reset_objects_()

        state_obs = (self.get_state(), self.render())
        return state_obs

    def reset_background_color_(self):

        if self.background == self.BACKGROUND_RANDOM_SAME_EP:
            self.background_index = np.random.randint(len(self.background_colors))
        elif self.background == self.BACKGROUND_DETERMINISTIC:
            self.background_index = 0

    def reset_objects_(self):

        self.objects = [[-1, -1] for _ in range(self.num_objects)]

        if self.immovable_fixed:
            # the first two object have a fixed position
            self.objects[0] = [1, 2]
            self.objects[1] = [4, 4]
            assert self.valid_pos(self.objects[0], 0)
            assert self.valid_pos(self.objects[1], 1)

        # Randomize object position.
        for i in range(len(self.objects)):

            if self.immovable_fixed and i < 2:
                continue

            # Resample to ensure objects don't fall on same spot.
            while not self.valid_pos(self.objects[i], i):
                self.objects[i] = [
                    np.random.choice(np.arange(self.width)),
                    np.random.choice(np.arange(self.height))
                ]

    def valid_pos(self, pos, obj_id):
        """Check if position is valid."""
        if pos[0] < 0 or pos[0] >= self.width:
            return False
        if pos[1] < 0 or pos[1] >= self.height:
            return False

        if self.collisions:
            for idx, obj_pos in enumerate(self.objects):
                if idx == obj_id:
                    continue

                if pos[0] == obj_pos[0] and pos[1] == obj_pos[1]:
                    return False

        return True

    def valid_move(self, obj_id, offset):
        """Check if move is valid."""
        old_pos = self.objects[obj_id]
        new_pos = [p + o for p, o in zip(old_pos, offset)]
        return self.valid_pos(new_pos, obj_id)

    def translate(self, obj_id, offset):
        """"Translate object pixel.

        Args:
            obj_id: ID of object.
            offset: (x, y) tuple of offsets.
        """

        # the first two objects are immovable
        if self.immovable:
            if obj_id in [0, 1]:
                return False

        if self.opposite_direction:
            if obj_id in [0, 1]:
                offset = (-offset[0], -offset[1])

        if not self.valid_move(obj_id, offset):
            return False

        self.objects[obj_id][0] += offset[0]
        self.objects[obj_id][1] += offset[1]

        return True

    def step(self, action):

        _, reward, done, _ = self.step_no_render(action)

        state_obs = (self.get_state(), self.render())

        return state_obs, reward, done, None

    def step_no_render(self, action):

        self.step_background_color_()

        done = False
        reward = 0

        direction = action % 4
        obj = action // 4
        self.translate(obj, self.DIRECTIONS[direction])

        return None, reward, done, None

    def step_background_color_(self):

        if self.background == self.BACKGROUND_DETERMINISTIC:
            self.background_index = (self.background_index + 1) % len(self.background_colors)

    def get_state_id(self):

        state_id = []

        for obj in self.objects:
            state_id.append(obj[0])
            state_id.append(obj[1])

        return np.array(state_id)
