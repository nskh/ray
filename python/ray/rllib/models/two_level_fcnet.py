from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from ray.rllib.models.model import Model
from ray.rllib.models.fcnet import FullyConnectedNetwork


class TwoLevelFCNetwork(Model):
    """
    Two-level fully connected network, consisting of a number of
    subpolicies and a pre-specified function which chooses among the
    subpolicies.
    """

    def _init(self, inputs, num_outputs, options):
        num_subpolicies = options.get("num_subpolicies", 1)
        subhiddens = options.get("fcnet_hiddens", [[256, 256]] * 1)
        fcnet_activation = options.get("fcnet_activation", "tanh")
        if fcnet_activation == "tanh":
            activation = tf.nn.tanh
        elif fcnet_activation == "relu":
            activation = tf.nn.relu
        print("Constructing two level fcnet {} {}".format(subhiddens,
                                                         activation))

        # function which maps from observation to subpolicy observation
        to_subpolicy_state = options.get("to_subpolicy_state", lambda x, k: x)
        # function which maps from observation to choice of subpolicy
        choose_policy = options.get("choose_policy", lambda x: 0)

        attention = tf.one_hot(choose_policy(inputs), num_subpolicies)

        outputs = []
        for k in range(num_subpolicies):
            sub_options = options.copy()
            sub_options.update({"fcnet_hiddens": subhiddens[k]})
            sub_options.update({"fcnet_tag": k})
            subinput = to_subpolicy_state(inputs, k)
            fcnet = FullyConnectedNetwork(
                subinput, num_outputs, sub_options)
            output, last_layer = fcnet.outputs, fcnet.last_layer
            outputs.append(attention[k] * output)
        overall_output = tf.add_n(outputs)
        # TODO(cathywu) check that outputs is not used later on because it's
        # a list instead of a layer
        return overall_output, outputs
